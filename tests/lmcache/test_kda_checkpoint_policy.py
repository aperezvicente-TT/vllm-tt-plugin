# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""KDA checkpoint policy (page_contract.md §5) on the mock: ExampleConnector through
the real TTScheduler (chunked prefill, 64-token alignment) and the real runner wiring,
with a mock backend exposing 14b's ``export_kda_state(kv, slot)`` /
``import_kda_state(kv, slot, states)``.

Policy under test: a prompt's prefill stops on its last page boundary and checkpoints
the slot's KDA state there; the tail runs as the next chunk; a prefix hit of N tokens
imports the N-token checkpoint into the request's slot before the continuation; a hit
whose boundary has no checkpoint is truncated to the longest checkpointed boundary.
No device, no LMCache (venv-vllm is enough)."""

from __future__ import annotations

import hashlib
import os
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_tt_plugin.kv_connector as kvc
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

import vllm_tt_plugin  # noqa: F401
from tests.test_kv_connector import (
    BLOCK,
    LATENT,
    LAYER_NAMES,
    _layer_index,
    _prompt,
    _request,
    fingerprint,
)
from vllm_tt_plugin.config import store_tt_output_tokens_per_step
from vllm_tt_plugin.kv_connector import (
    KDACheckpointStore,
    TTKVConnectorHost,
    TTPageAlignedConnector,
    validate_tt_kv_transfer_config,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler



@pytest.fixture(autouse=True)
def _tail_split_on(monkeypatch):
    """the KDA-checkpoint tail split on short prompts is what these tests check: GLM53_PF_TAIL_SPLIT_MIN off (its
    behaviour has its own tests, tests/test_tail_split_min.py)"""
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_MIN", "0")


@pytest.fixture(autouse=True)
def _sync_checkpoint_writes(monkeypatch):
    """these tests move / delete checkpoint files right after a save: synchronous writes, no RAM tier (the async writer has its own
    tests, test_kda_ckpt_store_async.py)"""
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_RAM_GB", "0")  # disk semantics: no RAM tier behind the files
    monkeypatch.setenv("GLM53_KV_SAVE_DEFER", "0")  # the export inside its step (the deferral: test_kda_ckpt_store_async)

class _ChunkedStubModel:
    # The platform keeps chunked prefill only for models that declare it.
    model_capabilities = {"supports_chunked_prefill": True}


@contextmanager
def _stub_model_resolution():
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(_ChunkedStubModel, None),
        ),
    ):
        yield


NUM_BLOCKS = 48
MAX_MODEL_LEN = 2048
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"


def _state_key(tokens: list[int]) -> int:
    return int.from_bytes(
        hashlib.sha256(bytes(t % 251 for t in tokens)).digest()[:7], "little"
    )


class _MockKDABackend:
    """Per-slot recurrent state = (tokens seen, hash of them); 14b's two calls."""

    def __init__(self):
        self.slot_state: dict[int, tuple[int, int]] = {}
        self.exports: list[tuple[int, int]] = []
        self.imports: list[tuple[int, int]] = []

    def export_kda_state(self, kv, slot):
        n, key = self.slot_state[slot]
        self.exports.append((slot, n))
        return [
            {
                "L": 0,
                "S": torch.tensor([float(n)]),
                "C": torch.tensor([key], dtype=torch.int64),
            }
        ]

    def import_kda_state(self, kv, slot, states):
        n, key = int(states[0]["S"][0]), int(states[0]["C"][0])
        self.slot_state[slot] = (n, key)
        self.imports.append((slot, n))


def _config(storage: str, kda_dir: str, align=None) -> tuple[VllmConfig, KVCacheConfig]:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": storage,
            "tt_kda_checkpoint_dir": kda_dir,
        },
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            kv_transfer_config=kv_transfer_config,
            additional_config={"tt": {"prefill_chunk_alignment": align or BLOCK}},
        )
    validate_tt_kv_transfer_config(config)
    store_tt_output_tokens_per_step(config, 1)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    spec = MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=LATENT, dtype=torch.bfloat16
    )
    return config, KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(LAYER_NAMES), spec)],
    )


class _MockRunner:
    """Real execute_model / sample_tokens; host stubs that track slots and KDA state."""

    def __init__(
        self,
        connector,
        kv_cache_config,
        backend: _MockKDABackend,
        store: KDACheckpointStore,
    ):
        self.caches = {
            n: torch.zeros(NUM_BLOCKS, 1, BLOCK, LATENT, dtype=torch.bfloat16)
            for n in LAYER_NAMES
        }
        self.backend = backend
        self.slots: dict[str, int] = {}
        r = TTModelRunner.__new__(TTModelRunner)
        r._device_grammar_poisoned = False
        r.async_decode_scheduling = False
        r._pending_samples = deque()
        r.async_decode = SimpleNamespace(
            can_attempt_steady_decode_from_scheduler=lambda so: False,
            must_drain_pending_async_steps=lambda *_: False,
            wait_for_all_pending_async_steps=lambda: None,
        )
        r.kv_cache_config = kv_cache_config
        r.kv_caches = self.caches
        r._layer_to_group_idx = None
        r.parallel_config = SimpleNamespace(data_parallel_size=1)
        r.tt_data_parallel_size = 1
        r._num_speculative_tokens = 0
        r.model = backend
        r.build_model_input = self._build_model_input
        r._forward_with_model_input = self._forward
        r._finish_front_packed_sync = self._finish
        r.kv_connector_host = TTKVConnectorHost(
            connector,
            kv_cache_config,
            self.caches,
            model=backend,
            kda_store=store,
            slot_of=lambda rid: self.slots.get(rid),
        )
        self.runner = r
        self.blocks: dict[str, list[int]] = {}
        self.tokens: dict[str, list[int]] = {}
        self.computed: dict[str, int] = {}
        self.forward_rows: list[tuple[str, int, int]] = []

    def _build_model_input(self, so: SchedulerOutput, grammar_output):
        for nr in so.scheduled_new_reqs:
            self.blocks[nr.req_id] = list(nr.block_ids[0])
            self.tokens[nr.req_id] = list(nr.prompt_token_ids or [])
            self.computed[nr.req_id] = nr.num_computed_tokens
            self.slots[nr.req_id] = min(set(range(8)) - set(self.slots.values()))
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            if cr.new_block_ids[i] is not None:
                self.blocks[rid].extend(cr.new_block_ids[i][0])
            if i < len(cr.new_token_ids) and cr.new_token_ids[i]:
                self.tokens[rid].extend(cr.new_token_ids[i])
            self.computed[rid] = cr.num_computed_tokens[i]
        for rid in so.finished_req_ids:
            self.slots.pop(rid, None)
        if not so.total_num_scheduled_tokens:
            return None
        rows = list(so.num_scheduled_tokens)
        bt = torch.zeros(len(rows), MAX_MODEL_LEN // BLOCK, dtype=torch.int32)
        for i, rid in enumerate(rows):
            bt[i, : len(self.blocks[rid])] = torch.tensor(
                self.blocks[rid], dtype=torch.int32
            )
        starts = torch.tensor([self.computed[r] for r in rows])
        ends = torch.tensor(
            [self.computed[r] + so.num_scheduled_tokens[r] for r in rows]
        )
        is_prefill = any(self.computed[r] < len(self.tokens[r]) for r in rows)
        toks = torch.zeros(len(rows), int(ends.max()), dtype=torch.int32)
        for i, rid in enumerate(rows):
            t = self.tokens[rid][: int(ends[i])]
            toks[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
        return SimpleNamespace(
            row_req_ids=rows,
            input_positions=starts,
            prompt_lens=ends if is_prefill else None,
            block_tables=bt,
            block_tables_per_group=[bt],
            input_tokens=toks,
            defer_device_sampling=False,
            spec_mode=None,
            prefill_empty_slots=None,
        )

    def _forward(self, model_input):
        for i, rid in enumerate(model_input.row_req_ids):
            start = int(model_input.input_positions[i])
            end = (
                int(model_input.prompt_lens[i])
                if model_input.prompt_lens is not None
                else start + 1
            )
            self.forward_rows.append((rid, start, end))
            slot = self.slots[rid]
            toks = model_input.input_tokens[i].tolist()
            if start > 0:
                # A continuation must start from the state after exactly `start` tokens.
                assert self.backend.slot_state.get(slot) == (
                    start,
                    _state_key(toks[:start]),
                ), (rid, start, self.backend.slot_state.get(slot))
            self.backend.slot_state[slot] = (end, _state_key(toks[:end]))
            for name, cache in self.caches.items():
                li = _layer_index(name)
                for pos in range(start, end):
                    blk = int(model_input.block_tables[i, pos // BLOCK])
                    cache[blk, 0, pos % BLOCK] = fingerprint(
                        li, pos, int(model_input.input_tokens[i, pos])
                    )
        return SimpleNamespace(model_input=model_input)

    def _finish(self, grammar_output, *, fwd):
        mi = fwd.model_input
        ids = list(mi.row_req_ids)
        # Like the runner's chunked-prefill output: an intermediate chunk (more
        # prompt tokens follow) emits no token.
        sampled = []
        for i, rid in enumerate(ids):
            end = int(mi.prompt_lens[i]) if mi.prompt_lens is not None else None
            sampled.append(
                [] if end is not None and end < len(self.tokens[rid]) else [5]
            )
        return ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def step(self, so):
        out = self.runner.execute_model(so)
        if out is None:
            out = self.runner.sample_tokens(None)
        return out


def test_kda_checkpoint_policy_save_at_boundary_import_on_hit_truncate_without(
    tmp_path,
):
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    assert isinstance(scheduler.connector, TTPageAlignedConnector)
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    worker = ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock = _MockRunner(worker, kv_cache_config, backend, store)
    toks = _prompt(200)

    # ---- A: 200-token prompt -> chunk 1 ends on the page boundary 192 and checkpoints
    scheduler.add_request(_request("A", toks))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 192}, (
        "prefill stops on the last page boundary"
    )
    out = mock.step(so)
    scheduler.update_from_output(so, out)
    assert mock.forward_rows == [("A", 0, 192)]
    assert backend.exports == [(mock.slots["A"], 192)]
    assert store.has(toks, 192) and store.load(toks, 192)[0]["S"][0] == 192
    # chunk 2: the 8-token tail; no checkpoint (unaligned end)
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 8}
    out = mock.step(so)
    scheduler.update_from_output(so, out)
    assert mock.forward_rows[-1] == ("A", 192, 200) and len(backend.exports) == 1
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 1}  # decode
    scheduler.update_from_output(so, mock.step(so))
    scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
    so_idle = scheduler.schedule()
    mock.step(so_idle)
    slot_a = mock.slots.get("A")
    assert slot_a is None

    # ---- B: A's prompt + 1 -> 192-token hit; checkpoint imported before the tail
    toks_b = toks + [9]
    scheduler.add_request(_request("B", toks_b))
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
    assert nr.num_computed_tokens == 192 and so.num_scheduled_tokens == {"B": 9}
    for t in mock.caches.values():
        t.zero_()
    mock.forward_rows.clear()
    out = mock.step(so)
    assert backend.imports == [(mock.slots["B"], 192)]
    assert mock.forward_rows == [
        ("B", 192, 201)
    ]  # the forward's own assert checked the state
    blocks_b = list(nr.block_ids[0])
    for name, cache in mock.caches.items():
        li = _layer_index(name)
        for pos in range(201):
            assert torch.equal(
                cache[blocks_b[pos // BLOCK], 0, pos % BLOCK],
                fingerprint(li, pos, toks_b[pos]),
            ), (name, pos)
    scheduler.update_from_output(so, out)
    scheduler.finish_requests(["B"], RequestStatus.FINISHED_ABORTED)
    mock.step(scheduler.schedule())

    # ---- C: same prompt, checkpoint deleted -> hit truncated to 0 (none left). C RECOMPUTES pages LMCache already
    # holds (A's computation: LMCache keeps the first writer), so its state at 192 need not match those pages: C
    # exports no checkpoint (Stage 12 P1-1, _KDA_PROVENANCE), a later hit stays shorter instead of mismatched
    saved192 = store.load(toks, 192)
    os.remove(store.path(toks, 192))
    store._ram.clear()  # the RAM tier too: the checkpoint is gone
    exports_before = len(backend.exports)
    scheduler.add_request(_request("C", toks_b))
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "C")
    assert nr.num_computed_tokens == 0 and so.num_scheduled_tokens == {"C": 192}
    assert "C" in kvc._KDA_PROVENANCE
    mock.forward_rows.clear()
    out = mock.step(so)
    assert mock.forward_rows == [("C", 0, 192)] and len(backend.exports) == exports_before
    assert not store.has(toks, 192)
    scheduler.update_from_output(so, out)
    scheduler.finish_requests(["C"], RequestStatus.FINISHED_ABORTED)
    mock.step(scheduler.schedule())
    assert "C" not in kvc._KDA_PROVENANCE  # cleared when the request finished

    # ---- D: with only a shorter checkpoint present the hit stops there
    store.save(toks, 128, saved192)  # a (fake) 128-token checkpoint ...
    store._ram.clear()
    backend.slot_state[99] = (128, _state_key(toks[:128]))
    store.save(toks, 128, backend.export_kda_state(None, 99))
    scheduler.add_request(_request("D", toks_b))
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "D")
    assert nr.num_computed_tokens == 128 and so.num_scheduled_tokens == {"D": 64}, (
        "192 -> 128 (last checkpoint)"
    )
    mock.forward_rows.clear()
    out = mock.step(so)
    assert mock.forward_rows == [("D", 128, 192)] and backend.imports[-1] == (
        mock.slots["D"],
        128,
    )
    scheduler.update_from_output(so, out)


def test_kda_checkpoint_policy_page_aligned_prompt_checkpoints_at_len_minus_page(
    tmp_path, monkeypatch
):
    """J7 (GLM53_KDA_CKPT_AT_END=0): a k-page prompt splits at floor64(len - 1) = len - 64, the hit length."""
    monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", "0")
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    mock = _MockRunner(
        ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config),
        kv_cache_config,
        backend,
        store,
    )
    toks = _prompt(192, seed=11)
    scheduler.add_request(_request("A", toks))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 128}, (
        "split at floor64(191) = 128, not at 192"
    )
    scheduler.update_from_output(so, mock.step(so))
    assert backend.exports == [(mock.slots["A"], 128)] and store.has(toks, 128)
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 64}
    scheduler.update_from_output(so, mock.step(so))
    scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
    mock.step(scheduler.schedule())
    # B = the same 192-token prompt: ExampleConnector reports align64(191) = 128
    # (its store key is align64(len): only an identical prompt is consistent for
    # it); the 128-token checkpoint exists -> hit 128, [128, 192) computed from
    # the imported state, no split.
    scheduler.add_request(_request("B", toks))
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
    assert nr.num_computed_tokens == 128 and so.num_scheduled_tokens == {"B": 64}
    mock.forward_rows.clear()
    out = mock.step(so)
    assert backend.imports == [(mock.slots["B"], 128)] and mock.forward_rows == [
        ("B", 128, 192)
    ]
    assert backend.exports[-1] == (mock.slots["B"], 192) and store.has(toks, 192)
    scheduler.update_from_output(so, out)
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"B": 1}  # decode
    scheduler.update_from_output(so, mock.step(so))


def test_kda_checkpoint_dir_requires_chunked_prefill(tmp_path):
    cfg = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"tt_kda_checkpoint_dir": str(tmp_path)},
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=True, enable_chunked_prefill=False
        ),
        cache_config=SimpleNamespace(block_size=BLOCK),
        additional_config={},
        speculative_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    with pytest.raises(ValueError, match="chunked prefill"):
        validate_tt_kv_transfer_config(cfg)


def test_host_refuses_a_kda_model_without_a_checkpoint_store():
    backend = _MockKDABackend()
    caches = {
        n: torch.zeros(4, 1, BLOCK, LATENT, dtype=torch.bfloat16) for n in LAYER_NAMES
    }
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                list(LAYER_NAMES),
                MLAAttentionSpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=LATENT,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )
    connector = SimpleNamespace(
        register_kv_caches=lambda c: None, set_host_xfer_buffer_ops=lambda op: None
    )
    with pytest.raises(ValueError, match="tt_kda_checkpoint_dir"):
        TTKVConnectorHost(connector, kv_cache_config, caches, model=backend)


def test_checkpoint_grid_is_the_prefill_alignment(tmp_path):
    """page 64, tt.prefill_chunk_alignment 128 (the served GLM-5.3-Flash config): the checkpoint split and the hit grid
    are the 128-token continuation grid, not the page -- a sparse-regime prefill only starts on 128 boundaries (2x4 mock
    2026-10-22 22:37Z: a 1,000-token prompt checkpointed at the page 960 left a tail chunk starting at 960)."""
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir, align=2 * BLOCK)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    assert scheduler.connector._block_size == 2 * BLOCK  # the hit grid
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    worker = ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock = _MockRunner(worker, kv_cache_config, backend, store)
    toks = _prompt(200)
    scheduler.add_request(_request("A", toks))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 128}, "the split is the last 128 boundary, not the page 192"
    scheduler.update_from_output(so, mock.step(so))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 72}, "the tail runs whole from 128 (no 192 chunk)"
    scheduler.update_from_output(so, mock.step(so))
    assert [r[1] % (2 * BLOCK) for r in mock.forward_rows] == [0, 0], mock.forward_rows  # every chunk starts on 128
    assert store.has(toks, 128) and not store.has(toks, 192)
    assert scheduler.connector.aligned_hit(toks, 199) == 128  # a hit resumes on the 128 grid


@pytest.mark.parametrize("tail_split_skip", ["1", "floor", "0"])
def test_short_prompt_is_not_checkpointed_under_the_store_floor(tmp_path, monkeypatch, tail_split_skip):
    """GLM53_LMCACHE_MIN_STORE_TOKENS = 1,024: the 200-token prompt that checkpoints at 192 by default (the test above)
    takes no KDA checkpoint. GLM53_PF_TAIL_SPLIT_SKIP 1 / floor (cf3h-d lever B): nothing is stored, so its prefill runs
    as ONE chunk [0, 200) (no tail split); 0: [0, 192) + [192, 200) as before"""
    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "1024")
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_SKIP", tail_split_skip)
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    worker = ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock = _MockRunner(worker, kv_cache_config, backend, store)
    # the request's committed history, as the real runner wires it (_runner_request_tokens)
    mock.runner.kv_connector_host._tokens_of = lambda rid: mock.tokens.get(rid)
    toks = _prompt(200)
    scheduler.add_request(_request("A", toks))
    split = tail_split_skip == "0"
    for _ in range(3):  # [0, 192), [192, 200), one decode step (split) / [0, 200), two decode steps
        so = scheduler.schedule()
        scheduler.update_from_output(so, mock.step(so))
    want = [("A", 0, 192), ("A", 192, 200)] if split else [("A", 0, 200)]
    assert mock.forward_rows[: len(want)] == want
    assert backend.exports == [] and not store.has(toks, 192)
    assert mock.runner.kv_connector_host.kda_ckpt_skipped_short == (1 if split else 0)  # a 200-token chunk end is off the page grid


def test_unresumable_checkpoints_are_not_exported_and_hits_align_to_zero(tmp_path):
    """Stage 12 (n29h7p2 1843): a model whose checkpoints cannot resume a hit (kv_checkpoints_resumable False: a paged
    pooled-key store without the pooled pages) exports none, so a repeat of the prompt gets no connector hit and
    computes from 0 -- nothing is loaded only for the model to refuse it"""
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    backend.kv_checkpoints_resumable = lambda: False
    worker = ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock = _MockRunner(worker, kv_cache_config, backend, store)
    assert mock.runner.kv_connector_host.checkpoints_off
    toks = _prompt(200)
    scheduler.add_request(_request("A", toks))
    for _ in range(3):  # 192 (page boundary), the 8-token tail, one decode
        so = scheduler.schedule()
        scheduler.update_from_output(so, mock.step(so))
    assert backend.exports == [] and not store.has(toks, 192)
    scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
    mock.step(scheduler.schedule())
    scheduler.add_request(_request("B", toks + [9]))
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
    assert nr.num_computed_tokens == 0, "no checkpoint -> the hit aligns to 0 -> nothing is retrieved"
    mock.forward_rows.clear()
    mock.step(so)
    assert mock.forward_rows[0][1] == 0 and backend.imports == []


@pytest.mark.parametrize("at_end", ["aligned", "all"])
def test_an_aligned_prompt_runs_whole_and_is_checkpointed_at_its_end(tmp_path, monkeypatch, at_end):
    """GLM53_KDA_CKPT_AT_END aligned / all (cf3h-d, program 2026-10-24T06:38:54Z / 06:41:05Z): the 192-token
    (grid-aligned) prompt runs as ONE chunk and is checkpointed at 192. Its exact repeat finds no checkpoint at or below
    the page the connector reports (128) and prefills from 0; a continuation of it resumes at 192."""
    monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", at_end)
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    mock = _MockRunner(
        ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config),
        kv_cache_config,
        backend,
        store,
    )
    toks = _prompt(192, seed=11)
    scheduler.add_request(_request("A", toks))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 192}  # no split
    scheduler.update_from_output(so, mock.step(so))
    assert backend.exports == [(mock.slots["A"], 192)] and store.has(toks, 192) and not store.has(toks, 128)
    scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
    mock.step(scheduler.schedule())
    scheduler.add_request(_request("B", toks))  # the exact repeat
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
    assert nr.num_computed_tokens == 0 and so.num_scheduled_tokens == {"B": 192}
    # a continuation (the next turn) whose connector match covers the stored prompt resumes at its end
    proxy = scheduler.connector
    cont = list(toks) + _prompt(40, seed=12)
    assert proxy.aligned_hit(cont, 200) == 192 and proxy.aligned_hit(cont, 191) == 128 * 0


def test_both_splits_an_aligned_prompt_and_checkpoints_before_and_at_its_end(tmp_path, monkeypatch):
    """GLM53_KDA_CKPT_AT_END=both (cf3h-e default, program 2026-10-24T20:38:48Z (i)): the 192-token (grid-aligned)
    prompt keeps the split (128 + 64) and is checkpointed at 128 AND at 192: its exact repeat resumes at 128 (cf3h-d's
    "aligned" sent it back to 0 / an older checkpoint), a continuation resumes at 192."""
    monkeypatch.delenv("GLM53_KDA_CKPT_AT_END", raising=False)  # the default
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    mock = _MockRunner(
        ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config),
        kv_cache_config,
        backend,
        store,
    )
    toks = _prompt(192, seed=11)
    scheduler.add_request(_request("A", toks))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 128}  # the split stays
    scheduler.update_from_output(so, mock.step(so))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"A": 64}
    scheduler.update_from_output(so, mock.step(so))
    assert store.has(toks, 128) and store.has(toks, 192)
    assert backend.exports == [(mock.slots["A"], 128), (mock.slots["A"], 192)]
    scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
    mock.step(scheduler.schedule())
    scheduler.add_request(_request("B", toks))  # the exact repeat
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
    assert nr.num_computed_tokens == 128 and so.num_scheduled_tokens == {"B": 64}
    proxy = scheduler.connector
    cont = list(toks) + _prompt(40, seed=12)
    assert proxy.aligned_hit(cont, 200) == 192


def test_an_unaligned_prompt_keeps_its_split_under_aligned_and_loses_it_under_all(tmp_path, monkeypatch):
    for at_end, first in (("aligned", 192), ("all", 200)):
        monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", at_end)
        storage, kda_dir = str(tmp_path / at_end / "kv"), str(tmp_path / at_end / "kda")
        os.makedirs(storage)
        config, kv_cache_config = _config(storage, kda_dir)
        scheduler = TTScheduler(
            vllm_config=config,
            kv_cache_config=kv_cache_config,
            block_size=BLOCK,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(config),
        )
        scheduler.add_request(_request("A", _prompt(200)))
        assert scheduler.schedule().num_scheduled_tokens == {"A": first}


def test_in_replay_runs_every_storing_prompt_whole(tmp_path, monkeypatch):
    """GLM53_KDA_CKPT_IN_REPLAY=1: the 200-token (unaligned) prompt that splits at 192 under aligned runs as ONE
    chunk; the replay snapshots the cut"""
    monkeypatch.setenv("GLM53_KDA_CKPT_IN_REPLAY", "1")
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    scheduler.add_request(_request("A", _prompt(200)))
    assert scheduler.schedule().num_scheduled_tokens == {"A": 200}



@pytest.mark.parametrize("cold", [True, False])
@pytest.mark.parametrize("tail_split_skip", ["1", "floor"])
def test_a_request_that_saves_nothing_runs_its_prompt_end_whole(tmp_path, monkeypatch, tail_split_skip, cold):
    """Stage 12 W item (program 2026-10-25T15:38:54Z, from 17's 14:58:53Z table): with the one-replay store off,
    the KDA-checkpoint tail split ran every prompt's last chunk as a separate step -- also for a request that stores
    nothing (a tt.cold probe: the connector marks it lmcache.skip_save) -- +0.29 s per 2K request. GLM53_PF_TAIL_SPLIT_SKIP
    =1: a request that saves nothing (under the store floor, or skip_save) runs its prompt end as ONE chunk; 'floor'
    keeps the split for a skip_save probe above the floor. A request that DOES store keeps the split under both (its
    checkpoint at the last 128 boundary is what a later hit restores)."""
    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "1024")
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_SKIP", tail_split_skip)
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend()
    worker = ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock = _MockRunner(worker, kv_cache_config, backend, store)
    mock.runner.kv_connector_host._tokens_of = lambda rid: mock.tokens.get(rid)
    toks = _prompt(1500)  # above the floor
    cut = (len(toks) - 1) // BLOCK * BLOCK  # the last checkpoint boundary on this config's grid (1,472)
    req = _request("A", toks)
    if cold:
        req.kv_transfer_params = {"tt.cold": True}  # the client's probe flag (the connector adds lmcache.skip_save)
    scheduler.add_request(req)
    for _ in range(3):
        so = scheduler.schedule()
        scheduler.update_from_output(so, mock.step(so))
    whole = cold and tail_split_skip == "1"
    want = [("A", 0, 1500)] if whole else [("A", 0, cut), ("A", cut, 1500)]
    assert mock.forward_rows[: len(want)] == want, mock.forward_rows
    if not cold:
        assert store.has(toks, cut)  # the storing request still checkpoints at the last boundary


@pytest.mark.parametrize("cold", [False, True])
def test_tail_split_min_runs_a_short_prompt_end_whole_for_storing_and_cold_alike(tmp_path, monkeypatch, cold):
    """GLM53_PF_TAIL_SPLIT_MIN = 8,192 (default; program 2026-10-28T02:28:53Z (2)): a prompt below it gets NO
    KDA-checkpoint tail split, storing or cold (the plan no longer depends on storing: the hit-path reference gate's
    like plans by construction), so a storing 1,500-token request saves the extra replay; at or above the minimum the
    split stands (test_a_request_that_saves_nothing_runs_its_prompt_end_whole, under =0)."""
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_MIN", "8192")
    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "1024")
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_SKIP", "floor")
    storage, kda_dir = str(tmp_path / "kv"), str(tmp_path / "kda")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage, kda_dir)
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    store = KDACheckpointStore(kda_dir)
    worker = ExampleConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock = _MockRunner(worker, kv_cache_config, _MockKDABackend(), store)
    mock.runner.kv_connector_host._tokens_of = lambda rid: mock.tokens.get(rid)
    toks = _prompt(1500)
    req = _request("A", toks)
    if cold:
        req.kv_transfer_params = {"tt.cold": True}
    scheduler.add_request(req)
    for _ in range(3):
        so = scheduler.schedule()
        scheduler.update_from_output(so, mock.step(so))
    assert mock.forward_rows[:1] == [("A", 0, 1500)], mock.forward_rows  # one chunk, no split at 1,472
