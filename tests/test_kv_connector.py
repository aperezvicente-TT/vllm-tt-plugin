# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for the TT KV-connector host (``vllm_tt_plugin.kv_connector``).

No device: the per-layer paged caches are torch tensors in the TT layout
``[num_blocks, 1, 64, 512]`` (doc/ttnn/15-lmcache/page_contract.md), the
"model" writes a per-token fingerprint into them, and vLLM's ``ExampleConnector``
(both roles) is the connector. The round trip drives the real ``TTScheduler``
and the real ``TTModelRunner.execute_model`` / ``sample_tokens`` wiring.
"""

from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

import vllm_tt_plugin  # noqa: F401  (activates the TT platform / ttnn import)
from vllm_tt_plugin.async_decode import DeferredDecodeOutput
from vllm_tt_plugin.config import store_tt_lane_count, store_tt_output_tokens_per_step
from vllm_tt_plugin.kv_connector import (
    TT_KV_PAGE_TOKENS,
    TTKVConnectorHost,
    TTPagedCacheHandle,
    build_step_slot_mapping,
    ensure_tt_kv_transfer_initialized,
    ensure_tt_kv_transfer_shutdown,
    resolve_connector_kv_views,
    tt_copy_kv_blocks,
    validate_tt_kv_transfer_config,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler

# page_contract.md §1-2: 11 DSA layers (i % 4 == 3), 64-token pages, latent 512 bf16.
LATENT = 512
BLOCK = TT_KV_PAGE_TOKENS
DSA_LAYERS = tuple(i for i in range(45) if i % 4 == 3)
LAYER_NAMES = tuple(f"model.layers.{i}.self_attn" for i in DSA_LAYERS)
NUM_BLOCKS = 48
MAX_MODEL_LEN = 2048
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


def _mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=LATENT, dtype=torch.bfloat16
    )


def _kv_cache_config(num_blocks: int = NUM_BLOCKS) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(LAYER_NAMES), _mla_spec())],
    )


def _device_caches(num_blocks: int = NUM_BLOCKS) -> dict[str, torch.Tensor]:
    """The model's caches in the native TT layout."""
    return {
        n: torch.zeros(num_blocks, 1, BLOCK, LATENT, dtype=torch.bfloat16)
        for n in LAYER_NAMES
    }


def fingerprint(layer_idx: int, pos: int, token: int) -> torch.Tensor:
    row = torch.zeros(LATENT, dtype=torch.bfloat16)
    row[0] = pos % 256
    row[1] = pos // 256
    row[2] = layer_idx
    row[3] = token % 256
    return row


def _layer_index(name: str) -> int:
    return int(name.split(".")[2])


# ---------------------------------------------------------------------------
# CopyBlocksOp (host transfer buffer)
# ---------------------------------------------------------------------------



@pytest.fixture(autouse=True)
def _save_in_step(monkeypatch):
    """these tests check the connector calls of a step: its save inside the step (GLM53_KV_SAVE_DEFER's own tests:
    tests/lmcache/test_kda_ckpt_store_async.py)"""
    monkeypatch.setenv("GLM53_KV_SAVE_DEFER", "0")


def test_copy_kv_blocks_round_trip_through_host_buffer():
    """d2h into a host buffer, h2d into another device cache: bit-exact pages."""
    torch.manual_seed(0)
    src_dev = {
        n: torch.randn(8, 1, BLOCK, LATENT).to(torch.bfloat16) for n in LAYER_NAMES[:2]
    }
    dst_dev = {
        n: torch.zeros(8, 1, BLOCK, LATENT, dtype=torch.bfloat16)
        for n in LAYER_NAMES[:2]
    }
    # Connector-facing views (what register_kv_caches hands out) and a host
    # buffer in that same shape, as a connector would allocate it.
    src_view = {n: t.view(8, BLOCK, LATENT) for n, t in src_dev.items()}
    dst_view = {n: t.view(8, BLOCK, LATENT) for n, t in dst_dev.items()}
    host = {
        n: torch.zeros(4, BLOCK, LATENT, dtype=torch.bfloat16) for n in LAYER_NAMES[:2]
    }

    tt_copy_kv_blocks(src_view, host, [5, 2, 7], [0, 1, 3], "d2h")
    tt_copy_kv_blocks(host, dst_view, [3, 0, 1], [6, 1, 4], "h2d")

    for n in LAYER_NAMES[:2]:
        assert torch.equal(dst_dev[n][6], src_dev[n][7])
        assert torch.equal(dst_dev[n][1], src_dev[n][5])
        assert torch.equal(dst_dev[n][4], src_dev[n][2])
        untouched = [i for i in range(8) if i not in (6, 1, 4)]
        assert not dst_dev[n][untouched].any()
        assert not host[n][2].any()


def test_copy_kv_blocks_reshapes_between_native_and_mla_page_shapes():
    src = {
        "l": torch.arange(4 * BLOCK * LATENT, dtype=torch.float32).view(
            4, 1, BLOCK, LATENT
        )
    }
    host = {"l": torch.zeros(2, BLOCK, LATENT)}
    tt_copy_kv_blocks(src, host, [3, 1], [0, 1], "d2h")
    assert torch.equal(host["l"][0], src["l"][3, 0])
    back = {"l": torch.zeros(4, 1, BLOCK, LATENT)}
    tt_copy_kv_blocks(host, back, [1, 0], [0, 2], "h2d")
    assert torch.equal(back["l"][0], src["l"][1])
    assert torch.equal(back["l"][2], src["l"][3])


def test_copy_kv_blocks_contract_edges():
    t = {"l": torch.zeros(2, BLOCK, LATENT)}
    tt_copy_kv_blocks({}, t, [0], [0], "h2d")  # no-op, upstream contract
    tt_copy_kv_blocks(t, t, [], [], "d2h")
    with pytest.raises(ValueError, match="direction"):
        tt_copy_kv_blocks(t, t, [0], [0], "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="destination blocks"):
        tt_copy_kv_blocks(t, t, [0, 1], [0], "d2h")
    with pytest.raises(IndexError):
        tt_copy_kv_blocks(t, t, [2], [0], "d2h")
    with pytest.raises(KeyError):
        tt_copy_kv_blocks(t, {"other": t["l"]}, [0], [0], "d2h")


def test_paged_cache_handle_routes_through_page_reads_and_writes():
    """A ttnn-backed cache is wrapped; the copy op uses its page methods."""
    backing = torch.zeros(6, 1, BLOCK, LATENT, dtype=torch.bfloat16)

    class _Fake(TTPagedCacheHandle):
        def read_pages(self, ids):
            return backing[ids].reshape(len(ids), *self.shape[1:])

        def write_pages(self, ids, pages):
            backing[ids] = pages.reshape(len(ids), 1, BLOCK, LATENT)

    handle = _Fake(device_tensor=None, shape=(6, BLOCK, LATENT), dtype=torch.bfloat16)
    host = {"l": torch.ones(2, BLOCK, LATENT, dtype=torch.bfloat16)}
    tt_copy_kv_blocks(host, {"l": handle}, [0, 1], [4, 2], "h2d")
    assert backing[4].eq(1).all() and backing[2].eq(1).all() and not backing[0].any()
    out = {"l": torch.zeros(1, BLOCK, LATENT, dtype=torch.bfloat16)}
    tt_copy_kv_blocks({"l": handle}, out, [4], [0], "d2h")
    assert out["l"].eq(1).all()


# ---------------------------------------------------------------------------
# Connector-facing views and slot mappings
# ---------------------------------------------------------------------------


def test_resolve_views_squeezes_mla_latent_and_aliases_the_model_cache():
    caches = _device_caches(4)
    views = resolve_connector_kv_views(caches, _kv_cache_config(4))
    assert list(views) == list(LAYER_NAMES)
    v = views[LAYER_NAMES[0]]
    assert (
        tuple(v.tensor.shape) == (4, BLOCK, LATENT)
        and v.is_mla
        and v.block_size == BLOCK
    )
    v.tensor[2, 5, 0] = 7  # write through the view ...
    assert caches[LAYER_NAMES[0]][2, 0, 5, 0] == 7  # ... lands in the model's cache
    assert v.group_index == 0 and v.layer_index == DSA_LAYERS[0]


def test_resolve_views_prefers_the_model_hook_and_refuses_kv_pairs():
    config = _kv_cache_config(4)
    hook_caches = _device_caches(4)
    model = SimpleNamespace(kv_cache_connector_views=lambda: hook_caches)
    views = resolve_connector_kv_views(object(), config, model)
    assert views[LAYER_NAMES[3]].device_tensor is hook_caches[LAYER_NAMES[3]]

    pairs = {
        n: (torch.zeros(4, 1, BLOCK, LATENT), torch.zeros(4, 1, BLOCK, LATENT))
        for n in LAYER_NAMES
    }
    with pytest.raises(TypeError, match="kv_cache_connector_views"):
        resolve_connector_kv_views(pairs, config)
    with pytest.raises(ValueError, match="no paged tensor"):
        resolve_connector_kv_views(
            {LAYER_NAMES[0]: hook_caches[LAYER_NAMES[0]]}, config
        )


def test_resolve_views_indexes_a_per_layer_sequence_by_layer_name():
    seq = [None] * 45
    for i in DSA_LAYERS:
        seq[i] = torch.zeros(3, 1, BLOCK, LATENT, dtype=torch.bfloat16)
    views = resolve_connector_kv_views(seq, _kv_cache_config(3))
    assert views[LAYER_NAMES[-1]].device_tensor is seq[43]
    # Legacy single "foo" spec over a per-layer sequence of non-MLA caches.
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
    )
    cfg = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["foo"], spec)],
    )
    views = resolve_connector_kv_views(
        [torch.zeros(3, 2, 16, 8), torch.zeros(3, 2, 16, 8)], cfg
    )
    assert list(views) == ["model.layers.0.attn", "model.layers.1.attn"]
    assert tuple(views["model.layers.1.attn"].tensor.shape) == (3, 2, 16, 8)


def test_build_step_slot_mapping_prefill_and_decode():
    bt = torch.tensor([[5, 9, 0, 0], [3, 0, 0, 0]], dtype=torch.int32)
    prefill = SimpleNamespace(
        row_req_ids=["a", "b"],
        input_positions=torch.tensor([60, 0]),
        prompt_lens=torch.tensor([70, 2]),
        block_tables=bt,
    )
    slots, starts = build_step_slot_mapping(prefill, bt, BLOCK)
    assert starts.tolist() == [0, 10, 12]
    assert slots.tolist() == [
        5 * BLOCK + 60,
        5 * BLOCK + 61,
        5 * BLOCK + 62,
        5 * BLOCK + 63,
    ] + [9 * BLOCK + o for o in range(6)] + [3 * BLOCK, 3 * BLOCK + 1]
    decode = SimpleNamespace(
        row_req_ids=["a"],
        input_positions=torch.tensor([70, -1, -1]),
        prompt_lens=None,
        block_tables=torch.tensor([[5, 9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )
    slots, starts = build_step_slot_mapping(decode, decode.block_tables, BLOCK)
    assert slots.tolist() == [9 * BLOCK + 6] and starts.tolist() == [0, 1]
    with pytest.raises(IndexError, match="block column"):
        build_step_slot_mapping(
            SimpleNamespace(
                row_req_ids=["a"],
                input_positions=torch.tensor([0]),
                prompt_lens=torch.tensor([5 * BLOCK]),
                block_tables=bt,
            ),
            bt,
            BLOCK,
        )


def test_build_step_slot_mapping_long_rows_match_the_per_position_rule():
    # the 1M line's held-store flush maps 8,192-token prefill rows against 16,392-column tables
    g = torch.Generator().manual_seed(7)
    cols = 16392
    bt = torch.randint(1, 16400, (3, cols), generator=g, dtype=torch.int32)
    mi = SimpleNamespace(
        row_req_ids=["a", "b", "c"],
        input_positions=torch.tensor([0, 524288, 1040000]),
        prompt_lens=torch.tensor([8192, 524288 + 8192, 1048000]),
        block_tables=bt,
    )
    slots, starts = build_step_slot_mapping(mi, bt, BLOCK)
    ref, ref_starts = [], [0]
    for row, (s, e) in enumerate([(0, 8192), (524288, 532480), (1040000, 1048000)]):
        ref += [int(bt[row, p // BLOCK]) * BLOCK + p % BLOCK for p in range(s, e)]
        ref_starts.append(len(ref))
    assert slots.dtype == torch.int64 and starts.dtype == torch.int32
    assert slots.tolist() == ref and starts.tolist() == ref_starts
    empty = SimpleNamespace(row_req_ids=["a"], input_positions=torch.tensor([-1]), prompt_lens=None, block_tables=bt)
    slots, starts = build_step_slot_mapping(empty, bt[:1], BLOCK)
    assert slots.tolist() == [] and slots.dtype == torch.int64 and starts.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Worker-process connector lifecycle
# ---------------------------------------------------------------------------


def _plain_vllm_config(tmp_path, connector="ExampleConnector"):
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        cache_config=SimpleNamespace(block_size=BLOCK, enable_prefix_caching=False),
        additional_config={},
        speculative_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1),
    )


def test_ensure_tt_kv_transfer_initialized_without_torch_distributed(tmp_path):
    assert not kv_transfer_state.has_kv_transfer_group()
    config = _plain_vllm_config(tmp_path)
    try:
        connector = ensure_tt_kv_transfer_initialized(config, _kv_cache_config())
        assert isinstance(connector, ExampleConnector)
        assert connector.role is KVConnectorRole.WORKER
        assert kv_transfer_state.get_kv_transfer_group() is connector
        # Idempotent: a second call returns the installed connector.
        assert (
            ensure_tt_kv_transfer_initialized(config, _kv_cache_config()) is connector
        )
    finally:
        ensure_tt_kv_transfer_shutdown()
    assert not kv_transfer_state.has_kv_transfer_group()
    assert (
        ensure_tt_kv_transfer_initialized(
            SimpleNamespace(kv_transfer_config=None), None
        )
        is None
    )


def test_validate_tt_kv_transfer_config_pins_hma_off_and_refuses_lane_mode(
    vllm_config, tmp_path
):
    validate_tt_kv_transfer_config(vllm_config)  # no connector: nothing to do
    vllm_config.kv_transfer_config = _plain_vllm_config(tmp_path).kv_transfer_config
    vllm_config.cache_config.block_size = BLOCK
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = None
    validate_tt_kv_transfer_config(vllm_config)
    assert vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is True
    store_tt_lane_count(vllm_config, 2)  # single engine, two in-process lanes
    with pytest.raises(ValueError, match="lane coordinator"):
        validate_tt_kv_transfer_config(vllm_config)


def test_deferred_output_post_hook_runs_once_inside_the_resolution():
    import threading

    class _Out(DeferredDecodeOutput):
        def __init__(self):
            self._completion_event = threading.Event()
            self._init_deferred()
            self.impl_calls = 0

        def _get_output_impl(self):
            self.impl_calls += 1
            return {"n": self.impl_calls}

    out = _Out()
    seen = []
    out.set_post_output_hook(lambda o: (seen.append(o), {**o, "hooked": True})[1])
    assert out.get_output() == {"n": 1, "hooked": True}
    assert out.get_output() == {"n": 1, "hooked": True}  # replayed, not re-run
    assert seen == [{"n": 1}] and out.impl_calls == 1
    with pytest.raises(RuntimeError):
        out.set_post_output_hook(lambda o: o)


# ---------------------------------------------------------------------------
# Round trip: TTScheduler + ExampleConnector + the runner's execute_model wiring
# ---------------------------------------------------------------------------


class _StubModel:
    pass


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
            return_value=(_StubModel, None),
        ),
    ):
        yield


def _engine_config(storage: str) -> tuple[VllmConfig, KVCacheConfig]:
    """A real VllmConfig with --kv-transfer-config ExampleConnector (A3 shape)."""
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
        enable_chunked_prefill=False,
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
        kv_connector_extra_config={"shared_storage_path": storage},
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            kv_transfer_config=kv_transfer_config,
        )
    validate_tt_kv_transfer_config(config)
    assert config.scheduler_config.disable_hybrid_kv_cache_manager is True
    store_tt_output_tokens_per_step(config, 1)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    return config, _kv_cache_config()


def _scheduler(config, kv_cache_config) -> TTScheduler:
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(request_id: str, prompt: list[int]) -> Request:
    init_none_hash(sha256)
    sp = SamplingParams(max_tokens=4, ignore_eos=True)
    sp.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


class RecordingConnector(ExampleConnector):
    """Worker-role ExampleConnector that records the host-visible call order."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[tuple[str, str | None]] = []
        self.seen_metadata: list[type] = []

    def handle_preemptions(self, meta):
        self.calls.append(("handle_preemptions", None))
        return super().handle_preemptions(meta)

    def start_load_kv(self, forward_context, **kw):
        self.calls.append(("start_load_kv", None))
        for name, m in forward_context.attn_metadata.items():
            self.seen_metadata.append(type(m))
        return super().start_load_kv(forward_context, **kw)

    def wait_for_layer_load(self, layer_name):
        self.calls.append(("wait_for_layer_load", layer_name))
        return super().wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw):
        self.calls.append(("save_kv_layer", layer_name))
        assert isinstance(attn_metadata, MLACommonMetadata)
        assert tuple(kv_layer.shape) == (NUM_BLOCKS, BLOCK, LATENT)
        return super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kw)

    def wait_for_save(self):
        self.calls.append(("wait_for_save", None))
        return super().wait_for_save()

    def get_finished(self, finished_req_ids):
        self.calls.append(("get_finished", None))
        return super().get_finished(finished_req_ids)


class _MockRunner:
    """A ``TTModelRunner`` whose model/forward are host stubs.

    The real ``execute_model`` / ``sample_tokens`` run; ``build_model_input``,
    ``_forward_with_model_input`` and ``_finish_front_packed_sync`` are
    instance-level stubs that keep a block-table bookkeeping like the persistent
    batch would and write the layer fingerprints into the native caches.
    """

    def __init__(self, connector: RecordingConnector, kv_cache_config: KVCacheConfig):
        self.caches = _device_caches()
        self.connector = connector
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
        # ``_is_lane_mode`` is a property over these two.
        r.parallel_config = SimpleNamespace(data_parallel_size=1)
        r.tt_data_parallel_size = 1
        r._num_speculative_tokens = 0
        r.model = None
        r.build_model_input = self._build_model_input
        r._forward_with_model_input = self._forward
        r._finish_front_packed_sync = self._finish
        r.kv_connector_host = TTKVConnectorHost(connector, kv_cache_config, self.caches)
        self.runner = r
        self.blocks: dict[str, list[int]] = {}
        self.tokens: dict[str, list[int]] = {}
        self.computed: dict[str, int] = {}
        self.forward_rows: list[tuple[str, int, int]] = []

    # -- persistent-batch stand-in -------------------------------------------
    def _build_model_input(self, so: SchedulerOutput, grammar_output):
        for nr in so.scheduled_new_reqs:
            self.blocks[nr.req_id] = list(nr.block_ids[0])
            self.tokens[nr.req_id] = list(nr.prompt_token_ids or [])
            self.computed[nr.req_id] = nr.num_computed_tokens
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            if cr.new_block_ids[i] is not None:
                self.blocks[rid].extend(cr.new_block_ids[i][0])
            if i < len(cr.new_token_ids) and cr.new_token_ids[i]:
                self.tokens[rid].extend(cr.new_token_ids[i])
            self.computed[rid] = cr.num_computed_tokens[i]
        if not so.total_num_scheduled_tokens:
            return None
        rows = list(so.num_scheduled_tokens)
        width = MAX_MODEL_LEN // BLOCK
        bt = torch.zeros(len(rows), width, dtype=torch.int32)
        for i, rid in enumerate(rows):
            bt[i, : len(self.blocks[rid])] = torch.tensor(
                self.blocks[rid], dtype=torch.int32
            )
        starts = torch.tensor([self.computed[r] for r in rows])
        ends = torch.tensor(
            [self.computed[r] + so.num_scheduled_tokens[r] for r in rows]
        )
        is_prefill = bool((ends > starts + 1).any()) or any(
            self.computed[r] < len(self.tokens[r]) for r in rows
        )
        toks = torch.zeros(len(rows), int(ends.max()), dtype=torch.int32)
        for i, rid in enumerate(rows):
            t = self.tokens[rid][: int(ends[i])]
            toks[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
        return SimpleNamespace(
            row_req_ids=rows,
            input_positions=starts if is_prefill else starts.clone(),
            prompt_lens=ends if is_prefill else None,
            block_tables=bt,
            block_tables_per_group=[bt],
            input_tokens=toks,
            defer_device_sampling=False,
            spec_mode=None,
        )

    # -- the "device program" --------------------------------------------------
    def _forward(self, model_input):
        self.connector.calls.append(("forward", None))
        for i, rid in enumerate(model_input.row_req_ids):
            start = int(model_input.input_positions[i])
            end = (
                int(model_input.prompt_lens[i])
                if model_input.prompt_lens is not None
                else start + 1
            )
            self.forward_rows.append((rid, start, end))
            for name, cache in self.caches.items():
                li = _layer_index(name)
                for pos in range(start, end):
                    blk = int(model_input.block_tables[i, pos // BLOCK])
                    cache[blk, 0, pos % BLOCK] = fingerprint(
                        li, pos, int(model_input.input_tokens[i, pos])
                    )
        return SimpleNamespace(model_input=model_input)

    def _finish(self, grammar_output, *, fwd):
        ids = list(fwd.model_input.row_req_ids)
        return ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[5] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def step(self, so: SchedulerOutput) -> ModelRunnerOutput:
        self.connector.calls.clear()
        out = self.runner.execute_model(so)
        if out is None:
            out = self.runner.sample_tokens(None)
        return out


def _prompt(n: int, seed: int = 7) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 128, (n,), generator=g).tolist()


def _expected_pages(
    name: str, prompt: list[int], block_ids: list[int], num_tokens: int
) -> dict[int, torch.Tensor]:
    """block id -> the [64, 512] page a correct model wrote for tokens < num_tokens."""
    li = _layer_index(name)
    pages: dict[int, torch.Tensor] = {}
    for pos in range(num_tokens):
        blk = block_ids[pos // BLOCK]
        pages.setdefault(blk, torch.zeros(BLOCK, LATENT, dtype=torch.bfloat16))[
            pos % BLOCK
        ] = fingerprint(li, pos, prompt[pos])
    return pages


@pytest.mark.parametrize("prompt_len", [200])
def test_example_connector_page_round_trip_through_runner(tmp_path, prompt_len):
    """Save per layer during a mock prefill; load on a second request's prefix hit."""
    storage = str(tmp_path / "kv")
    os.makedirs(storage)
    config, kv_cache_config = _engine_config(storage)
    scheduler = _scheduler(config, kv_cache_config)
    assert scheduler.connector is not None
    prompt = _prompt(prompt_len)
    saved_tokens = (prompt_len - 1) // BLOCK * BLOCK  # ExampleConnector: align64(len)

    # ---- request A: prefill = store --------------------------------------
    worker_a = RecordingConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock_a = _MockRunner(worker_a, kv_cache_config)
    scheduler.add_request(_request("A", prompt))
    so = scheduler.schedule()
    assert (
        so.num_scheduled_tokens["A"] == prompt_len
        and so.kv_connector_metadata is not None
    )
    blocks_a = list(so.scheduled_new_reqs[0].block_ids[0])
    assert 0 not in blocks_a

    out = mock_a.step(so)
    assert isinstance(out, ModelRunnerOutput) and out.kv_connector_output is not None
    names = [c[0] for c in worker_a.calls]
    # Order: preemptions, load + every layer waited, THEN the forward, THEN every
    # layer saved.
    assert names[:2] == ["handle_preemptions", "start_load_kv"]
    assert names[2 : 2 + len(LAYER_NAMES)] == ["wait_for_layer_load"] * len(LAYER_NAMES)
    assert names[2 + len(LAYER_NAMES)] == "forward"
    assert names[3 + len(LAYER_NAMES) : 3 + 2 * len(LAYER_NAMES)] == [
        "save_kv_layer"
    ] * len(LAYER_NAMES)
    assert names[-2:] == ["wait_for_save", "get_finished"]
    assert [c[1] for c in worker_a.calls if c[0] == "save_kv_layer"] == list(
        LAYER_NAMES
    )
    assert all(m is MLACommonMetadata for m in worker_a.seen_metadata)
    files = sorted(Path(storage).rglob("*.safetensors"))
    assert len(files) == len(LAYER_NAMES)
    scheduler.update_from_output(so, out)

    # The saved pages are the pages the model wrote (bit for bit).
    import safetensors.torch

    for f in files:
        name = f.stem
        saved = safetensors.torch.load_file(str(f))["kv_cache"]
        assert tuple(saved.shape) == (saved_tokens, LATENT)
        expect = mock_a.caches[name].view(NUM_BLOCKS * BLOCK, LATENT)[
            torch.tensor(
                [b * BLOCK + o for b in blocks_a for o in range(BLOCK)][:saved_tokens]
            )
        ]
        assert torch.equal(saved, expect)

    # Finish A so its blocks go back to the pool (B must get different pages).
    scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
    so_idle = scheduler.schedule()
    assert so_idle.total_num_scheduled_tokens == 0 and "A" in so_idle.finished_req_ids
    out_idle = mock_a.step(so_idle)
    # No forward: the connector still ran its load / get_finished pair.
    assert [c[0] for c in worker_a.calls] == [
        "handle_preemptions",
        "start_load_kv",
        "get_finished",
    ]
    assert (
        out_idle is EMPTY_MODEL_RUNNER_OUTPUT
        or out_idle.kv_connector_output is not None
    )

    # ---- request B: prefix hit = load into a fresh runner ----------------
    worker_b = RecordingConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    mock_b = _MockRunner(worker_b, kv_cache_config)
    prompt_b = prompt + [9]
    scheduler.add_request(_request("B", prompt_b))
    so = scheduler.schedule()
    assert so.scheduled_new_reqs[0].num_computed_tokens == saved_tokens
    assert so.num_scheduled_tokens["B"] == len(prompt_b) - saved_tokens
    blocks_b = list(so.scheduled_new_reqs[0].block_ids[0])
    meta = so.kv_connector_metadata
    assert len(meta.requests) == 1 and not meta.requests[0].is_store

    out = mock_b.step(so)
    assert out.kv_connector_output is not None
    # The mock forward only wrote tokens [saved_tokens, len); the prefix pages
    # came from the connector's load through the plugin's host layer.
    assert mock_b.forward_rows == [("B", saved_tokens, len(prompt_b))]
    for name, cache in mock_b.caches.items():
        expected = _expected_pages(name, prompt_b, blocks_b, len(prompt_b))
        for blk, page in expected.items():
            assert torch.equal(cache[blk, 0], page), f"{name} block {blk}"
        untouched = [b for b in range(NUM_BLOCKS) if b not in expected]
        assert not cache[untouched].any()
    scheduler.update_from_output(so, out)


# ---------------------------------------------------------------------------
# Join findings J1-J4 and the KDA checkpoint policy pieces
# ---------------------------------------------------------------------------


def test_single_process_groups_are_installed_for_the_worker_connector(tmp_path):
    """J1/J2: with no torch.distributed, the worker gets world-size-1 TP/PP groups."""
    from vllm.distributed import parallel_state as ps

    from vllm_tt_plugin.kv_connector import (
        TTSingleProcessGroup,
        tt_single_process_groups_installed,
    )

    assert ps._TP is None and not tt_single_process_groups_installed()
    config = _plain_vllm_config(tmp_path)
    try:
        connector = ensure_tt_kv_transfer_initialized(config, _kv_cache_config())
        assert isinstance(connector, ExampleConnector)
        assert tt_single_process_groups_installed()
        tp = ps.get_tp_group()
        assert isinstance(tp, TTSingleProcessGroup)
        assert ps.get_tensor_model_parallel_rank() == 0
        assert ps.get_tensor_model_parallel_world_size() == 1
        assert ps.get_pp_group().is_last_rank and ps.get_pp_group().is_first_rank
        obj = {"engine": "x"}
        assert tp.broadcast_object(obj, src=0) is obj  # LMCache lookup broadcasts
        t = torch.arange(3)
        assert tp.broadcast(t, src=0) is t
        # upstream's own init path ran (it broadcast engine_id over the stub)
        assert config.kv_transfer_config.engine_id
    finally:
        ensure_tt_kv_transfer_shutdown()
    assert ps._TP is None and ps._PP is None
    assert not tt_single_process_groups_installed()


class _FakeTTGpuConnector:
    """Duck type of A2's VLLMPagedMemTTConnector setup surface."""

    def __init__(self):
        self.copy_blocks = None
        self.device_kv_caches = None

    def set_host_xfer_buffer_ops(self, copy_blocks):
        self.copy_blocks = copy_blocks

    def register_device_kv_caches(self, caches):
        self.device_kv_caches = caches


class _FakeLMCacheWrapper:
    """vLLM's LMCacheConnectorV1 shape: holds the engine, forwards nothing (J4)."""

    def __init__(self, gpu):
        self._lmcache_engine = SimpleNamespace(
            lmcache_engine=SimpleNamespace(gpu_connector=gpu)
        )
        self.registered = None

    def register_kv_caches(self, kv_caches):
        self.registered = kv_caches

    def set_host_xfer_buffer_ops(self, op):
        return None  # the wrapper drops it, as upstream does


def test_tt_gpu_connector_receives_the_copy_op_and_device_handles():
    """J4: the host hands A2's connector tt_copy_kv_blocks + the device handles."""
    gpu = _FakeTTGpuConnector()
    wrapper = _FakeLMCacheWrapper(gpu)
    caches = _device_caches(4)
    host = TTKVConnectorHost(wrapper, _kv_cache_config(4), caches)
    assert host.tt_gpu_connector is gpu
    assert gpu.copy_blocks is tt_copy_kv_blocks, "A2's connector has no copy op"
    assert gpu.device_kv_caches is host.kv_caches
    assert wrapper.registered is host.kv_caches and not host.device_path

    # A wrapper without a TT GPU connector (ExampleConnector): nothing to hand over.
    plain = _FakeLMCacheWrapper(None)
    plain._lmcache_engine = None
    host2 = TTKVConnectorHost(plain, _kv_cache_config(4), _device_caches(4))
    assert host2.tt_gpu_connector is None


def test_device_path_registers_host_transfer_buffers():
    """ttnn-backed caches: connectors see host buffers; A2 gets the handles."""
    backing = {
        n: torch.zeros(4, 1, BLOCK, LATENT, dtype=torch.bfloat16) for n in LAYER_NAMES
    }

    class _Fake(TTPagedCacheHandle):
        def __init__(self, name):
            super().__init__(
                device_tensor=None, shape=(4, BLOCK, LATENT), dtype=torch.bfloat16
            )
            self.name = name

        def read_pages(self, ids):
            return backing[self.name][ids].reshape(len(ids), BLOCK, LATENT)

        def write_pages(self, ids, pages):
            backing[self.name][ids] = pages.reshape(len(ids), 1, BLOCK, LATENT)

    model = SimpleNamespace(
        kv_cache_connector_views=lambda: {n: _Fake(n) for n in LAYER_NAMES}
    )
    gpu = _FakeTTGpuConnector()
    wrapper = _FakeLMCacheWrapper(gpu)
    host = TTKVConnectorHost(wrapper, _kv_cache_config(4), object(), model=model)
    assert host.device_path and host.host_xfer_buffers is not None
    assert wrapper.registered is host.host_xfer_buffers
    assert tuple(host.host_xfer_buffers[LAYER_NAMES[0]].shape) == (4, BLOCK, LATENT)
    assert isinstance(gpu.device_kv_caches[LAYER_NAMES[0]], _Fake)
    # The copy op moves a host page into the (fake) device cache through the handle.
    host.host_xfer_buffers[LAYER_NAMES[0]][2] = 3
    tt_copy_kv_blocks(host.host_xfer_buffers, host.kv_caches, [2], [1], "h2d")
    assert backing[LAYER_NAMES[0]][1].eq(3).all()


def test_page_aligned_connector_rounds_hits_down_to_a_page(tmp_path):
    """J3: LMCache's len-1 hits end on a 64-token page; KDA store truncates further."""
    from vllm_tt_plugin.kv_connector import KDACheckpointStore, TTPageAlignedConnector

    calls = []

    class _Inner:
        role = "scheduler"

        def __init__(self, ext):
            self.ext = ext

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            calls.append((request.request_id, num_computed_tokens))
            return self.ext, False

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            calls.append(("alloc", request.request_id, num_external_tokens))

        def request_finished(self, request, block_ids):
            return False, None

    req = SimpleNamespace(request_id="r", prompt_token_ids=list(range(200)))
    proxy = TTPageAlignedConnector(_Inner(199), BLOCK)
    assert proxy.get_num_new_matched_tokens(req, 0) == (192, False)
    # 64 + 199 exceeds the 200-token prompt: clamped to 200 -> page 192 -> 128 new
    assert proxy.get_num_new_matched_tokens(req, 64) == (128, False)
    assert TTPageAlignedConnector(_Inner(135), BLOCK).get_num_new_matched_tokens(
        req, 64
    ) == (128, False)
    assert TTPageAlignedConnector(_Inner(63), BLOCK).get_num_new_matched_tokens(
        req, 0
    ) == (0, False)
    assert TTPageAlignedConnector(_Inner(0), BLOCK).get_num_new_matched_tokens(
        req, 0
    ) == (0, False)
    assert TTPageAlignedConnector(_Inner(None), BLOCK).get_num_new_matched_tokens(
        req, 0
    ) == (None, False)
    assert proxy.request_finished(req, []) == (False, None)  # forwarded
    assert calls[0] == ("r", 0)
    # LMCache's adapter asserts it gets its own count back (199), not vLLM's 192;
    # a hit vLLM rounded to nothing stays 0 (no load).
    proxy.get_num_new_matched_tokens(req, 0)
    proxy.update_state_after_alloc(req, None, 192)
    assert calls[-1] == ("alloc", "r", 199)
    none = TTPageAlignedConnector(_Inner(63), BLOCK)
    assert none.get_num_new_matched_tokens(req, 0) == (0, False)
    none.update_state_after_alloc(req, None, 0)
    assert calls[-1] == ("alloc", "r", 0)

    store = KDACheckpointStore(str(tmp_path / "kda"))
    with_store = TTPageAlignedConnector(_Inner(199), BLOCK, store)
    assert with_store.get_num_new_matched_tokens(req, 0) == (
        0,
        False,
    )  # no checkpoint at all
    store.save(req.prompt_token_ids, 128, [{"L": 0}])
    assert with_store.get_num_new_matched_tokens(req, 0) == (128, False)
    store.save(req.prompt_token_ids, 192, [{"L": 0}])
    assert with_store.get_num_new_matched_tokens(req, 0) == (192, False)
    assert with_store.get_num_new_matched_tokens(req, 64) == (128, False)


def test_kda_checkpoint_store_round_trip(tmp_path):
    from vllm_tt_plugin.kv_connector import KDACheckpointStore

    store = KDACheckpointStore(str(tmp_path / "kda"))
    toks = list(range(300))
    states = [{"L": 1, "S": torch.ones(2, 2), "C": torch.zeros(3)}]
    assert not store.has(toks, 64)
    store.save(toks, 64, states)
    assert store.has(toks, 64) and store.has(toks[:64] + [7, 8], 64)
    assert not store.has(toks[1:], 64)  # different prefix
    loaded = store.load(toks, 64)
    assert torch.equal(loaded[0]["S"], states[0]["S"]) and loaded[0]["L"] == 1
    assert store.load(toks, 128) is None
    assert store.longest_checkpointed(toks, 300, BLOCK) == 64
    store.save(toks, 192, states)
    assert store.longest_checkpointed(toks, 300, BLOCK) == 192
    assert store.longest_checkpointed(toks, 191, BLOCK) == 64
    with pytest.raises(ValueError):
        store.key(toks[:10], 64)
    assert (
        KDACheckpointStore.from_vllm_config(SimpleNamespace(kv_transfer_config=None))
        is None
    )


# ---------------------------------------------------------------------------
# Stage 9 real-engine findings J5-J10
# ---------------------------------------------------------------------------


def test_worker_reports_single_rank_handshake_metadata(tmp_path):
    """J6: vLLM's EngineCore asks every worker for connector handshake metadata."""
    from vllm_tt_plugin.worker import TTWorker

    worker = TTWorker.__new__(TTWorker)
    assert worker.get_kv_connector_handshake_metadata() is None  # no connector
    config = _plain_vllm_config(tmp_path)
    try:
        connector = ensure_tt_kv_transfer_initialized(config, _kv_cache_config())
        # ExampleConnector hands nothing over: None, like the GPU worker.
        assert worker.get_kv_connector_handshake_metadata() is None
        # A connector with metadata: keyed by the single-process (pp, tp) rank.
        connector.get_handshake_metadata = lambda: {"engine": "x"}
        assert worker.get_kv_connector_handshake_metadata() == {(0, 0): {"engine": "x"}}
    finally:
        ensure_tt_kv_transfer_shutdown()


def test_kv_shape_hint_from_specs_and_validation_publishes_it(tmp_path):
    """J5: the KV spec's latent geometry travels in kv_connector_extra_config."""
    from vllm_tt_plugin.kv_connector import (
        TT_KV_SHAPE_HINT_KEY,
        tt_kv_shape_hint_from_specs,
    )

    specs = {n: _mla_spec() for n in LAYER_NAMES}
    hint = tt_kv_shape_hint_from_specs(specs)
    assert hint == {
        "num_layers": 11,
        "latent_dim": LATENT,
        "tokens_per_page": BLOCK,
        "dtype": "bfloat16",
        "use_mla": True,
        "layer_names": list(LAYER_NAMES),
    }
    assert (
        tt_kv_shape_hint_from_specs(
            {
                "foo": FullAttentionSpec(
                    block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
                )
            }
        )
        is None
    )

    class _Model:
        @classmethod
        def get_kv_cache_spec(cls, vllm_config):
            return specs

    config = _plain_vllm_config(tmp_path)
    config.scheduler_config.enable_chunked_prefill = True
    validate_tt_kv_transfer_config(config, model_class=_Model)
    assert (
        config.kv_transfer_config.kv_connector_extra_config[TT_KV_SHAPE_HINT_KEY]
        == hint
    )


def test_short_tmpdir_is_forced_for_lmcache(tmp_path, monkeypatch):
    """J8: LMCache's ipc socket path needs a short TMPDIR."""
    import tempfile

    from vllm_tt_plugin.kv_connector import ensure_short_tmpdir_for_lmcache

    long_dir = tmp_path / ("x" * 40)
    long_dir.mkdir()
    monkeypatch.setenv("TMPDIR", str(long_dir))
    tempfile.tempdir = None
    cfg = _plain_vllm_config(tmp_path)
    assert ensure_short_tmpdir_for_lmcache(cfg) is None  # ExampleConnector: untouched
    assert os.environ["TMPDIR"] == str(long_dir)
    cfg.kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1", kv_role="kv_both"
    )
    assert ensure_short_tmpdir_for_lmcache(cfg) == "/tmp"
    assert os.environ["TMPDIR"] == "/tmp" and tempfile.gettempdir() == "/tmp"
    short = tmp_path / "s"
    cfg.kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={"tt_lmcache_tmpdir": str(short)},
    )
    if len(str(short)) <= 24:
        assert ensure_short_tmpdir_for_lmcache(cfg) == str(short)
    monkeypatch.setenv("TMPDIR", "/tmp")
    tempfile.tempdir = None
    assert (
        ensure_short_tmpdir_for_lmcache(
            cfg.__class__(
                **{
                    **cfg.__dict__,
                    "kv_transfer_config": KVTransferConfig(
                        kv_connector="LMCacheConnectorV1", kv_role="kv_both"
                    ),
                }
            )
        )
        == "/tmp"
    )
    tempfile.tempdir = None


def test_lmcache_torch_device_is_cpu_on_tt():
    """J9: the adapter's torch device must not be the ttnn MeshDevice."""
    from vllm_tt_plugin.kv_connector import fix_lmcache_torch_device

    mesh = object()  # stands in for ttnn.MeshDevice (not a torch.device)
    wrapper = SimpleNamespace(_lmcache_engine=SimpleNamespace(device=mesh))
    assert fix_lmcache_torch_device(wrapper)
    assert wrapper._lmcache_engine.device == torch.device("cpu")
    ok = SimpleNamespace(_lmcache_engine=SimpleNamespace(device=torch.device("cpu")))
    assert not fix_lmcache_torch_device(ok)
    assert not fix_lmcache_torch_device(SimpleNamespace())  # not LMCache


def test_squeezed_handle_presents_a_4d_latent_handle_as_3d():
    backing = torch.zeros(5, 1, BLOCK, LATENT, dtype=torch.bfloat16)

    class _Fake4D(TTPagedCacheHandle):
        def __init__(self):
            super().__init__(
                device_tensor="dev", shape=(5, 1, BLOCK, LATENT), dtype=torch.bfloat16
            )

        def read_pages(self, ids):
            return backing[ids]

        def write_pages(self, ids, pages):
            backing[ids] = pages.reshape(len(ids), 1, BLOCK, LATENT)

    model = SimpleNamespace(
        kv_cache_connector_views=lambda: {n: _Fake4D() for n in LAYER_NAMES}
    )
    views = resolve_connector_kv_views(object(), _kv_cache_config(5), model)
    v = views[LAYER_NAMES[0]].tensor
    assert tuple(v.shape) == (5, BLOCK, LATENT) and isinstance(v, TTPagedCacheHandle)
    host = {LAYER_NAMES[0]: torch.full((2, BLOCK, LATENT), 3.0, dtype=torch.bfloat16)}
    tt_copy_kv_blocks(host, {LAYER_NAMES[0]: v}, [0, 1], [4, 2], "h2d")
    assert backing[4].eq(3).all() and backing[2].eq(3).all() and not backing[0].any()
    out = {LAYER_NAMES[0]: torch.zeros(1, BLOCK, LATENT, dtype=torch.bfloat16)}
    tt_copy_kv_blocks({LAYER_NAMES[0]: v}, out, [4], [0], "d2h")
    assert out[LAYER_NAMES[0]].eq(3).all()


def test_ttnn_page_path_on_host_ttnn_tensors():
    """J10: the ttnn branches run on HOST ttnn tensors (no device) with real values.

    ``list(tensor.shape)`` (a ttnn Shape takes no slice), ``to_torch`` reads,
    rebuilt host tensors on writes, the handle following the rebuilt tensor,
    and the full CopyBlocksOp round trip through ``TTPagedCacheHandle``.
    """
    ttnn = pytest.importorskip("ttnn")
    from vllm_tt_plugin.kv_connector import (
        _ttnn_read_pages,
        _ttnn_shape,
        connector_facing_tensor,
    )

    torch.manual_seed(1)
    x = torch.randn(6, 1, BLOCK, LATENT).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    assert _ttnn_shape(t) == [6, 1, BLOCK, LATENT]
    with pytest.raises(TypeError):
        t.shape[1:]  # the J10 crash
    assert torch.equal(_ttnn_read_pages(t, [4, 1]), x[[4, 1]])

    handle = TTPagedCacheHandle(t, (6, 1, BLOCK, LATENT), torch.bfloat16)
    spec = _mla_spec()
    facing = connector_facing_tensor(handle, spec)
    assert tuple(facing.shape) == (6, BLOCK, LATENT)
    # d2h: device (host ttnn) -> host buffer, bit-exact
    host = {"l": torch.zeros(3, BLOCK, LATENT, dtype=torch.bfloat16)}
    tt_copy_kv_blocks({"l": facing}, host, [5, 0, 3], [0, 1, 2], "d2h")
    assert torch.equal(host["l"][0], x[5, 0]) and torch.equal(host["l"][2], x[3, 0])
    # h2d: host buffer -> (rebuilt) host ttnn tensor; the handle follows it
    host["l"][1] = 7
    tt_copy_kv_blocks(host, {"l": facing}, [1, 0], [2, 4], "h2d")
    now = ttnn.to_torch(handle.device_tensor)
    assert now[2].eq(7).all() and torch.equal(now[4, 0], x[5, 0])
    assert torch.equal(now[0], x[0]) and torch.equal(now[1], x[1])
    assert torch.equal(
        _ttnn_read_pages(facing.device_tensor, [2]).reshape(BLOCK, LATENT),
        torch.full((BLOCK, LATENT), 7.0, dtype=torch.bfloat16),
    )


def test_lmcache_adapter_shutdown_is_forwarded_and_bounded():
    """vLLM's LMCacheConnectorV1 wrapper never forwards shutdown to its adapter."""
    import threading

    from vllm_tt_plugin.kv_connector import shutdown_lmcache_engine

    calls = []
    wrapper = SimpleNamespace(
        _lmcache_engine=SimpleNamespace(shutdown=lambda: calls.append(1))
    )
    assert shutdown_lmcache_engine(wrapper) and calls == [1]
    assert not shutdown_lmcache_engine(SimpleNamespace())  # not LMCache

    def boom():
        raise RuntimeError("x")

    assert shutdown_lmcache_engine(
        SimpleNamespace(_lmcache_engine=SimpleNamespace(shutdown=boom))
    )
    gate = threading.Event()
    stuck = SimpleNamespace(_lmcache_engine=SimpleNamespace(shutdown=gate.wait))
    assert not shutdown_lmcache_engine(stuck, timeout_s=0.2)  # bounded
    gate.set()
