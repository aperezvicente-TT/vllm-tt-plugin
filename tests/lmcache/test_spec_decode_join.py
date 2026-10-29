# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Speculative decoding x LMCache on the mock (Stage 12 x Stage 17,
doc/ttnn/17-mtp-spec-decode/lmcache_spec_contract.md).

The real ``TTScheduler`` with ``num_speculative_tokens = 1``, vLLM's
``LMCacheConnectorV1`` over the real LMCache engine (64-token chunks,
``save_decode_cache: true`` so generation chunks are stored too) and a mock runner
whose decode steps are verifies: row 0 is the committed token at ``c``, row 1 the
draft at ``c + 1``; both rows write fingerprints into the caches; the step commits
``n + 1`` tokens with ``n`` scripted per step. A 190-token prompt puts the page /
chunk boundary 192 inside the first verify, so:

* ``n = 0``: the draft at 191 is rejected. Nothing reaches the store that step
  (LMCache's tracker holds 191 committed tokens); the next step's row 0
  overwrites position 191 with the committed token and the chunk ``[128, 192)``
  is stored after that step: the stored rows are the committed tokens' rows.
  The boundary checkpoint 192 is deferred (``decode_boundaries_deferred``) and
  taken at the next step from candidate row 0.
* ``n = 1``: the draft is accepted; the chunk is stored at that step and the
  checkpoint 192 is taken from candidate row 1 (``B - c - 1``).

Then request B with the 192-token prefix: a 192-token LMCache hit whose pages
equal the committed tokens' fingerprints, and the 192 checkpoint imported. The
``no_hook`` variant (the shipped int28 generator: no ``export_spec_boundary_state``)
takes no decode-time checkpoint and B resumes at the prefill boundary 128.

Run alone in its own pytest process (LMCache's config and engine are process
singletons):
  PYTHONPATH=src /opt/glm53/venv-lmcache/bin/python -m pytest \
      tests/lmcache/test_spec_decode_join.py -v
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("lmcache")

from vllm.config import (  # noqa: E402
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.distributed import kv_transfer  # noqa: E402,F401
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from vllm_tt_plugin.config import store_tt_output_tokens_per_step  # noqa: E402
from vllm_tt_plugin.kv_connector import (  # noqa: E402
    KDACheckpointStore,
    TTKVConnectorHost,
    TTPageAlignedConnector,
    connector_spec_decode_mode,
    ensure_tt_kv_transfer_initialized,
    ensure_tt_kv_transfer_shutdown,
    step_boundary_checkpoints,
    validate_tt_kv_transfer_config,
)
from vllm_tt_plugin.lmcache import page_contract as pc  # noqa: E402
from vllm_tt_plugin.model_runner import TTModelRunner  # noqa: E402
from vllm_tt_plugin.scheduler import TTScheduler  # noqa: E402

from .test_kda_checkpoint_policy import _stub_model_resolution  # noqa: E402
from .tt_lmcache_testing import (  # noqa: E402
    BLOCK_SIZE,
    LATENT_DIM,
    LAYER_NAMES,
    fingerprint,
    layer_index,
    prompt,
)

NUM_BLOCKS = 96
MAX_MODEL_LEN = 4096
CHUNK = BLOCK_SIZE  # 64-token LMCache chunks: the boundary 192 is a chunk end
PROMPT_LEN = 190  # the first verify (190 committed, 191 draft) crosses 192
K = 1
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"
CONNECTOR_SPEC = "vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector"


# ------------------------------------------------------------ pure policy pure policy

@pytest.fixture(autouse=True)
def _tail_split_on(monkeypatch):
    """these hits resume at the KDA-checkpoint tail split of a short prompt: GLM53_PF_TAIL_SPLIT_MIN off (under it a
    one-chunk prompt keeps no checkpoint by design; tests/lmcache/test_kda_ckpt_store_async.py covers its rule)"""
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_MIN", "0")

def test_step_boundary_checkpoints_commit_defer_and_plain_rows():
    # c = 190, K = 1 verify wrote 190..191; n = 0 commits 1 token: P = 191.
    assert step_boundary_checkpoints(190, 1, 64, num_valid_drafts=1) == ([], [192])
    # n = 1 commits 2: P = 192 -> boundary 192 from candidate row 1.
    assert step_boundary_checkpoints(190, 2, 64, num_valid_drafts=1) == ([(192, 1)], [])
    # the deferred boundary at the next step (c = 191): row 0 holds it.
    assert step_boundary_checkpoints(191, 2, 64, num_valid_drafts=1) == ([(192, 0)], [])
    assert step_boundary_checkpoints(191, 1, 64, num_valid_drafts=1) == ([(192, 0)], [])
    # K = 3 with n = 3 crossing two boundaries: rows 1 and 65 - 62 - 1 ... one each.
    assert step_boundary_checkpoints(62, 4, 64, num_valid_drafts=3) == ([(64, 1)], [])
    assert step_boundary_checkpoints(126, 4, 2, num_valid_drafts=3) == (
        [(128, 1), (130, 3)],
        [],
    )
    # a plain decode row (no drafts): boundary only when c + 1 is one, own row.
    assert step_boundary_checkpoints(191, 1, 64) == ([(192, 0)], [])
    assert step_boundary_checkpoints(190, 1, 64) == ([], [])
    # padding / empty rows
    assert step_boundary_checkpoints(-1, 1, 64) == ([], [])
    assert step_boundary_checkpoints(190, 0, 64) == ([], [])


def test_connector_spec_decode_mode_admits_known_connectors_only():
    assert "committed tokens only" in connector_spec_decode_mode(
        SimpleNamespace(kv_connector="LMCacheConnectorV1")
    )
    assert "prompt pages only" in connector_spec_decode_mode(
        SimpleNamespace(kv_connector="ExampleConnector")
    )
    with pytest.raises(ValueError, match="not known to store committed tokens only"):
        connector_spec_decode_mode(SimpleNamespace(kv_connector="NixlConnector"))


def test_validate_admits_lmcache_with_speculative_decoding(tmp_path):
    cfg = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
            kv_connector_extra_config={"tt_kda_checkpoint_dir": str(tmp_path)},
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=True, enable_chunked_prefill=True
        ),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        additional_config={},
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    validate_tt_kv_transfer_config(cfg)  # admitted (was a ValueError)
    cfg.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector", kv_role="kv_both"
    )
    with pytest.raises(ValueError, match="speculative decoding on TT is validated"):
        validate_tt_kv_transfer_config(cfg)


def test_from_runner_admits_a_speculating_runner():
    """The host's own guard is gone: a runner with K drafts builds the host."""
    caches = {
        n: torch.zeros(4, 1, BLOCK_SIZE, LATENT_DIM, dtype=torch.bfloat16)
        for n in LAYER_NAMES
    }
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                list(LAYER_NAMES),
                MLAAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=LATENT_DIM,
                    dtype=torch.bfloat16,
                ),
            )
        ],
    )
    connector = SimpleNamespace(
        register_kv_caches=lambda c: None, set_host_xfer_buffer_ops=lambda op: None
    )
    runner = SimpleNamespace(
        _is_lane_mode=False,
        _num_speculative_tokens=2,
        _req_state_slot={"A": 3},
        kv_cache_config=kv_cache_config,
        kv_caches=caches,
        model=None,
        _layer_to_group_idx=None,
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        requests={"A": SimpleNamespace(prompt_token_ids=[1, 2], output_token_ids=[3])},
    )
    with patch(
        "vllm_tt_plugin.kv_connector.kv_transfer_state.get_kv_transfer_group",
        return_value=connector,
    ):
        host = TTKVConnectorHost.from_runner(runner)
    assert host.spec_k == 2
    assert host._slot_of("A") == 3 and host._tokens_of("A") == [1, 2, 3]


# ------------------------------------------------------------ environment environment
@pytest.fixture(scope="module")
def lmcache_config_file(tmp_path_factory):
    import yaml
    from lmcache.integration.vllm import utils as lm_utils
    from lmcache.v1.config import LMCacheEngineConfig

    root = tmp_path_factory.mktemp("lmcache_spec")
    cfg = {
        "chunk_size": CHUNK,
        "local_cpu": True,
        "max_local_cpu_size": 0.5,
        "local_cpu_use_hugepages": False,
        "local_disk": f"file://{root}/disk/",
        "max_local_disk_size": 2.0,
        "use_layerwise": False,
        "save_decode_cache": True,  # generation chunks are stored too
        "save_unfull_chunk": True,
        "enable_p2p": False,
        "extra_config": {
            "gpu_connector": CONNECTOR_SPEC,
            "tt_tokens_per_page": BLOCK_SIZE,
            "save_only_first_rank": False,
            "use_odirect": False,
        },
    }
    if "lmcache_rpc_port" in LMCacheEngineConfig.__dataclass_fields__:
        cfg["lmcache_rpc_port"] = f"specjoin_{os.getpid()}"
    path = root / "lmcache_tt.yaml"
    path.write_text(yaml.safe_dump(cfg))
    os.environ["LMCACHE_CONFIG_FILE"] = str(path)
    if hasattr(lm_utils, "_config_instance"):
        lm_utils._config_instance = None
    yield str(path)


def _engine_config(kda_dir: str) -> tuple[VllmConfig, KVCacheConfig]:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="bfloat16",
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
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={"tt_kda_checkpoint_dir": kda_dir},
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            kv_transfer_config=kv_transfer_config,
            additional_config={"tt": {"prefill_chunk_alignment": BLOCK_SIZE}},
        )
    # Attached after construction (tests/spec convention): the platform hook
    # resolves no plan for the stub model; the scheduler needs only K.
    config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=K,
        prompt_lookup_max=3,
        prompt_lookup_min=1,
        target_model_config=config.model_config,
        target_parallel_config=config.parallel_config,
    )
    arch = config.model_config.model_arch_config
    for k, v in dict(
        is_deepseek_mla=True, head_size=pc.LMCACHE_HIDDEN_ELEMS, total_num_kv_heads=1
    ).items():
        object.__setattr__(arch, k, v)
    config.model_config.hf_config.num_hidden_layers = pc.NUM_DSA_LAYERS
    config.model_config.get_num_layers = lambda parallel_config: pc.NUM_DSA_LAYERS
    validate_tt_kv_transfer_config(config)  # admits LMCache + spec, logs the mode
    store_tt_output_tokens_per_step(config, 1)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=LATENT_DIM,
        dtype=torch.bfloat16,
    )
    return config, KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(LAYER_NAMES), spec)],
    )


def _request(request_id: str, toks: list[int], max_tokens: int = 16) -> Request:
    init_none_hash(sha256)
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sp.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=toks,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def oracle(pos: int) -> int:
    """The model's own token at position ``pos`` (what a verify commits)."""
    return (pos * 7 + 3) % 97 + 10


class _MockKDABackend:
    """``export_kda_state`` (prefill boundaries) and, unless ``no_hook``,
    ``export_spec_boundary_state`` (decode boundaries from a candidate row)."""

    def __init__(self, *, no_hook: bool):
        self.exports: list[tuple[int, int]] = []
        self.boundary_exports: list[tuple[int, int, int | None]] = []
        self.imports: list[tuple[int, dict]] = []
        if no_hook:
            # the int28 generator: no candidate-row export
            self.export_spec_boundary_state = None  # type: ignore[assignment]

    def export_kda_state(self, kv, slot):
        self.exports.append((slot, -1))
        return {"kind": "prefill", "slot": slot}

    def export_spec_boundary_state(self, kv, slot, n_tokens, candidate_row):
        self.boundary_exports.append((slot, int(n_tokens), candidate_row))
        return {"kind": "boundary", "n_tokens": int(n_tokens), "row": candidate_row}

    def import_kda_state(self, kv, slot, states):
        self.imports.append((slot, states))


class _SpecMockRunner:
    """Real ``execute_model`` / ``sample_tokens``; decode steps are K=1 verifies.

    ``accept_plan`` scripts ``n`` for request A's successive verifies; the draft
    planted for a step is the oracle token when that step accepts, else a wrong
    one. Every verify row writes its fingerprint (committed token or draft) into
    the caches, like the device would.
    """

    def __init__(self, connector, kv_cache_config, backend, store, accept_plan):
        self.caches = {
            n: torch.zeros(NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM, dtype=torch.bfloat16)
            for n in LAYER_NAMES
        }
        self.backend = backend
        self.accept_plan = list(accept_plan)
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
        r._num_speculative_tokens = K
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
            spec_k=K,
            tokens_of=lambda rid: self.tokens.get(rid),
        )
        self.runner = r
        self.blocks: dict[str, list[int]] = {}
        self.tokens: dict[str, list[int]] = {}  # committed history: prompt + outputs
        self.computed: dict[str, int] = {}
        self.forward_rows: list[tuple[str, int, int]] = []
        self.verifies: list[tuple[str, int, int]] = []  # (req, c, n)

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
            self.computed[rid] = cr.num_computed_tokens[i]
        for rid in so.finished_req_ids:
            self.slots.pop(rid, None)
        if not so.total_num_scheduled_tokens:
            return None
        rows = list(so.num_scheduled_tokens)
        bt = torch.zeros(len(rows), MAX_MODEL_LEN // BLOCK_SIZE, dtype=torch.int32)
        for i, rid in enumerate(rows):
            bt[i, : len(self.blocks[rid])] = torch.tensor(
                self.blocks[rid], dtype=torch.int32
            )
        is_prefill = any(self.computed[r] < self._prompt_len(r) for r in rows)
        if is_prefill:
            starts = torch.tensor([self.computed[r] for r in rows])
            ends = torch.tensor(
                [self.computed[r] + so.num_scheduled_tokens[r] for r in rows]
            )
            toks = torch.zeros(len(rows), int(ends.max()), dtype=torch.int32)
            for i, rid in enumerate(rows):
                t = self.tokens[rid][: int(ends[i])]
                toks[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
            return SimpleNamespace(
                row_req_ids=rows,
                input_positions=starts,
                prompt_lens=ends,
                block_tables=bt,
                block_tables_per_group=[bt],
                input_tokens=toks,
                defer_device_sampling=False,
                spec_mode=None,
                prefill_empty_slots=None,
                num_valid_drafts=None,
            )
        # decode: a K=1 verify block per row: [committed token at c, draft at c+1]
        drafts = so.scheduled_spec_decode_tokens
        positions = torch.full((len(rows), 1 + K), -1, dtype=torch.int32)
        toks = torch.full((len(rows), 1 + K), -1, dtype=torch.int32)
        nvd = torch.zeros(len(rows), dtype=torch.int32)
        draft_ids = torch.full((len(rows), K), -1, dtype=torch.int32)
        for i, rid in enumerate(rows):
            c = self.computed[rid]
            assert so.num_scheduled_tokens[rid] == 1 + len(drafts.get(rid, []))
            positions[i, 0] = c
            toks[i, 0] = self.tokens[rid][c]
            for j, d in enumerate(drafts.get(rid, [])):
                positions[i, 1 + j] = c + 1 + j
                toks[i, 1 + j] = d
                draft_ids[i, j] = d
            nvd[i] = len(drafts.get(rid, []))
        return SimpleNamespace(
            row_req_ids=rows,
            input_positions=positions,
            prompt_lens=None,
            block_tables=bt,
            block_tables_per_group=[bt],
            input_tokens=toks,
            defer_device_sampling=False,
            spec_mode="argmax_ids",
            prefill_empty_slots=None,
            num_valid_drafts=nvd,
            draft_token_ids=draft_ids,
        )

    def _prompt_len(self, rid: str) -> int:
        return self._prompt_lens[rid]

    def _forward(self, model_input):
        for i, rid in enumerate(model_input.row_req_ids):
            if model_input.prompt_lens is not None:
                start, end = (
                    int(model_input.input_positions[i]),
                    int(model_input.prompt_lens[i]),
                )
                row_tokens = {
                    p: int(model_input.input_tokens[i, p]) for p in range(start, end)
                }
            else:
                c = int(model_input.input_positions[i, 0])
                width = 1 + int(model_input.num_valid_drafts[i])
                start, end = c, c + width
                row_tokens = {
                    c + j: int(model_input.input_tokens[i, j]) for j in range(width)
                }
            self.forward_rows.append((rid, start, end))
            for name, cache in self.caches.items():
                li = layer_index(name)
                for pos, tok in row_tokens.items():
                    blk = int(model_input.block_tables[i, pos // BLOCK_SIZE])
                    cache[blk, 0, pos % BLOCK_SIZE] = fingerprint(li, pos, tok)
        return SimpleNamespace(model_input=model_input)

    def _finish(self, grammar_output, *, fwd):
        mi = fwd.model_input
        ids = list(mi.row_req_ids)
        sampled = []
        for i, rid in enumerate(ids):
            if mi.prompt_lens is not None:
                end = int(mi.prompt_lens[i])
                if end < self._prompt_len(rid):
                    sampled.append([])  # intermediate chunk
                    continue
                block = [oracle(end)]
            else:
                c = int(mi.input_positions[i, 0])
                nvd = int(mi.num_valid_drafts[i])
                block = []
                for j in range(nvd):
                    draft = int(mi.input_tokens[i, 1 + j])
                    if draft != oracle(c + 1 + j):
                        break
                    block.append(draft)
                block.append(oracle(c + 1 + len(block)))  # the bonus token
                self.verifies.append((rid, c, len(block) - 1))
            self.tokens[rid].extend(block)
            sampled.append(block)
        return ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def step(self, so: SchedulerOutput) -> ModelRunnerOutput:
        out = self.runner.execute_model(so)
        if out is None:
            out = self.runner.sample_tokens(None)
        return out

    def plant_draft(self, scheduler: TTScheduler, rid: str, accept: bool) -> int:
        """The draft for the next verify: position = the request's committed length."""
        pos = len(self.tokens[rid])
        draft = oracle(pos) if accept else oracle(pos) + 1
        scheduler.update_draft_token_ids(
            DraftTokenIds(req_ids=[rid], draft_token_ids=[[draft]])
        )
        return draft


def _tt_connector(worker):
    impl = getattr(worker, "_lmcache_engine", None) or getattr(worker, "_impl", None)
    eng = getattr(impl, "lmcache_engine", None)
    return impl, eng, getattr(eng, "gpu_connector", None)


@pytest.mark.parametrize(
    "first_accept, no_hook",
    [(False, False), (True, False), (False, True)],
    ids=["reject-then-accept", "accept-first", "reject-no-hook"],
)
def test_spec_step_stores_committed_chunks_and_checkpoints_accepted_boundaries(
    lmcache_config_file, tmp_path, first_accept, no_hook
):
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    kda_dir = str(tmp_path / "kda")
    config, kv_cache_config = _engine_config(kda_dir)
    toks = prompt(PROMPT_LEN, seed=23)
    worker = ensure_tt_kv_transfer_initialized(config, kv_cache_config)
    assert type(worker).__name__ == "LMCacheConnectorV1"
    store = KDACheckpointStore(kda_dir)
    backend = _MockKDABackend(no_hook=no_hook)
    mock = _SpecMockRunner(worker, kv_cache_config, backend, store, [first_accept])
    mock._prompt_lens = {"A": PROMPT_LEN, "B": PROMPT_LEN + 3}
    host = mock.runner.kv_connector_host
    assert host.spec_k == K
    impl, eng, conn = _tt_connector(worker)
    assert conn is not None and host.tt_gpu_connector is conn
    scheduler = TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )
    assert scheduler.num_spec_tokens == K
    try:
        assert isinstance(scheduler.connector, TTPageAlignedConnector)
        # ---- A: prefill in two chunks: [0, 128) checkpoints 128 (P), [128, 190) tail
        scheduler.add_request(_request("A", toks))
        so = scheduler.schedule()
        assert so.num_scheduled_tokens == {"A": 128}
        scheduler.update_from_output(so, mock.step(so))
        slot_a = mock.slots["A"]
        # GLM53_KV_SAVE_DEFER (default): the prefill step's store / checkpoint run at the next step's begin (or the
        # scheduler's next lookup); flushing here = that moment (nothing ran in between)
        from vllm_tt_plugin.kv_connector import flush_deferred_saves

        flush_deferred_saves()
        assert backend.exports == [(slot_a, -1)] and store.has(toks, 128)
        so = scheduler.schedule()
        assert so.num_scheduled_tokens == {"A": PROMPT_LEN - 128}
        scheduler.update_from_output(so, mock.step(so))
        assert mock.tokens["A"] == toks + [oracle(PROMPT_LEN)]
        flush_deferred_saves()  # the tail chunk's deferred store: before the verify step (its begin_step would run it)
        stored_before = conn.stats["from_gpu_tokens"]

        # ---- verify 1: c = 190 committed, draft at 191; the boundary 192 is inside
        draft = mock.plant_draft(scheduler, "A", accept=first_accept)
        so = scheduler.schedule()
        assert so.num_scheduled_tokens == {"A": 1 + K}
        assert so.scheduled_spec_decode_tokens == {"A": [draft]}
        out = mock.step(so)
        n = len(out.sampled_token_ids[0]) - 1
        assert mock.verifies[-1] == ("A", PROMPT_LEN, n)
        assert n == (1 if first_accept else 0)
        assert mock.forward_rows[-1] == ("A", 190, 192), "the verify wrote 190 and 191"
        stored_after_v1 = conn.stats["from_gpu_tokens"] - stored_before
        scheduler.update_from_output(so, out)
        req_a = scheduler.requests["A"]
        committed_len = PROMPT_LEN + 1 + n + 1  # tokens with a known id
        assert len(req_a.all_token_ids) == committed_len
        assert req_a.num_computed_tokens == PROMPT_LEN + 1 + n, (
            "vLLM counts the accepted tokens only (rejected drafts are not computed)"
        )
        # LMCache decides a step's store at schedule time from the tokens vLLM
        # had committed by then (191 here, whatever the verify accepts), so the
        # chunk [128, 192) is stored by the NEXT step in both paths.
        assert stored_after_v1 == 0, "a verify step stores no row it wrote itself"
        if first_accept:
            # P = 192: the boundary checkpoint comes from candidate row 1 =
            # B - c - 1, taken by this step (the state is committed).
            assert backend.boundary_exports == [(mock.slots["A"], 192, 1)]
            assert host.decode_boundaries_checkpointed == 1
            assert host.decode_boundaries_deferred == 0
            assert store.has(mock.tokens["A"], 192)
        else:
            # P = 191: the boundary 192 the rejected draft row crossed is
            # deferred to the next step.
            assert backend.boundary_exports == []
            assert host.decode_boundaries_deferred == 1
            assert host.decode_boundaries_checkpointed == 0
            assert not store.has(mock.tokens["A"], 192)

        # ---- verify 2: the next step; with n = 0 before, row 0 re-writes 191
        draft = mock.plant_draft(scheduler, "A", accept=True)
        so = scheduler.schedule()
        c2 = PROMPT_LEN + 1 + n
        assert so.scheduled_spec_decode_tokens == {"A": [draft]}
        out = mock.step(so)
        assert mock.forward_rows[-1] == ("A", c2, c2 + 2)
        scheduler.update_from_output(so, out)
        history = list(mock.tokens["A"])
        assert history[:PROMPT_LEN] == toks and len(history) >= 193
        # The chunk [128, 192) lands with this step's save: after row 0
        # re-wrote 191 with the committed token (reject path).
        assert conn.stats["from_gpu_tokens"] > stored_before
        if not first_accept:
            if no_hook:
                assert backend.boundary_exports == []
                assert host.decode_boundaries_skipped == 1
                assert host.decode_boundaries_checkpointed == 0
                assert not store.has(history, 192)
            else:
                assert backend.boundary_exports == [(mock.slots["A"], 192, 0)]
                assert host.decode_boundaries_checkpointed == 1
                assert store.has(history, 192)
                assert store.load(history, 192)["row"] == 0
        else:
            assert store.load(history, 192)["row"] == 1
        scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
        mock.step(scheduler.schedule())

        # The engine holds the 192-token prefix of the COMMITTED history.
        prefix = history[:192]
        for _ in range(100):
            if eng.lookup(prefix) >= 192:
                break
            time.sleep(0.05)
        assert eng.lookup(prefix) >= 192, eng.lookup(prefix)

        # ---- B: the committed 192-token prefix + 3 -> hit; pages == committed rows
        toks_b = prefix + [1, 2, 3]
        for t in mock.caches.values():
            t.zero_()
        mock.forward_rows.clear()
        scheduler.add_request(_request("B", toks_b))
        so = scheduler.schedule()
        nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
        hit = nr.num_computed_tokens
        expected_hit = 128 if no_hook else 192
        assert hit == expected_hit, (hit, expected_hit)
        out = mock.step(so)
        blocks_b = list(nr.block_ids[0])
        for name, cache in mock.caches.items():
            li = layer_index(name)
            for pos in range(hit):
                got = cache[blocks_b[pos // BLOCK_SIZE], 0, pos % BLOCK_SIZE]
                assert torch.equal(got, fingerprint(li, pos, history[pos])), (
                    name,
                    pos,
                    "a stored row must be the committed token's, never a draft's",
                )
        assert backend.imports[-1][0] == mock.slots["B"]
        imported = backend.imports[-1][1]
        if no_hook:
            assert imported == {"kind": "prefill", "slot": slot_a}
        else:
            assert imported["kind"] == "boundary" and imported["n_tokens"] == 192
        # B's prefill stops on its last page boundary (192) like any prompt.
        assert mock.forward_rows == [("B", hit, 192 if hit < 192 else len(toks_b))]
        scheduler.update_from_output(so, out)
    finally:
        scheduler.finish_requests(
            [r for r in ("A", "B") if r in scheduler.requests],
            RequestStatus.FINISHED_ABORTED,
        )
        ensure_tt_kv_transfer_shutdown(stop_lmcache=False)
        with contextlib.suppress(Exception):
            LMCacheEngineBuilder.destroy(ENGINE_NAME)
