# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A1 + A2 + A3 joint mock integration (branch glm53-lmcache-join).

The real ``TTModelRunner.execute_model`` with A1's ``TTKVConnectorHost`` drives vLLM's
``LMCacheConnectorV1`` (worker role) whose LMCache engine selects A2's
``VLLMPagedMemTTConnector`` through ``extra_config["gpu_connector"]`` on a CPU-only torch,
CPU + disk tiers; the real ``TTScheduler`` runs the scheduler-role connector (ZMQ lookup to
the worker's lookup server in this same process). No device, no ttnn: the caches are the
host-side ``[num_blocks, 1, 64, 512]`` bf16 tensors and A1's connector-facing views write
through to them, so the CopyBlocksOp is not needed on this path.

Flow (A3 harness): request A prefill = store; two decode steps = no connector traffic;
request B, same prompt, on cold caches = LMCache prefix hit, pages injected bit-exactly,
only the tail computed; byte counters from A2's connector stats.

Run alone in its own pytest process (LMCache's config and engine are process singletons):
  LMCACHE_JOIN_EVIDENCE_JSONL=<path> PYTHONPATH=src /opt/glm53/venv-lmcache/bin/python -m pytest tests/lmcache/test_join_lmcache_runner.py -v
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from contextlib import contextmanager
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
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

import vllm_tt_plugin  # noqa: E402
from vllm_tt_plugin.config import store_tt_output_tokens_per_step  # noqa: E402
from vllm_tt_plugin.kv_connector import (  # noqa: E402
    TTKVConnectorHost,
    TTSingleProcessGroup,
    ensure_tt_kv_transfer_initialized,
    ensure_tt_kv_transfer_shutdown,
    tt_copy_kv_blocks,
    validate_tt_kv_transfer_config,
)
from vllm_tt_plugin.lmcache import page_contract as pc  # noqa: E402
from vllm_tt_plugin.model_runner import TTModelRunner  # noqa: E402
from vllm_tt_plugin.scheduler import TTScheduler  # noqa: E402

from .tt_lmcache_testing import (  # noqa: E402
    BLOCK_SIZE,
    LATENT_DIM,
    LAYER_NAMES,
    ROW_BYTES,
    align,
    fingerprint,
    layer_index,
    prompt,
)

NUM_BLOCKS = 96
MAX_MODEL_LEN = 4096
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"
CONNECTOR_SPEC = "vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector"


# ----------------------------------------------------------------------------- environment
@pytest.fixture(scope="module")
def tp_group():
    """Findings J1/J2 are fixed in the plugin: ``ensure_tt_kv_transfer_initialized``
    installs a world-size-1 ``TTSingleProcessGroup`` as vLLM's TP/PP groups before the
    connector is created (LMCache's worker role reads ``get_tp_group()`` broadcasts,
    ``get_tensor_model_parallel_rank()`` and ``get_pp_group().is_last_rank``). Nothing is
    patched here; the test asserts the production stub after the init."""
    yield None


@pytest.fixture(scope="module")
def lmcache_config_file(tmp_path_factory):
    """A2's docs/lmcache/lmcache_tt.yaml shape with test-sized tiers under tmp."""
    import yaml
    from lmcache.integration.vllm import utils as lm_utils
    from lmcache.v1.config import LMCacheEngineConfig

    root = tmp_path_factory.mktemp("lmcache")
    cfg = {
        "chunk_size": pc.LMCACHE_CHUNK_TOKENS,
        "local_cpu": True,
        "max_local_cpu_size": 0.5,
        "local_cpu_use_hugepages": False,
        "local_disk": f"file://{root}/disk/",
        "max_local_disk_size": 2.0,
        "use_layerwise": False,
        "save_decode_cache": False,
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
        cfg["lmcache_rpc_port"] = f"a3join_{os.getpid()}"  # ZMQ ipc name: never collide with A1/A2 runs
    path = root / "lmcache_tt.yaml"
    path.write_text(yaml.safe_dump(cfg))
    os.environ["LMCACHE_CONFIG_FILE"] = str(path)
    if hasattr(lm_utils, "_config_instance"):
        lm_utils._config_instance = None
    yield str(path)


class _StubModel:
    pass


@contextmanager
def _stub_model_resolution():
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch("vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups", return_value=None),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch("vllm.model_executor.model_loader.utils.get_model_architecture", return_value=(_StubModel, None)),
    ):
        yield


def _engine_config() -> tuple[VllmConfig, KVCacheConfig]:
    model_config = ModelConfig(model=str(LOCAL_MODEL_CONFIG), dtype="bfloat16", seed=42, skip_tokenizer_init=True)
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=False)
    kv_transfer_config = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            kv_transfer_config=kv_transfer_config,
        )
    # LMCache derives kv_shape from the model config (vllm_service_factory.py:94-110):
    # the GLM-5.3-Flash numbers the TT connector requires -- MLA, 11 DSA layers, one kv head
    # of 544 (512 latent + 32 reserved) -- stated on the mock config.
    arch = config.model_config.model_arch_config
    for k, v in dict(is_deepseek_mla=True, head_size=pc.LMCACHE_HIDDEN_ELEMS, total_num_kv_heads=1).items():
        object.__setattr__(arch, k, v)
    config.model_config.hf_config.num_hidden_layers = pc.NUM_DSA_LAYERS
    # get_num_layers reads the arch config's layer count, not hf_config: shadow it on the instance
    config.model_config.get_num_layers = lambda parallel_config: pc.NUM_DSA_LAYERS
    mc, pcfg = config.model_config, config.parallel_config
    assert mc.use_mla and mc.get_head_size() == 544 and mc.get_num_kv_heads(pcfg) == 1
    assert mc.get_num_layers(pcfg) == pc.NUM_DSA_LAYERS
    validate_tt_kv_transfer_config(config)
    store_tt_output_tokens_per_step(config, 1)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    spec = MLAAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=LATENT_DIM, dtype=torch.bfloat16)
    return config, KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(list(LAYER_NAMES), spec)])


def _request(request_id: str, toks: list[int]) -> Request:
    init_none_hash(sha256)
    sp = SamplingParams(max_tokens=4, ignore_eos=True)
    sp.update_from_generation_config({}, eos_token_id=2)
    return Request(request_id=request_id, prompt_token_ids=toks, sampling_params=sp, pooling_params=None,
                   block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256))


# ----------------------------------------------------------------------------- the mock runner
class _MockRunner:
    """A1's test runner (tests/test_kv_connector.py): the real execute_model / sample_tokens, host stubs
    for build_model_input / forward / finish, fingerprints written into the native caches."""

    def __init__(self, connector, kv_cache_config: KVCacheConfig):
        self.caches = {n: torch.zeros(NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM, dtype=torch.bfloat16) for n in LAYER_NAMES}
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
        bt = torch.zeros(len(rows), MAX_MODEL_LEN // BLOCK_SIZE, dtype=torch.int32)
        for i, rid in enumerate(rows):
            bt[i, : len(self.blocks[rid])] = torch.tensor(self.blocks[rid], dtype=torch.int32)
        starts = torch.tensor([self.computed[r] for r in rows])
        ends = torch.tensor([self.computed[r] + so.num_scheduled_tokens[r] for r in rows])
        is_prefill = bool((ends > starts + 1).any()) or any(self.computed[r] < len(self.tokens[r]) for r in rows)
        toks = torch.zeros(len(rows), int(ends.max()), dtype=torch.int32)
        for i, rid in enumerate(rows):
            t = self.tokens[rid][: int(ends[i])]
            toks[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
        return SimpleNamespace(row_req_ids=rows, input_positions=starts, prompt_lens=ends if is_prefill else None,
                               block_tables=bt, block_tables_per_group=[bt], input_tokens=toks,
                               defer_device_sampling=False, spec_mode=None)

    def _forward(self, model_input):
        for i, rid in enumerate(model_input.row_req_ids):
            start = int(model_input.input_positions[i])
            end = int(model_input.prompt_lens[i]) if model_input.prompt_lens is not None else start + 1
            self.forward_rows.append((rid, start, end))
            for name, cache in self.caches.items():
                li = layer_index(name)
                for pos in range(start, end):
                    blk = int(model_input.block_tables[i, pos // BLOCK_SIZE])
                    cache[blk, 0, pos % BLOCK_SIZE] = fingerprint(li, pos, int(model_input.input_tokens[i, pos]))
        return SimpleNamespace(model_input=model_input)

    def _finish(self, grammar_output, *, fwd):
        ids = list(fwd.model_input.row_req_ids)
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 sampled_token_ids=[[5] for _ in ids], logprobs=None, prompt_logprobs_dict={}, pooler_output=[])

    def step(self, so: SchedulerOutput) -> ModelRunnerOutput:
        out = self.runner.execute_model(so)
        if out is None:
            out = self.runner.sample_tokens(None)
        return out


def _tt_connector(worker):
    """A2's GPU connector inside vLLM's LMCacheConnectorV1 wrapper (worker role)."""
    impl = getattr(worker, "_lmcache_engine", None) or getattr(worker, "_impl", None)
    eng = getattr(impl, "lmcache_engine", None)
    conn = getattr(eng, "gpu_connector", None)
    return impl, eng, conn


def _evidence(row: dict) -> None:
    path = os.environ.get("LMCACHE_JOIN_EVIDENCE_JSONL")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row)) + "\n")


# ----------------------------------------------------------------------------- the test
@pytest.mark.parametrize("prompt_len", [200, 2048])
def test_lmcache_tt_connector_prefill_store_then_prefix_hit_through_runner(tp_group, lmcache_config_file, prompt_len):
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector

    assert vllm_tt_plugin.__file__.startswith(str(Path(__file__).resolve().parents[2])), vllm_tt_plugin.__file__
    config, kv_cache_config = _engine_config()
    toks = prompt(prompt_len)
    hit_expect = align(prompt_len)  # A2: the stored span is align64(len - 1), the lookup key the same span

    # Worker first: its LMCache lookup server must exist before the scheduler's lookup client asks it.
    worker = ensure_tt_kv_transfer_initialized(config, kv_cache_config)
    assert type(worker).__name__ == "LMCacheConnectorV1"
    from vllm.distributed import parallel_state as ps

    assert isinstance(ps.get_tp_group(), TTSingleProcessGroup), "J1: production stub TP group"
    assert ps.get_tensor_model_parallel_rank() == 0 and ps.get_pp_group().is_last_rank  # J2
    mock_a = _MockRunner(worker, kv_cache_config)  # register_kv_caches -> LMCache post_init (A2's finding)
    impl, eng, conn = _tt_connector(worker)
    assert isinstance(conn, VLLMPagedMemTTConnector), f"LMCache must select the TT connector, got {type(conn)}"
    # J4: neither vLLM's wrapper nor LMCache's adapter forwards set_host_xfer_buffer_ops;
    # the plugin host hands the TT connector the copy op and the device handles itself.
    assert conn.copy_blocks is tt_copy_kv_blocks, "A2's connector did not receive tt_copy_kv_blocks"
    assert conn.device_kv_caches is mock_a.runner.kv_connector_host.kv_caches
    assert mock_a.runner.kv_connector_host.tt_gpu_connector is conn
    scheduler = TTScheduler(vllm_config=config, kv_cache_config=kv_cache_config, block_size=BLOCK_SIZE, log_stats=True,
                            structured_output_manager=StructuredOutputManager(config))
    try:
        # J3: the scheduler-side connector is the page-aligning proxy over LMCache's.
        # Inside the try so a failure still shuts LMCache's threads down (else pytest hangs).
        assert type(scheduler.connector).__name__ == "TTPageAlignedConnector"
        assert type(scheduler.connector.inner).__name__ == "LMCacheConnectorV1"
        # ---- request A: prefill = store -----------------------------------------------------------------------
        scheduler.add_request(_request("A", toks))
        so = scheduler.schedule()
        nr = so.scheduled_new_reqs[0]
        assert nr.num_computed_tokens == 0 and so.num_scheduled_tokens["A"] == prompt_len, "cold cache: no hit"
        blocks_a = list(nr.block_ids[0])
        assert 0 not in blocks_a
        s0 = dict(conn.stats)
        t0 = time.perf_counter()
        out = mock_a.step(so)
        prefill_s = time.perf_counter() - t0
        assert isinstance(out, ModelRunnerOutput) and out.kv_connector_output is not None
        assert mock_a.forward_rows == [("A", 0, prompt_len)]
        assert conn.stats["to_gpu_row_bytes"] == s0["to_gpu_row_bytes"], "a store must not load"
        # Stage 12 GLM53_KV_SAVE_DEFER (default): the prefill step's store runs after its token left -- at the next
        # step's begin / the scheduler's next lookup -- so nothing is stored by the prefill step itself
        assert conn.stats["from_gpu_tokens"] == s0["from_gpu_tokens"], "the prefill step's store is deferred"
        scheduler.update_from_output(so, out)
        # decode steps: no connector traffic (save_decode_cache: false) besides the deferred prefill store
        for _ in range(2):
            so = scheduler.schedule()
            assert so.num_scheduled_tokens == {"A": 1}
            s1 = dict(conn.stats)
            out = mock_a.step(so)
            assert conn.stats["to_gpu_row_bytes"] == s1["to_gpu_row_bytes"]
            scheduler.update_from_output(so, out)
        from vllm_tt_plugin.kv_connector import flush_deferred_saves

        flush_deferred_saves()  # (already ran at the first decode step's begin when the runner calls begin_step)
        from_gpu_bytes = conn.stats["from_gpu_row_bytes"] - s0["from_gpu_row_bytes"]
        from_gpu_tokens = conn.stats["from_gpu_tokens"] - s0["from_gpu_tokens"]
        stored_tokens_a = from_gpu_tokens // len(LAYER_NAMES)
        assert from_gpu_bytes == len(LAYER_NAMES) * stored_tokens_a * ROW_BYTES
        assert stored_tokens_a >= hit_expect, (stored_tokens_a, hit_expect)

        scheduler.finish_requests(["A"], RequestStatus.FINISHED_ABORTED)
        so_idle = scheduler.schedule()
        assert so_idle.total_num_scheduled_tokens == 0 and "A" in so_idle.finished_req_ids
        mock_a.step(so_idle)
        # LMCache keys the (partial) chunk on the tokens it stored -- the whole prompt here
        # (save_unfull_chunk) -- so the probe is the full prompt; the adapter's own lookup is the same.
        for _ in range(50):
            if eng.lookup(toks) >= hit_expect:
                break
            time.sleep(0.05)
        looked_up = eng.lookup(toks)
        assert looked_up >= hit_expect, f"engine lookup {looked_up} < {hit_expect}"

        # ---- request B: same prompt on cold caches = prefix hit through the runner ---------------------------
        for t in mock_a.caches.values():
            t.zero_()
        mock_a.forward_rows.clear()
        scheduler.add_request(_request("B", toks))
        so = scheduler.schedule()
        nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
        hit = nr.num_computed_tokens
        # J3: LMCache reports len - 1 hit tokens; the scheduler-side TTPageAlignedConnector
        # rounds the hit down to the 64-token page so the continuation starts on a page.
        assert hit == hit_expect, (hit, hit_expect)
        assert type(scheduler.connector).__name__ == "TTPageAlignedConnector"
        assert so.num_scheduled_tokens["B"] == prompt_len - hit
        blocks_b = list(nr.block_ids[0])
        s2 = dict(conn.stats)
        t0 = time.perf_counter()
        out = mock_a.step(so)
        hit_s = time.perf_counter() - t0
        to_gpu_bytes = conn.stats["to_gpu_row_bytes"] - s2["to_gpu_row_bytes"]
        loaded_tokens = (conn.stats["to_gpu_tokens"] - s2["to_gpu_tokens"]) // len(LAYER_NAMES)
        # LMCache retrieves every token it hit (up to len - 1); vLLM counts the block-aligned part as computed
        assert hit <= loaded_tokens <= prompt_len, (loaded_tokens, hit)
        assert to_gpu_bytes == len(LAYER_NAMES) * loaded_tokens * ROW_BYTES
        assert mock_a.forward_rows == [("B", hit, prompt_len)], "only the tail is computed"
        for name, cache in mock_a.caches.items():
            li = layer_index(name)
            for pos in range(prompt_len):
                blk = blocks_b[pos // BLOCK_SIZE]
                assert torch.equal(cache[blk, 0, pos % BLOCK_SIZE], fingerprint(li, pos, toks[pos])), (
                    name, pos, "loaded" if pos < hit else "computed")
            used = set(blocks_b)
            assert not cache[[b for b in range(NUM_BLOCKS) if b not in used]].any()
        scheduler.update_from_output(so, out)
        so = scheduler.schedule()
        assert so.num_scheduled_tokens == {"B": 1}
        _evidence(dict(test="test_lmcache_tt_connector_prefill_store_then_prefix_hit_through_runner", prompt_len=prompt_len,
                       layers=len(LAYER_NAMES), block_size=BLOCK_SIZE, stored_tokens=stored_tokens_a, hit_tokens=hit, loaded_tokens=loaded_tokens, engine_lookup=looked_up,
                       save_bytes=from_gpu_bytes, load_bytes=to_gpu_bytes, bytes_per_token_per_layer=ROW_BYTES,
                       prefill_step_s=round(prefill_s, 4), hit_step_s=round(hit_s, 4), connector="LMCacheConnectorV1",
                       gpu_connector=type(conn).__name__, lmcache_config=lmcache_config_file, vllm="0.26.0",
                       device="none (mock, torch views)", status="pass"))
    finally:
        scheduler.finish_requests([r for r in ("A", "B") if r in scheduler.requests], RequestStatus.FINISHED_ABORTED)
        ensure_tt_kv_transfer_shutdown(stop_lmcache=False)
        try:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)
        except Exception:
            pass
