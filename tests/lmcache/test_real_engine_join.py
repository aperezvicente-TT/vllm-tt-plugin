# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The join plugin through a REAL in-process vLLM 0.26 engine (Stage 9's pattern).

Stage 9's engine harness (tt-metal s9 tree: ``tests/vllm/engine_harness.py``,
``glm53_mock_stack.py``) runs the real V1 EngineCore, TTScheduler, TTWorker and
TTModelRunner over a fake 8-device mesh and the GLM adapter with the CPU
MockModel; this module drives it from the plugin side without any shim, so the
startup blockers Stage 9 found (J5 LMCache kv_shape, J6 worker handshake
metadata, J7 page-aligned prompts) stay fixed.

Needs ``PYTHONPATH=<join>/src:<tt-metal s9 tree>`` and a venv with LMCache for the
LMCache start-up test (venv-lmcache); skipped otherwise.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import importlib.util

import pytest

# The CPU mock stack serves 2,051 positions per request (glm53_mock_stack.py). Pin the dense-exact program contract the
# tree's host suites run against (tests/vllm/conftest.py) before the adapter is imported: its class capability is read
# at import, and the served contract (65,536 tokens) would make the adapter refuse the mock generator.
_spec = importlib.util.find_spec("models.autoports.zai_org_glm_5_3_flash")
if _spec is not None and _spec.submodule_search_locations:
    _dense = (
        Path(list(_spec.submodule_search_locations)[0])
        / "tests/vllm/contracts/context_contract_dense_exact_2051.json"
    )
    if _dense.exists():
        os.environ.setdefault("GLM53_CONTEXT_CONTRACT", str(_dense))

H = pytest.importorskip(
    "models.autoports.zai_org_glm_5_3_flash.tests.vllm.engine_harness"
)
stack = pytest.importorskip(
    "models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack"
)

H.engine_env()

from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import (  # noqa: E402
    layer_kinds,
    text_config_of,
)
from vllm import SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from vllm_tt_plugin.kv_connector import (  # noqa: E402
    TTSingleProcessGroup,
    ensure_tt_kv_transfer_shutdown,
)

if not Path(H.SNAPSHOT).exists():
    pytest.skip("GLM-5.3-Flash snapshot config not available", allow_module_level=True)



@pytest.fixture(autouse=True)
def _tail_split_on(monkeypatch):
    """these hits resume at the KDA-checkpoint tail split of a short prompt: GLM53_PF_TAIL_SPLIT_MIN off (under it a
    one-chunk prompt keeps no checkpoint by design; tests/lmcache/test_kda_ckpt_store_async.py covers its rule)"""
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_MIN", "0")

@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(H.SNAPSHOT)


@pytest.fixture
def mock_latent_pages(hf_config, monkeypatch):
    """The 3-layer mock has one sparse_dsa layer standing in for all 11 of the spec."""
    mla = [
        i for i, k in enumerate(layer_kinds(text_config_of(hf_config))) if k == "mla"
    ]

    def latent_pages(self, kv_cache):
        lat = next(
            c.latent for lay, c in zip(self.layers, kv_cache.caches) if not lay.is_kda
        )
        return {i: lat for i in mla}

    monkeypatch.setattr(stack.RecordingMockModel, "latent_pages", latent_pages)
    yield mla


def _run_one(llm, rid, prompt, n):
    eng = llm.llm_engine
    params = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt), params)
    out = None
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                out = o
    return list(out.outputs[0].token_ids), out


def _example_kv(tmp_path):
    return KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": str(tmp_path / "pages"),
            "tt_kda_checkpoint_dir": str(tmp_path / "kda"),
        },
    )


@pytest.mark.parametrize("prompt_len, hit, at_end", [(200, 192, "aligned"), (192, 128, "0"), (192, 0, "aligned")])
def test_engine_example_connector_prefix_hit_no_shims(
    tmp_path, mock_latent_pages, prompt_len, hit, at_end, monkeypatch
):
    """J6 (no handshake shim) + J7 (GLM53_KDA_CKPT_AT_END=0: a 192-token prompt checkpoints and hits at 128).
    "aligned" (cf3h-d): an unaligned 200-token prompt still splits and hits at 192; the grid-aligned 192-token prompt
    runs whole and is checkpointed at its end, so its exact repeat finds no checkpoint at or below 128 and prefills
    from 0 (a continuation resumes at 192: test_kda_checkpoint_policy)."""
    monkeypatch.setenv("GLM53_KDA_CKPT_AT_END", at_end)
    from vllm.distributed import parallel_state as ps

    # One process, several engines: the previous engine's worker connector must
    # not survive in kv_transfer_state (the in-process LLM teardown is lazy).
    ensure_tt_kv_transfer_shutdown(stop_lmcache=False)
    with H.fake_tt_mesh():
        llm = H.make_llm(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            async_scheduling=False,
            kv_transfer_config=_example_kv(tmp_path),
            tt_config={"prefill_chunk_alignment": 64},
        )
        try:
            assert isinstance(
                ps.get_tp_group(), TTSingleProcessGroup
            )  # J1 in the real engine
            core = llm.llm_engine.engine_core.engine_core
            model = core.model_executor.driver_worker.worker.model_runner.model
            gen = H.live_generator()
            prompt = stack.prompt_ids(51, prompt_len)
            a, _ = _run_one(llm, "a", prompt, 8)
            exports = model.counters["kda_exports"]
            b, out_b = _run_one(llm, "b", prompt, 8)
            ref = stack.reference_greedy(gen.backend, prompt, 8)
            assert a == ref and b == ref
            assert exports >= 1 and model.counters["kda_imports"] == (1 if hit else 0)
            assert gen.counters["prefix_state_imports"] == (1 if hit else 0)
            assert out_b.num_cached_tokens == hit, out_b.num_cached_tokens
            assert len(os.listdir(tmp_path / "kda")) >= 1
        finally:
            del llm
            ensure_tt_kv_transfer_shutdown(stop_lmcache=False)


@pytest.fixture
def lmcache_config_file(tmp_path):
    lmcache = pytest.importorskip("lmcache")  # noqa: F841
    import yaml
    from lmcache.integration.vllm import utils as lm_utils
    from lmcache.v1.config import LMCacheEngineConfig

    cfg = {
        "chunk_size": 256,
        "local_cpu": True,
        "max_local_cpu_size": 0.5,
        "local_cpu_use_hugepages": False,
        "use_layerwise": False,
        "save_decode_cache": False,
        "save_unfull_chunk": True,
        "enable_p2p": False,
        "extra_config": {
            "gpu_connector": "vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector",
            "tt_tokens_per_page": 64,
            "save_only_first_rank": False,
            "use_odirect": False,
        },
    }
    if "lmcache_rpc_port" in LMCacheEngineConfig.__dataclass_fields__:
        cfg["lmcache_rpc_port"] = f"a1eng_{os.getpid()}"
    path = tmp_path / "lmcache_tt.yaml"
    path.write_text(yaml.safe_dump(cfg))
    os.environ["LMCACHE_CONFIG_FILE"] = str(path)
    if hasattr(lm_utils, "_config_instance"):
        lm_utils._config_instance = None
    yield str(path)


def test_engine_starts_with_lmcache_connector_on_the_real_glm_config(
    tmp_path, mock_latent_pages, lmcache_config_file
):
    """J5 + J6 + J8 + J9: the engine comes up with LMCacheConnectorV1 on the real
    GLM-5.3-Flash config (vLLM: use_mla False, 45 layers) and LMCache's metadata
    carries the KV spec's geometry. No request is run: the CPU mock's latent is
    float64 and not 512 wide, so LMCache's page binding would refuse it (the
    mock-cluster server rehearsal covers the request path)."""
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from vllm.distributed import kv_transfer

    from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector

    long_tmp = tmp_path / ("t" * 40)
    long_tmp.mkdir()
    os.environ["TMPDIR"] = str(long_tmp)  # J8: the plugin must shorten it
    ensure_tt_kv_transfer_shutdown(stop_lmcache=False)
    kv = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={"tt_kda_checkpoint_dir": str(tmp_path / "kda")},
    )
    with H.fake_tt_mesh():
        llm = H.make_llm(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            async_scheduling=False,
            kv_transfer_config=kv,
            tt_config={"prefill_chunk_alignment": 64},
        )
        try:
            assert os.environ["TMPDIR"] == "/tmp"  # J8
            worker = kv_transfer.get_kv_transfer_group()
            assert type(worker).__name__ == "LMCacheConnectorV1"
            impl = worker._lmcache_engine
            assert impl.device == __import__("torch").device("cpu")  # J9
            engine = impl.lmcache_engine
            conn = engine.gpu_connector
            assert isinstance(conn, VLLMPagedMemTTConnector)
            assert conn.num_layers == 11 and conn.latent_bytes_per_token == 1024
            assert tuple(engine.metadata.kv_shape) == (11, 1, 256, 1, 544)  # J5
            assert engine.metadata.use_mla
            core = llm.llm_engine.engine_core.engine_core
            host = (
                core.model_executor.driver_worker.worker.model_runner.kv_connector_host
            )
            assert host.tt_gpu_connector is conn and conn.copy_blocks is not None  # J4
            assert type(core.scheduler.connector).__name__ == "TTPageAlignedConnector"
        finally:
            del llm
            ensure_tt_kv_transfer_shutdown(stop_lmcache=False)
            with contextlib.suppress(Exception):
                LMCacheEngineBuilder.destroy(ENGINE_NAME)
