# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A3 harness: prefill + decode through the real TTScheduler with vLLM's ExampleConnector attached via
kv_transfer_config, the worker side driven by the reference host (connector_host_ref.py).  No device, no tt-metal.

Asserts (doc/ttnn/15-lmcache/page_contract.md): the scheduler-side connector emits one store ReqMeta with the request's
block ids on its prefill step and nothing on decode steps; the worker-side callbacks fire once per DSA layer, in layer
order, with slot mappings block_id * 64 + offset; a second request with the same prompt is a connector prefix hit of
align64(len - 1) tokens whose pages are injected per layer bit-exactly; bytes moved per request match 1,024 B/token/layer.
Evidence rows go to $A3_EVIDENCE_JSONL when set."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
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
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import store_tt_output_tokens_per_step
from vllm_tt_plugin.scheduler import TTScheduler

from .connector_host_ref import (
    BLOCK_SIZE,
    LATENT_DIM,
    LAYER_NAMES,
    ConnectorHostRef,
    RecordingConnector,
    fingerprint,
    layer_index,
    mock_compute,
)

MAX_MODEL_LEN = 4096
NUM_BLOCKS = 96  # 6,144 tokens of pool: 32 users x 2K would be 1,024 blocks; the harness only needs two requests
LOCAL_MODEL_CONFIG = Path(__file__).parent.parent / "model_configs" / "qwen2"
ROW_BYTES = LATENT_DIM * 2


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


def _config(storage: str) -> tuple[VllmConfig, KVCacheConfig]:
    model_config = ModelConfig(model=str(LOCAL_MODEL_CONFIG), dtype="float16", seed=42, skip_tokenizer_init=True)
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    # GLM-5.3-Flash refuses vLLM prefix caching (tt/generator_vllm.py:869-870); the connector path does not need it
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=False
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
    config.scheduler_config.disable_hybrid_kv_cache_manager = True  # ExampleConnector has no HMA support
    store_tt_output_tokens_per_step(config, 1)
    cache_config.num_gpu_blocks = NUM_BLOCKS
    # tt/generator_vllm.py:864-883: one MLAAttentionSpec per DSA layer, block 64, 1 kv head, head_size 512, bf16
    spec = MLAAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=LATENT_DIM, dtype=torch.bfloat16)
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(LAYER_NAMES), spec)],
    )
    return config, kv_cache_config


def _scheduler(config, kv_cache_config) -> TTScheduler:
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _prompt(n: int, seed: int = 7) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 128, (n,), generator=g).tolist()


def _request(request_id: str, prompt: list[int], max_tokens: int = 8) -> Request:
    init_none_hash(sha256)
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sp.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _runner_output(so, tokens: dict[str, list[int]]) -> ModelRunnerOutput:
    ids = list(so.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=[tokens.get(r, [5]) for r in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _evidence(row: dict) -> None:
    path = os.environ.get("A3_EVIDENCE_JSONL")
    if path:
        row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _align(n: int) -> int:
    return (n - 1) // BLOCK_SIZE * BLOCK_SIZE


@pytest.mark.parametrize("prompt_len", [200, 2048])
def test_prefill_save_then_prefix_hit_load(tmp_path, prompt_len):
    storage = str(tmp_path / "kv")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage)
    scheduler = _scheduler(config, kv_cache_config)
    assert scheduler.connector is not None, "TTScheduler must inherit vLLM's scheduler-side connector"

    kv = {n: torch.zeros(NUM_BLOCKS, BLOCK_SIZE, LATENT_DIM, dtype=torch.bfloat16) for n in LAYER_NAMES}
    worker = RecordingConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    host = ConnectorHostRef(worker, kv)
    prompt = _prompt(prompt_len)
    saved_tokens = _align(prompt_len)  # ExampleConnector stores align64(len(prompt)) rows and keys on align64(len - 1)
    hit_tokens = _align(prompt_len - 1 + 1)
    assert saved_tokens == hit_tokens

    # ---- request A: prefill = store ------------------------------------------------------------------------------
    scheduler.add_request(_request("A", prompt))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens["A"] == prompt_len
    meta = so.kv_connector_metadata
    assert meta is not None and len(meta.requests) == 1 and meta.requests[0].is_store
    blocks_a = list(so.scheduled_new_reqs[0].block_ids[0])
    assert len(blocks_a) == -(-prompt_len // BLOCK_SIZE)
    assert 0 not in blocks_a, "block 0 is vLLM's null block (tt/generator_vllm.py:611-618)"
    expect_slots = [b * BLOCK_SIZE + o for b in blocks_a for o in range(BLOCK_SIZE)][:saved_tokens]
    assert meta.requests[0].slot_mapping.tolist() == expect_slots

    t0 = time.perf_counter()
    rep = host.step(so, mock_compute)
    prefill_s = time.perf_counter() - t0
    assert rep.names("wait_for_layer_load") == list(LAYER_NAMES)
    assert rep.names("save_kv_layer") == list(LAYER_NAMES)
    assert rep.names("start_load_kv") == [None] and rep.load_bytes == 0
    save_bytes_a = rep.save_bytes
    assert save_bytes_a == len(LAYER_NAMES) * saved_tokens * ROW_BYTES
    files = sorted(p for p in Path(storage).rglob("*.safetensors"))
    assert len(files) == len(LAYER_NAMES)
    on_disk = sum(p.stat().st_size for p in files)
    assert on_disk >= rep.save_bytes
    scheduler.update_from_output(so, _runner_output(so, {"A": [11]}))

    # ---- decode steps: no connector traffic ----------------------------------------------------------------------
    for step in range(2):
        so = scheduler.schedule()
        assert so.num_scheduled_tokens == {"A": 1}
        assert so.kv_connector_metadata is None or so.kv_connector_metadata.requests == []
        rep = host.step(so, mock_compute)
        assert rep.load_bytes == 0 and rep.save_bytes == 0
        scheduler.update_from_output(so, _runner_output(so, {"A": [12 + step]}))
    scheduler.finish_requests("A", RequestStatus.FINISHED_ABORTED)
    so = scheduler.schedule()  # drains the finish
    scheduler.update_from_output(so, _runner_output(so, {}))

    # ---- request B, same prompt: connector prefix hit, pages injected per layer -------------------------------------
    for t in kv.values():
        t.zero_()  # cold device: whatever B reads must have come through start_load_kv
    scheduler.add_request(_request("B", prompt))
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
    assert nr.num_computed_tokens == hit_tokens, "external (connector) tokens count as computed"
    assert so.num_scheduled_tokens["B"] == prompt_len - hit_tokens
    meta = so.kv_connector_metadata
    assert len(meta.requests) == 1 and not meta.requests[0].is_store
    blocks_b = list(nr.block_ids[0])
    assert meta.requests[0].slot_mapping.tolist() == [b * BLOCK_SIZE + o for b in blocks_b for o in range(BLOCK_SIZE)][:hit_tokens]

    t0 = time.perf_counter()
    rep = host.step(so, mock_compute)
    hit_s = time.perf_counter() - t0
    assert rep.names("start_load_kv") == [None]
    assert rep.load_bytes == len(LAYER_NAMES) * hit_tokens * ROW_BYTES
    assert rep.save_bytes == 0, "ExampleConnector: a hit is not re-stored"
    for name, t in kv.items():
        li = layer_index(name)
        flat = t.reshape(-1, LATENT_DIM)
        for p in range(prompt_len):
            s = blocks_b[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE
            assert torch.equal(flat[s], fingerprint(li, p, prompt[p])), (name, p, "loaded" if p < hit_tokens else "computed")
    scheduler.update_from_output(so, _runner_output(so, {"B": [21]}))
    so = scheduler.schedule()
    assert so.num_scheduled_tokens == {"B": 1}
    assert so.kv_connector_metadata is None or so.kv_connector_metadata.requests == []

    _evidence(
        dict(
            test="test_prefill_save_then_prefix_hit_load",
            prompt_len=prompt_len,
            layers=len(LAYER_NAMES),
            block_size=BLOCK_SIZE,
            saved_tokens=saved_tokens,
            hit_tokens=hit_tokens,
            save_bytes=save_bytes_a,
            load_bytes=rep.load_bytes,
            bytes_per_token_per_layer=ROW_BYTES,
            on_disk_bytes=on_disk,
            prefill_step_s=round(prefill_s, 4),
            hit_step_s=round(hit_s, 4),
            connector="ExampleConnector",
            vllm="0.26.0",
            device="none (mock)",
            status="pass",
        )
    )


def test_no_hit_for_a_different_prompt(tmp_path):
    storage = str(tmp_path / "kv")
    os.makedirs(storage)
    config, kv_cache_config = _config(storage)
    scheduler = _scheduler(config, kv_cache_config)
    kv = {n: torch.zeros(NUM_BLOCKS, BLOCK_SIZE, LATENT_DIM, dtype=torch.bfloat16) for n in LAYER_NAMES}
    host = ConnectorHostRef(RecordingConnector(config, KVConnectorRole.WORKER, kv_cache_config), kv)
    scheduler.add_request(_request("A", _prompt(300, seed=1)))
    so = scheduler.schedule()
    host.step(so, mock_compute)
    scheduler.update_from_output(so, _runner_output(so, {"A": [1]}))
    scheduler.add_request(_request("B", _prompt(300, seed=2)))
    so = scheduler.schedule()
    nr = next(r for r in so.scheduled_new_reqs if r.req_id == "B")
    assert nr.num_computed_tokens == 0 and so.num_scheduled_tokens["B"] == 300
    assert all(r.is_store for r in so.kv_connector_metadata.requests)
    rep = host.step(so, mock_compute)
    assert rep.load_bytes == 0 and rep.save_bytes == len(LAYER_NAMES) * _align(300) * ROW_BYTES
    _evidence(dict(test="test_no_hit_for_a_different_prompt", prompt_len=300, status="pass", device="none (mock)"))
