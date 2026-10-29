# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A2: the full LMCache engine (CPU + disk tiers) with the TT connector selected through
LMCache's platform registry (``CreateGPUConnector`` -> extra_config["gpu_connector"]),
driven in the A3 harness sequence: request A prefill = store of align64(len-1) tokens;
request B, same prompt, on a cold mock device = lookup hit + retrieve, pages injected
bit-exactly; bytes = layers x tokens x 1,024."""

from __future__ import annotations

import time

import pytest
import torch
from lmcache.utils import EngineType, mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector

from vllm_tt_plugin.lmcache import page_contract as pc
from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector
from vllm_tt_plugin.lmcache.mock_paged_cache import MockTTPagedLatentCache

from .tt_lmcache_testing import (
    BLOCK_SIZE,
    LATENT_DIM,
    LAYER_NAMES,
    ROW_BYTES,
    align,
    evidence,
    fingerprint_rows,
    layer_index,
    pick_blocks,
    prompt,
    slot_mapping,
    ttnn_available,
)

NUM_BLOCKS = 96
CONNECTOR_SPEC = "vllm_tt_plugin.lmcache.connector:VLLMPagedMemTTConnector"
BACKENDS = ["torch"] + (["ttnn"] if ttnn_available() else [])


def _write(
    cache: MockTTPagedLatentCache, slots: torch.Tensor, positions, tokens
) -> None:
    for name in cache.layer_names:
        dev = (
            cache.device_pages(name).clone()
            if cache.backend == "ttnn"
            else cache.device_caches[name]
        )
        dev.reshape(-1, LATENT_DIM)[slots] = fingerprint_rows(
            layer_index(name), positions, tokens
        )
        if cache.backend == "ttnn":
            cache.device_caches[name] = cache._to_device(dev)


@pytest.fixture
def engine(tmp_path):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=pc.LMCACHE_CHUNK_TOKENS,
        local_cpu=True,
        max_local_cpu_size=0.5,
        local_disk=str(tmp_path / "disk"),
        max_local_disk_size=2.0,
        # prefix hits land on 64-token page boundaries inside a 256-token chunk
        save_unfull_chunk=True,
        lmcache_instance_id="a2-tt-engine",
        extra_config={
            "gpu_connector": CONNECTOR_SPEC,
            "tt_tokens_per_page": BLOCK_SIZE,
            # defaults to use_mla (True): one worker process here, the MLA rank
            # broadcast has no peers
            "save_only_first_rank": False,
        },
    )
    md = pc.lmcache_metadata(
        kv_connector_extra_config={"tt_tokens_per_page": BLOCK_SIZE}
    )
    conn = CreateGPUConnector(config, md, EngineType.VLLM)
    assert isinstance(conn, VLLMPagedMemTTConnector), (
        "LMCache's registry must select the TT connector on cpu"
    )
    eng = LMCacheEngineBuilder.get_or_create(
        "a2-tt-engine",
        config,
        md,
        conn,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )
    # as the vLLM adapter does: creates the StorageManager (local_cpu + local_disk)
    eng.post_init()
    try:
        yield eng, conn, config
    finally:
        LMCacheEngineBuilder.destroy("a2-tt-engine")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("prompt_len", [200, 2048])
def test_prefill_store_then_prefix_hit_retrieve(engine, backend, prompt_len):
    eng, conn, config = engine
    toks = prompt(prompt_len)
    saved = align(prompt_len)
    positions = list(range(prompt_len))

    # ---- request A: the model wrote its pages; the save-out stores align64(len - 1)
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend=backend)
    blocks_a = pick_blocks(prompt_len, NUM_BLOCKS, seed=11)
    assert 0 not in blocks_a
    slots_a = slot_mapping(blocks_a, prompt_len)
    _write(a, slots_a, positions, toks)
    conn.set_host_xfer_buffer_ops(a.copy_blocks)
    conn.register_device_kv_caches(a.device_caches)
    assert eng.lookup(toks[:saved]) == 0
    t0 = time.perf_counter()
    eng.store(
        torch.tensor(toks[:saved]),
        kvcaches=a.host_list,
        slot_mapping=slots_a[:saved],
        req_id="A",
    )
    store_s = time.perf_counter() - t0
    assert conn.stats["from_gpu_row_bytes"] == len(LAYER_NAMES) * saved * ROW_BYTES
    assert (
        conn.stats["d2h_pages"] == saved // BLOCK_SIZE
        and conn.stats["to_gpu_row_bytes"] == 0
    )
    assert eng.lookup(toks[:saved]) == saved
    # LMCache keys a partial trailing chunk by the hash of its exact span: a lookup of
    # a longer prompt matches only the full-chunk prefix (A1 must look up the saved
    # length, align64(len - 1), not the raw prompt).
    assert (
        eng.lookup(toks) == saved // pc.LMCACHE_CHUNK_TOKENS * pc.LMCACHE_CHUNK_TOKENS
    )

    # ---- request B, same prompt, cold device: hit + retrieve
    b = MockTTPagedLatentCache(NUM_BLOCKS, backend=backend)
    blocks_b = pick_blocks(prompt_len, NUM_BLOCKS, seed=12)
    slots_b = slot_mapping(blocks_b, prompt_len)
    conn.set_host_xfer_buffer_ops(b.copy_blocks)
    conn.register_device_kv_caches(b.device_caches)
    hit = eng.lookup(
        toks[: align(prompt_len)]
    )  # the adapter's "prompt minus last token" aligned to the page
    assert hit == saved
    t0 = time.perf_counter()
    ret = eng.retrieve(
        torch.tensor(toks[:hit]),
        kvcaches=b.host_list,
        slot_mapping=slots_b[:hit],
        req_id="B",
    )
    retrieve_s = time.perf_counter() - t0
    assert int(ret.sum()) == hit
    assert conn.stats["to_gpu_row_bytes"] == len(LAYER_NAMES) * hit * ROW_BYTES
    assert conn.stats["h2d_pages"] == hit // BLOCK_SIZE
    for name in LAYER_NAMES:
        got = b.device_tokens(name, slots_b[:hit])
        want = fingerprint_rows(layer_index(name), positions[:hit], toks[:hit])
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name
        assert torch.count_nonzero(b.device_tokens(name, slots_b[hit:])) == 0
    different = prompt(prompt_len, seed=99)
    assert eng.lookup(different) == 0
    evidence(
        dict(
            test="test_prefill_store_then_prefix_hit_retrieve",
            backend=backend,
            prompt_len=prompt_len,
            layers=len(LAYER_NAMES),
            block_size=BLOCK_SIZE,
            chunk_size=pc.LMCACHE_CHUNK_TOKENS,
            saved_tokens=saved,
            hit_tokens=hit,
            save_bytes=conn.stats["from_gpu_row_bytes"],
            load_bytes=conn.stats["to_gpu_row_bytes"],
            bytes_per_token_per_layer=ROW_BYTES,
            d2h_pages=conn.stats["d2h_pages"],
            h2d_pages=conn.stats["h2d_pages"],
            store_s=round(store_s, 4),
            retrieve_s=round(retrieve_s, 4),
            tiers="local_cpu+local_disk",
            connector="VLLMPagedMemTTConnector via CreateGPUConnector(cpu)",
            lmcache_c_ops="python_ops_fallback",
            vllm="0.26.0",
            device="none (mock)",
            status="pass",
        )
    )


def test_registry_requires_connector_spec(tmp_path):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        lmcache_instance_id="a2-tt-noreg",
    )
    with pytest.raises(RuntimeError, match="gpu_connector"):
        CreateGPUConnector(config, pc.lmcache_metadata(), EngineType.VLLM)
