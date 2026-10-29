# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A2: VLLMPagedMemTTConnector round trips on the mock paged latent cache (no device).

Template: LMCache tests/v1/test_gpu_connector.py (to_gpu/from_gpu/batched on paged
KV with a slot mapping, bitwise comparison), with the A3 harness conventions
(fingerprint rows, 1,024 B/token/layer counters, block 0 never used)."""

from __future__ import annotations

import time

import pytest
import torch
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat

from vllm_tt_plugin.lmcache import page_contract as pc
from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector
from vllm_tt_plugin.lmcache.mock_paged_cache import MockTTPagedLatentCache

from .tt_lmcache_testing import (
    BLOCK_SIZE,
    LATENT_DIM,
    LAYER_NAMES,
    ROW_BYTES,
    evidence,
    fingerprint_rows,
    layer_index,
    pick_blocks,
    prompt,
    slot_mapping,
    ttnn_available,
)

NUM_BLOCKS = 48
BACKENDS = ["torch"] + (["ttnn"] if ttnn_available() else [])


def _memobj(num_tokens: int, num_layers: int = pc.NUM_DSA_LAYERS):
    shape = torch.Size([1, num_layers, num_tokens, pc.LMCACHE_HIDDEN_ELEMS])
    return AdHocMemoryAllocator(device="cpu").allocate(
        [shape], [pc.LATENT_DTYPE], fmt=MemoryFormat.KV_MLA_FMT
    )


def _write_fingerprints(
    cache: MockTTPagedLatentCache, slots: torch.Tensor, positions, tokens
) -> None:
    """The model stand-in: write the scheduled tokens' rows into the mock device."""
    for name in cache.layer_names:
        dev = (
            cache.device_pages(name).clone()
            if cache.backend == "ttnn"
            else cache.device_caches[name]
        )
        flat = dev.reshape(-1, LATENT_DIM)
        flat[slots] = fingerprint_rows(layer_index(name), positions, tokens)
        if cache.backend == "ttnn":
            cache.device_caches[name] = cache._to_device(dev)


def _connector(cache: MockTTPagedLatentCache) -> VLLMPagedMemTTConnector:
    conn = VLLMPagedMemTTConnector.from_metadata(pc.lmcache_metadata())
    conn.set_host_xfer_buffer_ops(cache.copy_blocks)
    conn.register_device_kv_caches(cache.device_caches)
    return conn


def test_from_metadata_and_shape():
    md = pc.lmcache_metadata(chunk_size=256)
    conn = VLLMPagedMemTTConnector.from_metadata(
        md, use_gpu=True, device=torch.device("cpu")
    )
    assert conn.num_layers == 11 and conn.tokens_per_page == 64
    assert conn.get_shape(256) == torch.Size([1, 11, 256, 544])
    assert md.get_shapes(256) == [conn.get_shape(256)], (
        "LMCache sizes memobjs from metadata.kv_shape"
    )
    n_bytes = 256 * 544 * 2 * 11
    assert n_bytes == 4 * pc.PAGE_BYTES, "a 256-token chunk is exactly 4 contract pages"
    with pytest.raises(ValueError):
        VLLMPagedMemTTConnector.from_metadata(
            pc.lmcache_metadata(chunk_size=256, num_layers=11).__class__(
                **{**pc.lmcache_metadata().__dict__, "use_mla": False}
            )
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("prompt_len", [200, 2048])
def test_save_then_load_bit_exact(backend, prompt_len):
    """Device A (fingerprinted) -> memobjs via from_gpu; cold device B <- to_gpu.

    Bit-exact, bytes counted."""
    saved = (
        (prompt_len - 1) // BLOCK_SIZE * BLOCK_SIZE
    )  # align64(len - 1), as the A3 harness / contract §5
    toks = prompt(prompt_len)
    positions = list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend=backend)
    blocks_a = pick_blocks(prompt_len, NUM_BLOCKS, seed=1)
    slots_a = slot_mapping(blocks_a, prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    conn_a = _connector(a)

    # chunks of 256 tokens as LMCache cuts them (last one partial, whole pages)
    chunks = [(s, min(s + 256, saved)) for s in range(0, saved, 256)]
    objs = [_memobj(e - s) for s, e in chunks]
    t0 = time.perf_counter()
    conn_a.batched_from_gpu(
        objs,
        [s for s, _ in chunks],
        [e for _, e in chunks],
        kvcaches=a.host_list,
        slot_mapping=slots_a,
    )
    save_s = time.perf_counter() - t0
    assert conn_a.stats["from_gpu_row_bytes"] == len(LAYER_NAMES) * saved * ROW_BYTES
    assert (
        conn_a.stats["d2h_calls"] == 1
        and conn_a.stats["d2h_pages"] == saved // BLOCK_SIZE
    )
    assert a.calls[0][0] == "d2h" and a.calls[0][1] == blocks_a[: saved // BLOCK_SIZE]
    for obj in objs:
        assert obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        rows = obj.tensor[0]  # [L, n, 544]
        assert torch.count_nonzero(rows[..., LATENT_DIM:]) == 0, (
            "reserved pooled-key slice is zero in v1"
        )

    b = MockTTPagedLatentCache(NUM_BLOCKS, backend=backend)  # cold: zeros
    blocks_b = pick_blocks(prompt_len, NUM_BLOCKS, seed=2)
    slots_b = slot_mapping(blocks_b, prompt_len)
    conn_b = _connector(b)
    t0 = time.perf_counter()
    conn_b.batched_to_gpu(
        objs,
        [s for s, _ in chunks],
        [e for _, e in chunks],
        kvcaches=b.host_list,
        slot_mapping=slots_b,
    )
    load_s = time.perf_counter() - t0
    assert conn_b.stats["to_gpu_row_bytes"] == len(LAYER_NAMES) * saved * ROW_BYTES
    assert (
        conn_b.stats["h2d_calls"] == 1
        and conn_b.stats["h2d_pages"] == saved // BLOCK_SIZE
    )
    for name in LAYER_NAMES:
        got = b.device_tokens(name, slots_b[:saved])
        want = fingerprint_rows(layer_index(name), positions[:saved], toks[:saved])
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name
        rest = b.device_tokens(name, slots_b[saved:])
        assert torch.count_nonzero(rest) == 0, (
            "tokens past the saved prefix are untouched"
        )
    evidence(
        dict(
            test="test_save_then_load_bit_exact",
            backend=backend,
            prompt_len=prompt_len,
            layers=len(LAYER_NAMES),
            saved_tokens=saved,
            pages=saved // BLOCK_SIZE,
            save_row_bytes=conn_a.stats["from_gpu_row_bytes"],
            load_row_bytes=conn_b.stats["to_gpu_row_bytes"],
            page_bytes_moved=saved
            // BLOCK_SIZE
            * pc.LATENT_BYTES_PER_PAGE_LAYER
            * len(LAYER_NAMES),
            bytes_per_token_per_layer=ROW_BYTES,
            save_s=round(save_s, 4),
            load_s=round(load_s, 4),
            connector="VLLMPagedMemTTConnector",
            device="none (mock)",
            status="pass",
        )
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_chunk_partial_page_roundtrip(backend):
    """to_gpu/from_gpu with a partial last page (100 tokens = 1 full page + 36)."""
    n = 100
    a = MockTTPagedLatentCache(
        NUM_BLOCKS, backend=backend, seed=3
    )  # random device content
    blocks = [7, 3]
    slots = slot_mapping(blocks, n)
    conn = _connector(a)
    obj = _memobj(n)
    conn.from_gpu(obj, 0, n, kvcaches=a.host_list, slot_mapping=slots)
    for li, name in enumerate(LAYER_NAMES):
        want = a.device_tokens(name, slots)
        got = obj.tensor[0, li, :, :LATENT_DIM]
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    b = MockTTPagedLatentCache(NUM_BLOCKS, backend=backend)
    slots_b = slot_mapping([5, 9], n)
    conn_b = _connector(b)
    conn_b.to_gpu(obj, 0, n, kvcaches=b.host_list, slot_mapping=slots_b)
    for name in LAYER_NAMES:
        assert torch.equal(
            b.device_tokens(name, slots_b).view(torch.int16),
            a.device_tokens(name, slots).view(torch.int16),
        )
        # the 28 rows after token 100 in block 9 were not in the chunk: still zero on B
        tail = b.device_pages(name).reshape(-1, LATENT_DIM)[
            9 * BLOCK_SIZE + 36 : 10 * BLOCK_SIZE
        ]
        assert torch.count_nonzero(tail) == 0


def test_pooled_key_buffers_roundtrip():
    """With pooled-key host buffers registered the reserved slice carries them (B1)."""
    a = MockTTPagedLatentCache(NUM_BLOCKS, seed=4, with_pooled_keys=True)
    for t in a.pooled_host.values():
        t.copy_(torch.randn(t.shape).to(t.dtype))
    conn = _connector(a)
    conn.register_pooled_key_buffers(a.pooled_host)
    slots = slot_mapping([2, 4, 6, 8], 256)
    obj = _memobj(256)
    conn.from_gpu(obj, 0, 256, kvcaches=a.host_list, slot_mapping=slots)
    pooled = (
        obj.tensor[0, :, :, LATENT_DIM:].contiguous().view(torch.uint8)
    )  # [L, 256, 64 B]
    for li, name in enumerate(LAYER_NAMES):
        want = (
            a.pooled_host[name][[2, 4, 6, 8]]
            .reshape(4, -1)
            .view(torch.uint8)
            .reshape(256, 64)
        )
        assert torch.equal(pooled[li], want)
    b = MockTTPagedLatentCache(NUM_BLOCKS, with_pooled_keys=True)
    conn_b = _connector(b)
    conn_b.register_pooled_key_buffers(b.pooled_host)
    conn_b.to_gpu(
        obj, 0, 256, kvcaches=b.host_list, slot_mapping=slot_mapping([1, 3, 5, 7], 256)
    )
    for name in LAYER_NAMES:
        assert torch.equal(
            b.pooled_host[name][[1, 3, 5, 7]], a.pooled_host[name][[2, 4, 6, 8]]
        )


def test_slot_mapping_validation():
    a = MockTTPagedLatentCache(NUM_BLOCKS)
    conn = _connector(a)
    conn.initialize_kvcaches_ptr(kvcaches=a.host_list)
    assert conn.num_blocks == NUM_BLOCKS
    assert conn.pages_from_slot_mapping(slot_mapping([5, 2, 9], 150), 0, 150) == [
        (5, 64),
        (2, 64),
        (9, 22),
    ]
    assert conn.pages_from_slot_mapping(slot_mapping([5, 2, 9], 192), 64, 192) == [
        (2, 64),
        (9, 64),
    ]
    with pytest.raises(ValueError, match="page aligned"):
        conn.pages_from_slot_mapping(slot_mapping([5, 2], 128), 10, 128)
    bad = slot_mapping([5, 2], 128)
    bad[70] += 1
    with pytest.raises(ValueError, match="not contiguous"):
        conn.pages_from_slot_mapping(bad, 0, 128)
    with pytest.raises(ValueError, match="page boundary"):
        conn.pages_from_slot_mapping(slot_mapping([5], 64)[3:], 0, 61)
    with pytest.raises(ValueError, match="negative slot"):
        conn.pages_from_slot_mapping(torch.full((64,), -1, dtype=torch.long), 0, 64)
    with pytest.raises(ValueError, match="slot_mapping"):
        conn.to_gpu(_memobj(64), 0, 64, kvcaches=a.host_list)
    with pytest.raises(ValueError, match="per-layer host buffers"):
        VLLMPagedMemTTConnector.from_metadata(
            pc.lmcache_metadata()
        ).initialize_kvcaches_ptr(kvcaches=a.host_list[:3])
    with pytest.raises(ValueError, match="B per block"):
        VLLMPagedMemTTConnector.from_metadata(
            pc.lmcache_metadata()
        ).initialize_kvcaches_ptr(
            kvcaches=[
                torch.zeros(NUM_BLOCKS, 64, 256, dtype=torch.bfloat16)
                for _ in LAYER_NAMES
            ]
        )
    obj = _memobj(64)
    obj.metadata.fmt = MemoryFormat.KV_2LTD
    with pytest.raises(ValueError, match="KV_MLA_FMT"):
        conn.to_gpu(
            obj, 0, 64, kvcaches=a.host_list, slot_mapping=slot_mapping([5], 64)
        )


def test_pages_of_ranges_equals_the_per_range_translation():
    """pages_of_ranges (one vectorized pass for a batched call) == pages_from_slot_mapping per range: chunked ranges
    tiling the prompt, gaps / out of order, empty ranges, a partial last page per range; every invalid mapping raises
    the per-range translation's own error"""
    a = MockTTPagedLatentCache(NUM_BLOCKS)
    conn = _connector(a)
    g = torch.Generator().manual_seed(3)
    blocks = [int(b) for b in torch.randperm(NUM_BLOCKS, generator=g)[:20]]
    sm = slot_mapping(blocks, 20 * 64 - 17)
    n = sm.numel()
    cases = [
        ([0, 256, 512, 768, 1024], [256, 512, 768, 1024, n]),  # LMCache's chunks
        ([512, 0, 1024], [768, 192, n]),  # out of order, gaps, partial pages mid-prompt
        ([64, 64, 128], [64, 128, 130]),  # an empty range, a 2-token page
        ([0], [n]),
    ]
    for st, en in cases:
        assert conn.pages_of_ranges(sm, st, en) == [conn.pages_from_slot_mapping(sm, s, e) for s, e in zip(st, en)]
        assert conn.pages_of_ranges(sm.tolist(), st, en) == conn.pages_of_ranges(sm, st, en)
    bad = sm.clone()
    bad[300] += 1
    with pytest.raises(ValueError, match="not contiguous"):
        conn.pages_of_ranges(bad, [0, 256], [256, 512])
    assert conn.pages_of_ranges(bad, [0, 512], [256, 768]) == [conn.pages_from_slot_mapping(bad, 0, 256),
                                                               conn.pages_from_slot_mapping(bad, 512, 768)]  # uncovered
    neg = sm.clone()
    neg[256:320] = -1
    with pytest.raises(ValueError, match="negative slot"):
        conn.pages_of_ranges(neg, [0, 256], [256, 512])
    with pytest.raises(ValueError, match="page aligned"):
        conn.pages_of_ranges(sm, [0, 250], [250, 512])
    with pytest.raises(ValueError, match="page boundary"):
        conn.pages_of_ranges(sm[3:], [0], [61])
    with pytest.raises(ValueError, match="covers"):
        conn.pages_of_ranges(sm[:100], [0], [128])


def test_host_buffers_without_copy_op_are_the_cache():
    """Host-resident path: without a CopyBlocksOp the host transfer buffers are the
    paged cache (vLLM MLA shape)."""
    conn = VLLMPagedMemTTConnector.from_metadata(pc.lmcache_metadata())
    kv = {
        n: torch.randn(NUM_BLOCKS, BLOCK_SIZE, LATENT_DIM).to(torch.bfloat16)
        for n in LAYER_NAMES
    }
    slots = slot_mapping([4, 8, 12, 16], 256)
    obj = _memobj(256)
    conn.from_gpu(
        obj, 0, 256, kvcaches=kv, slot_mapping=slots
    )  # dict form, as the adapter's kv_caches
    assert conn.layer_names == list(LAYER_NAMES)
    kv2 = {n: torch.zeros_like(t) for n, t in kv.items()}
    conn.to_gpu(obj, 0, 256, kvcaches=kv2, slot_mapping=slots)
    for n in LAYER_NAMES:
        assert (
            torch.equal(kv2[n].view(torch.int16), kv[n].view(torch.int16)) is False
        )  # untouched blocks differ
        assert torch.equal(
            kv2[n][[4, 8, 12, 16]].view(torch.int16),
            kv[n][[4, 8, 12, 16]].view(torch.int16),
        )


class _FakeHitDma:
    """The model's LatentHitDma contract (GLM53_LMCACHE_PAGE_DMA): submit(pages in
    ITS layer order, block ids) then wait(); here it writes the row-major pages
    straight into the mock device caches (the tilize + descriptor execution is
    the model tree's test_hit_dma.py)."""

    def __init__(self, cache: MockTTPagedLatentCache):
        self.cache, self.submits, self.waits = cache, [], 0
        self.layer_names = list(reversed(cache.layer_names))  # a different order

    def submit(self, pages, block_ids):
        self.submits.append(list(block_ids))
        for name, layer_pages in zip(self.layer_names, pages, strict=True):
            dev = self.cache.device_caches[name]
            rows = torch.from_numpy(layer_pages).view(pc.LATENT_DTYPE)
            dev.view(dev.shape[0], -1)[torch.as_tensor(block_ids)] = rows.reshape(len(block_ids), -1)

    def wait(self):
        self.waits += 1


def test_hit_pages_go_through_the_page_dma():
    """GLM53_LMCACHE_PAGE_DMA: a hit's h2d hop is ONE submit of the union of
    pages to the model's page-DMA (in its layer order), not copy_blocks; the
    loaded rows are bit-exact."""
    prompt_len = 600
    saved = (prompt_len - 1) // BLOCK_SIZE * BLOCK_SIZE
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    blocks_a = pick_blocks(prompt_len, NUM_BLOCKS, seed=1)
    slots_a = slot_mapping(blocks_a, prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    chunks = [(s, min(s + 256, saved)) for s in range(0, saved, 256)]
    objs = [_memobj(e - s) for s, e in chunks]
    _connector(a).batched_from_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks],
                                   kvcaches=a.host_list, slot_mapping=slots_a)  # fmt: skip

    b = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    blocks_b = pick_blocks(prompt_len, NUM_BLOCKS, seed=2)
    slots_b = slot_mapping(blocks_b, prompt_len)
    conn_b = _connector(b)
    dma = _FakeHitDma(b)
    conn_b.set_hit_dma(dma)
    calls_before = len(b.calls)
    conn_b.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks],
                          kvcaches=b.host_list, slot_mapping=slots_b)  # fmt: skip
    assert len(b.calls) == calls_before, "copy_blocks not used for the hit"
    assert dma.submits == [blocks_b[: saved // BLOCK_SIZE]]
    assert conn_b.stats["h2d_dma_pages"] == saved // BLOCK_SIZE and conn_b.stats["h2d_calls"] == 1
    # whole pages: straight from the memory objects, the row-major host buffers never written
    assert conn_b.stats["h2d_dma_direct_pages"] == saved // BLOCK_SIZE
    assert all(torch.count_nonzero(t) == 0 for t in b.host_list), "host buffers untouched on the direct path"
    for name in LAYER_NAMES:
        got = b.device_tokens(name, slots_b[:saved])
        want = fingerprint_rows(layer_index(name), positions[:saved], toks[:saved])
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name
        assert torch.count_nonzero(b.device_tokens(name, slots_b[saved:])) == 0
    with pytest.raises(ValueError, match="page-DMA layers"):
        bad = _FakeHitDma(b)
        bad.layer_names = bad.layer_names[1:]
        conn_b.set_hit_dma(bad)


class _FakeObjDma(_FakeHitDma):
    """Stage 12 lever (c): the model's LatentHitDma.submit_objects contract -- [(rows, m)] per object, rows a strided
    [L, >= 64 m, >= 1,024 B] uint8 view in the CONNECTOR's layer order, layer_order[j] = the object layer for the
    model's layer j; here it rebuilds row-major pages from the strided rows and writes them like submit()"""

    slots = 1 << 20

    def __init__(self, cache):
        super().__init__(cache)
        self.object_submits = []

    def submit_objects(self, objects, block_ids, layer_order=None):
        self.object_submits.append((len(objects), list(block_ids)))
        pages = []
        for j in range(len(self.layer_names)):
            i = layer_order[j] if layer_order is not None else j
            rows = torch.cat([r[i, : m * BLOCK_SIZE, :1024] for r, m in objects]).contiguous()
            pages.append(rows.view(-1, BLOCK_SIZE * 1024).numpy())
        self.submit(pages, block_ids)


def test_hit_pages_go_through_the_fused_object_submit(monkeypatch):
    """lever (c): with the model's submit_objects the hit's h2d is ONE fused submit of the objects' strided rows (no
    cat), bit-exact like the cat path; GLM53_KV_LOAD_FUSED=0 keeps the cat path"""
    prompt_len = 600
    saved = (prompt_len - 1) // BLOCK_SIZE * BLOCK_SIZE
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    slots_a = slot_mapping(pick_blocks(prompt_len, NUM_BLOCKS, seed=1), prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    chunks = [(s, min(s + 256, saved)) for s in range(0, saved, 256)]
    objs = [_memobj(e - s) for s, e in chunks]
    _connector(a).batched_from_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks],
                                   kvcaches=a.host_list, slot_mapping=slots_a)  # fmt: skip
    for fused in ("1", "0"):
        monkeypatch.setenv("GLM53_KV_LOAD_FUSED", fused)
        b = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
        blocks_b = pick_blocks(prompt_len, NUM_BLOCKS, seed=2)
        slots_b = slot_mapping(blocks_b, prompt_len)
        conn_b = _connector(b)
        dma = _FakeObjDma(b)
        conn_b.set_hit_dma(dma)
        conn_b.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks],
                              kvcaches=b.host_list, slot_mapping=slots_b)  # fmt: skip
        assert (len(dma.object_submits) == 1) == (fused == "1")
        assert dma.submits == [blocks_b[: saved // BLOCK_SIZE]]
        assert conn_b.stats.get("h2d_fused_submits", 0) == (1 if fused == "1" else 0)
        for name in LAYER_NAMES:
            got = b.device_tokens(name, slots_b[:saved])
            want = fingerprint_rows(layer_index(name), positions[:saved], toks[:saved])
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (fused, name)


class _FakeReadDma(_FakeHitDma):
    """+ the store direction (the model's LatentHitDma.read_pages): rows of the mock device caches, ITS layer order,
    as [n, 64, 512] 2-byte arrays"""

    def __init__(self, cache):
        super().__init__(cache)
        self.reads = []

    def read_pages(self, block_ids):
        self.reads.append(list(block_ids))
        out = []
        for name in self.layer_names:
            dev = self.cache.device_caches[name]
            rows = dev.view(dev.shape[0], -1)[torch.as_tensor(block_ids)].contiguous()
            out.append(rows.view(torch.int16).numpy().view("u2").reshape(len(block_ids), 64, -1))
        return out


def test_store_pages_go_through_the_page_dma_read(monkeypatch):
    """Stage 12 item 2: with the model's page-DMA the store's d2h hop is ONE read_pages of the page union (no
    copy_blocks); the memory objects equal the copy_blocks path's bitwise; GLM53_KV_T2_STORE=0 is the old path."""
    prompt_len = 600
    saved = (prompt_len - 1) // BLOCK_SIZE * BLOCK_SIZE
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    blocks_a = pick_blocks(prompt_len, NUM_BLOCKS, seed=1)
    slots_a = slot_mapping(blocks_a, prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    chunks = [(s, min(s + 256, saved)) for s in range(0, saved, 256)]
    ref = [_memobj(e - s) for s, e in chunks]
    _connector(a).batched_from_gpu(ref, [s for s, _ in chunks], [e for _, e in chunks],
                                   kvcaches=a.host_list, slot_mapping=slots_a)  # fmt: skip
    for knob, via_dma in (("1", True), ("0", False)):
        monkeypatch.setenv("GLM53_KV_T2_STORE", knob)
        conn = _connector(a)
        dma = _FakeReadDma(a)
        conn.set_hit_dma(dma)
        calls_before = len(a.calls)
        got = [_memobj(e - s) for s, e in chunks]
        conn.batched_from_gpu(got, [s for s, _ in chunks], [e for _, e in chunks],
                              kvcaches=a.host_list, slot_mapping=slots_a)  # fmt: skip
        assert (len(a.calls) == calls_before) == via_dma
        assert dma.reads == ([blocks_a[: saved // BLOCK_SIZE]] if via_dma else [])
        assert conn.stats.get("d2h_dma_pages", 0) == (saved // BLOCK_SIZE if via_dma else 0)
        for r, g in zip(ref, got, strict=True):
            assert torch.equal(r.tensor.view(torch.int16), g.tensor.view(torch.int16))


@pytest.mark.parametrize("backend", BACKENDS)
def test_partial_page_load_does_not_carry_an_earlier_requests_rows(backend):
    """round-5 cross-request leak check (program 2026-10-23T13:58:48Z): the host transfer page of block 9 still
    holds an EARLIER request's rows (its store's d2h); a partial-page load of 36 rows into block 9 moves the whole page
    h2d -- rows 36..63 on the device must be zero, never the earlier request's"""
    n = 100
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend=backend, seed=3)
    slots = slot_mapping([7, 3], n)
    obj = _memobj(n)
    _connector(a).from_gpu(obj, 0, n, kvcaches=a.host_list, slot_mapping=slots)
    b = MockTTPagedLatentCache(NUM_BLOCKS, backend=backend)
    for t in b.host_list:  # the earlier request's page 9 in the host transfer buffer
        t.view(t.shape[0], -1)[9] = 7.0
    slots_b = slot_mapping([5, 9], n)
    _connector(b).to_gpu(obj, 0, n, kvcaches=b.host_list, slot_mapping=slots_b)
    for name in LAYER_NAMES:
        assert torch.equal(b.device_tokens(name, slots_b).view(torch.int16), a.device_tokens(name, slots).view(torch.int16))
        tail = b.device_pages(name).reshape(-1, LATENT_DIM)[9 * BLOCK_SIZE + 36 : 10 * BLOCK_SIZE]
        assert torch.count_nonzero(tail) == 0, "the earlier request's rows reached the device"


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("n", [64, 100, 300, 600])
def test_bulk_host_copies_match_the_per_page_loop(monkeypatch, pooled, n):
    """GLM53_LMCACHE_BULK_COPY (default 1, n29h7 M1 2026-10-23: the per-page, per-layer slice copies held the engine
    thread): one gather / scatter per layer gives the per-page loop's bytes both ways -- memory object after a store,
    host pages after a load, incl. a partial last page (zeros past its tokens) and the pooled-key slice"""
    blocks = [11, 3, 7, 19, 5, 23, 2, 13, 17, 29][: -(-n // BLOCK_SIZE)]
    slots = slot_mapping(blocks, n)
    objs, hosts = {}, {}
    for bulk in ("0", "1"):
        monkeypatch.setenv("GLM53_LMCACHE_BULK_COPY", bulk)
        monkeypatch.setattr(VLLMPagedMemTTConnector, "_BULK_PAGES", 3)  # several gather chunks at these sizes
        a = MockTTPagedLatentCache(NUM_BLOCKS, seed=8, with_pooled_keys=pooled)
        if pooled:
            for t in a.pooled_host.values():
                t.copy_(torch.randn(t.shape, generator=torch.Generator().manual_seed(1)).to(t.dtype))
        conn = _connector(a)
        if pooled:
            conn.register_pooled_key_buffers(a.pooled_host)
        obj = _memobj(n)
        conn.from_gpu(obj, 0, n, kvcaches=a.host_list, slot_mapping=slots)
        objs[bulk] = obj.tensor.clone()
        b = MockTTPagedLatentCache(NUM_BLOCKS, with_pooled_keys=pooled)
        for t in b.host_list:  # stale rows of an earlier request in every host page
            t.view(t.shape[0], -1)[:] = 7.0
        conn_b = _connector(b)
        if pooled:
            conn_b.register_pooled_key_buffers(b.pooled_host)
        conn_b.to_gpu(obj, 0, n, kvcaches=b.host_list, slot_mapping=slot_mapping(blocks[::-1], n))
        hosts[bulk] = [t.clone() for t in b.host_list] + ([t.clone() for t in b.pooled_host.values()] if pooled else [])
    assert torch.equal(objs["0"].view(torch.int16), objs["1"].view(torch.int16))
    for x, y in zip(hosts["0"], hosts["1"]):
        assert torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x, y.view(torch.int16) if y.dtype == torch.bfloat16 else y)


def test_store_job_snapshot_matches_the_synchronous_store_and_owns_its_rows():
    """GLM53_KV_STORE_WORKER: prefetch_store_pages reads a store's pages on the engine thread into a job snapshot; the
    worker's batched_from_gpu with the job set gives the synchronous store's bytes even after the device pages change
    (a later step, a reused block), and makes no device call"""
    n = 150
    a = MockTTPagedLatentCache(NUM_BLOCKS, seed=5)
    conn = _connector(a)
    slots = slot_mapping([7, 3, 9], n)
    obj_sync = _memobj(n)
    conn.from_gpu(obj_sync, 0, n, kvcaches=a.host_list, slot_mapping=slots)
    job = conn.prefetch_store_pages([7, 3, 9, 3])
    assert sorted(job["row"]) == [3, 7, 9] and job["nbytes"] > 0
    _write_fingerprints(a, slots, list(range(n)), [t + 1 for t in prompt(n)])  # the device moves on
    moves = conn.stats.get("d2h_calls", 0)
    obj_job = _memobj(n)
    conn.set_store_job(job)
    try:
        conn.batched_from_gpu([obj_job], [0], [n], kvcaches=a.host_list, slot_mapping=slots)
    finally:
        conn.set_store_job(None)
    assert conn.stats.get("d2h_calls", 0) == moves, "the job store made a device read"
    assert torch.equal(obj_job.tensor.view(torch.int16), obj_sync.tensor.view(torch.int16))
    obj_now = _memobj(n)
    conn.from_gpu(obj_now, 0, n, kvcaches=a.host_list, slot_mapping=slots)
    assert not torch.equal(obj_now.tensor.view(torch.int16), obj_sync.tensor.view(torch.int16))


def test_store_job_direct_read_equals_the_shared_row_snapshot():
    """with the plugin's own page copy (tt_copy_kv_blocks) prefetch_store_pages reads every layer straight into the job;
    the bytes equal the move-into-the-host-rows + snapshot path, and the shared host rows are left alone"""
    import vllm_tt_plugin.kv_connector as K

    a = MockTTPagedLatentCache(NUM_BLOCKS, seed=6)
    conn = _connector(a)
    conn.initialize_kvcaches_ptr(kvcaches=a.host_list)
    ids = [11, 2, 17, 5]
    job_snapshot = conn.prefetch_store_pages(ids)
    before = [t.clone() for t in a.host_list]
    conn.copy_blocks = K.tt_copy_kv_blocks
    job_direct = conn.prefetch_store_pages(ids)
    assert job_direct["row"] == job_snapshot["row"]
    for x, y in zip(job_direct["bytes"], job_snapshot["bytes"]):
        assert torch.equal(x, y)
    for x, y in zip(a.host_list, before):
        assert torch.equal(x.view(torch.int16), y.view(torch.int16)), "the direct read wrote the shared host rows"


class _EnqueuedHandle:
    """a device cache whose page reads can be ENQUEUED (the sharded pool's read_pages_async): the pending object's
    result() is the pages as the queue read them; the log records which thread enqueues / completes"""

    def __new__(cls, tensor, log):
        import vllm_tt_plugin.kv_connector as K

        class H(K.TTPagedCacheHandle):
            def read_pages(self, ids):
                return K._read_pages(self.device_tensor, ids)

            def read_pages_async(self, ids):
                import threading

                log.append(("enqueue", threading.current_thread().name))
                pages = self.read_pages(ids).clone()  # in queue order: the pages as the step left them

                class Pending:
                    nbytes = pages.numel() * pages.element_size()

                    def result(self):
                        log.append(("result", threading.current_thread().name))
                        return pages

                return Pending()

        return H(tensor, tuple(tensor.shape), tensor.dtype)


@pytest.mark.parametrize("async_read", ["1", "0"])
def test_store_job_enqueued_read_completes_on_the_worker(monkeypatch, async_read):
    """GLM53_KV_STORE_ASYNC_READ (cf3h): prefetch_store_pages only ENQUEUES the handles' page reads (the job carries the
    pending reads and their byte count, nothing is waited for); complete_store_job -- the worker's first step -- turns
    them into the bytes of the direct read; 0 reads now (the direct read)"""
    import threading

    import vllm_tt_plugin.kv_connector as K

    monkeypatch.setenv("GLM53_KV_STORE_ASYNC_READ", async_read)
    a = MockTTPagedLatentCache(NUM_BLOCKS, seed=8)
    conn = _connector(a)
    conn.initialize_kvcaches_ptr(kvcaches=a.host_list)
    conn.copy_blocks = K.tt_copy_kv_blocks
    ids = [11, 2, 17, 5]
    want = conn.prefetch_store_pages(ids)
    log: list = []
    conn.register_device_kv_caches({n: _EnqueuedHandle(t, log) for n, t in a.device_caches.items()})
    job = conn.prefetch_store_pages(ids)
    assert job["nbytes"] == want["nbytes"]
    if async_read == "1":
        assert not any(isinstance(t, torch.Tensor) for t in job["bytes"]) and {k for k, _ in log} == {"enqueue"}
        assert conn.stats["d2h_enqueued"] == len(a.device_caches)
        th = threading.Thread(target=conn.complete_store_job, args=(job,), name="tt-kv-store-worker")
        th.start()
        th.join()
        assert [w for k, w in log if k == "result"] == ["tt-kv-store-worker"] * len(a.device_caches)
    else:
        assert log == [] and all(isinstance(t, torch.Tensor) for t in job["bytes"])
    conn.complete_store_job(job)  # idempotent: rows stay rows
    for x, y in zip(job["bytes"], want["bytes"]):
        assert torch.equal(x, y)


def test_t2_store_fill_in_one_copy_equals_the_per_page_fill(monkeypatch):
    """Stage 12 put/get (program 2026-10-24T02:08:04Z): the page-DMA store fills each memory object in one copy per
    column group (_fill_pages) -- bitwise the per-page, per-layer fill it replaced; with pooled keys from the pooled
    source, a partial last page, and a page stored twice (its second object takes the per-page fallback)"""
    prompt_len = 700
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    blocks_a = pick_blocks(prompt_len, NUM_BLOCKS, seed=3)
    slots_a = slot_mapping(blocks_a, prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    rng = torch.Generator().manual_seed(4)
    store = {n: torch.randint(0, 256, (NUM_BLOCKS, 4096), generator=rng, dtype=torch.uint8) for n in a.layer_names}
    chunks = [(0, 256), (256, 512), (512, 650), (128, 256)]  # 650: a partial page; (128, 256) repeats pages 2-3

    def run(fill_on: bool):
        conn = _connector(a)
        dma = _FakeReadDma(a)
        conn.set_hit_dma(dma)
        conn.enable_pooled_pages(lambda ids: [store[n][torch.as_tensor(ids)] for n in dma.layer_names])
        if not fill_on:
            monkeypatch.setattr(conn, "_fill_pages", lambda *a_, **k_: False)
        objs = [_memobj(e - s) for s, e in chunks]
        conn.batched_from_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks],
                              kvcaches=a.host_list, slot_mapping=slots_a)  # fmt: skip
        return objs

    new, old = run(True), run(False)
    for x, y in zip(new, old, strict=True):
        assert torch.equal(x.tensor.view(torch.uint8), y.tensor.view(torch.uint8))
    assert any(bool(t.tensor.view(torch.uint8)[..., 1024:].any()) for t in new), "the pooled columns were filled"


def test_loaded_pooled_slices_are_the_objects_pooled_columns_per_page():
    """_keep_loaded_pooled (one allocation per object): each page's [L, T x 64] slice = the object's pooled columns of
    that page, a partial page zero-padded"""
    conn = VLLMPagedMemTTConnector.from_metadata(pc.lmcache_metadata())
    conn.enable_pooled_pages(lambda ids: [])
    obj = _memobj(150)  # pages of 64, 64, 22 tokens
    rows = obj.tensor.view(torch.uint8).view(pc.NUM_DSA_LAYERS, 150, -1)
    rows.copy_(torch.randint(0, 256, rows.shape, dtype=torch.uint8))
    conn._keep_loaded_pooled(obj, [(5, 64), (9, 64), (2, 22)])
    pooled = rows[..., 1024:]
    for k, (page, cnt) in enumerate([(5, 64), (9, 64), (2, 22)]):
        want = torch.zeros(pc.NUM_DSA_LAYERS, 64, pooled.shape[-1], dtype=torch.uint8)
        want[:, :cnt] = pooled[:, k * 64 : k * 64 + cnt]
        assert torch.equal(conn.loaded_pooled[page], want.reshape(pc.NUM_DSA_LAYERS, -1))


@pytest.mark.parametrize("t2", [False, True])
def test_store_job_with_pooled_keys_equals_the_inline_store(t2):
    """cf3h-c: a store-worker job of a pooled-pages connector (prefetch_store_pages reads the pooled source; the
    worker's batched_from_gpu fills from the job in one copy per column group) == the inline store, bitwise"""
    prompt_len = 700
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    blocks_a = pick_blocks(prompt_len, NUM_BLOCKS, seed=5)
    slots_a = slot_mapping(blocks_a, prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    rng = torch.Generator().manual_seed(6)
    store = {n: torch.randint(0, 256, (NUM_BLOCKS, 4096), generator=rng, dtype=torch.uint8) for n in a.layer_names}
    chunks = [(0, 256), (256, 512), (512, 640)]

    def conn_for():
        c = _connector(a)
        c.initialize_kvcaches_ptr(kvcaches=a.host_list)
        if t2:
            c.set_hit_dma(_FakeReadDma(a))
        names = c.hit_dma.layer_names if t2 else c.layer_names
        c.enable_pooled_pages(lambda ids: [store[n][torch.as_tensor(ids)] for n in names])
        return c

    inline = [_memobj(e - s) for s, e in chunks]
    conn_for().batched_from_gpu(inline, [s for s, _ in chunks], [e for _, e in chunks],
                                kvcaches=a.host_list, slot_mapping=slots_a)  # fmt: skip
    c = conn_for()
    pages = sorted({int(x) // BLOCK_SIZE for x in slots_a[:640].tolist()})
    job = c.prefetch_store_pages(pages)
    assert job["pooled"] is not None
    worker = [_memobj(e - s) for s, e in chunks]
    c.set_store_job(job)
    try:
        c.batched_from_gpu(worker, [s for s, _ in chunks], [e for _, e in chunks],
                           kvcaches=a.host_list, slot_mapping=slots_a)  # fmt: skip
    finally:
        c.set_store_job(None)
    for x, y in zip(inline, worker, strict=True):
        assert torch.equal(x.tensor.view(torch.uint8), y.tensor.view(torch.uint8))
    assert any(bool(t.tensor.view(torch.uint8)[..., 1024:].any()) for t in worker)


def test_keep_loaded_pooled_many_equals_the_per_object_keep():
    """lever (c): the DMA path's one-buffer pooled gather leaves exactly the per-object path's slices per page"""
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    objs = [_memobj(n) for n in (256, 256, 128, 64)]
    g = torch.Generator().manual_seed(9)
    for o in objs:
        o.tensor.view(torch.uint8).copy_(torch.randint(0, 256, o.tensor.view(torch.uint8).shape, generator=g, dtype=torch.uint8))
    spans, nxt = [], 0
    for o in objs:
        m = o.tensor.shape[2] // BLOCK_SIZE
        spans.append([(nxt + j, BLOCK_SIZE) for j in range(m)])
        nxt += m
    one, many = _connector(a), _connector(a)
    for c in (one, many):
        c.pooled_source = lambda ids: None  # only "pooled pages on" matters here
    for o, pages in zip(objs, spans):
        one._keep_loaded_pooled(o, pages)
    many._keep_loaded_pooled_many(list(zip(objs, spans)))
    assert sorted(one.loaded_pooled) == sorted(many.loaded_pooled) == list(range(nxt))
    for p in range(nxt):
        assert torch.equal(one.loaded_pooled[p], many.loaded_pooled[p]), p
    assert [torch.equal(x, y) for x, y in zip(one.take_loaded_pooled(list(range(nxt))),
                                              many.take_loaded_pooled(list(range(nxt))))] == [True] * pc.NUM_DSA_LAYERS
