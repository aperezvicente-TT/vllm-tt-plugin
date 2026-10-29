# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 12 x 14: LMCache pages for the sequence-sharded DSA pool on a FAKE 8-chip mesh (host tensors, no device).

The model hook hands every layer's LOCAL pool plus the block map (``kv_cache_connector_shard``); the host wraps them
in ``TTShardedPagedCacheHandle``; ``VLLMPagedMemTTConnector`` stores (reads each page from its OWNER chip only) and
loads (admits the hit's pages at their columns, writes each to its owner's local block). Proven here: the per-owner
round trip, bit-exact; a hit after a block remap (a restart with another mapping / another local block); a 256-token
chunk spanning two owners; the T2 page-in records; a STORE of a block without device content is refused; the
replicated path is untouched without the hook."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat

from vllm_tt_plugin.kv_connector import (
    TTPagedCacheHandle,
    TTShardedPagedCacheHandle,
    resolve_connector_kv_views,
    tt_copy_kv_blocks,
)
from vllm_tt_plugin.lmcache import page_contract as pc
from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector
from vllm_tt_plugin.lmcache.sharded_pool import (
    FakeMeshLatentPool,
    OwnerRule,
    PageInRequest,
    ShardedPagePool,
    SimpleOwnerBlockMap,
)

from .tt_lmcache_testing import BLOCK_SIZE, LATENT_DIM, LAYER_NAMES, fingerprint_rows, layer_index, prompt, slot_mapping

C = 8
NB_LOCAL = 12  # physical blocks per chip incl. the null block 0
NUM_BLOCKS = 80  # vLLM's logical pool
UNIT = 128


def _kv_cache_config(num_blocks: int):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MLAAttentionSpec

    spec = MLAAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=LATENT_DIM, dtype=torch.bfloat16)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=num_blocks * BLOCK_SIZE * LATENT_DIM * 2, shared_by=list(LAYER_NAMES))],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=list(LAYER_NAMES), kv_cache_spec=spec)],
    )


class _Mesh:
    """the fake device: one FakeMeshLatentPool (8 chips) + a block map + the model hooks the host resolves"""

    def __init__(self, nb_local=NB_LOCAL, num_blocks=NUM_BLOCKS):
        self.fake = FakeMeshLatentPool(C, nb_local, LAYER_NAMES)
        self.map = SimpleOwnerBlockMap(C, nb_local)
        self.model = SimpleNamespace(
            kv_cache_connector_views=lambda: {n: self.fake.layer_io(n) for n in LAYER_NAMES},
            kv_cache_connector_shard=lambda: dict(C=C, nb_local=nb_local, bpu=UNIT // BLOCK_SIZE,
                                                  num_blocks=num_blocks, map=self.map),
        )
        self.views = resolve_connector_kv_views(object(), _kv_cache_config(num_blocks), self.model)
        self.handles = {n: v.tensor for n, v in self.views.items()}
        pool = None
        for h in self.handles.values():
            inner = getattr(h, "inner", h)
            assert isinstance(inner, TTShardedPagedCacheHandle)
            pool = pool or inner.pool
        self.pool: ShardedPagePool = pool
        self.conn = VLLMPagedMemTTConnector.from_metadata(pc.lmcache_metadata(chunk_size=256))
        self.conn.set_host_xfer_buffer_ops(tt_copy_kv_blocks)
        self.conn.register_device_kv_caches(self.handles)
        self.conn.register_sharded_pool(self.pool)
        self.host = {n: torch.zeros((num_blocks, BLOCK_SIZE, LATENT_DIM), dtype=torch.bfloat16) for n in LAYER_NAMES}

    def prefill(self, blocks: list[int], toks: list[int], keep: int = 0) -> torch.Tensor:
        """the model stand-in: the request's row through the block map (the generator's set_row), then the positions'
        fingerprint rows written on each block's OWNER chip at its local block (the prefill's owner-only writes)"""
        phys = self.map.set_row(blocks, keep=-(-keep // BLOCK_SIZE))
        n = len(toks)
        for name in LAYER_NAMES:
            rows = fingerprint_rows(layer_index(name), list(range(n)), toks)
            for j, g in enumerate(phys):
                owner, local = self.pool.rule.split(g)
                lo, hi = j * BLOCK_SIZE, min((j + 1) * BLOCK_SIZE, n)
                if lo >= n or j * BLOCK_SIZE < keep:  # the hit's blocks hold the loaded pages already
                    continue
                self.fake.chips[name][owner][local, 0, : hi - lo] = rows[lo:hi]
        return slot_mapping(blocks, n)

    def device_tokens(self, name: str, blocks: list[int], n: int) -> torch.Tensor:
        """the token rows as the DECODE would read them: block j of the request from its owner chip's local block"""
        out = []
        for j, b in enumerate(blocks[: -(-n // BLOCK_SIZE)]):
            owner, local = self.pool.rule.split(self.map.physical(b))
            assert owner == self.pool.rule.owner_of_column(j)
            out.append(self.fake.chips[name][owner][local, 0])
        return torch.cat(out)[:n]


def _memobj(num_tokens: int):
    shape = torch.Size([1, pc.NUM_DSA_LAYERS, num_tokens, pc.LMCACHE_HIDDEN_ELEMS])
    return AdHocMemoryAllocator(device="cpu").allocate([shape], [pc.LATENT_DTYPE], fmt=MemoryFormat.KV_MLA_FMT)


def _store(mesh: _Mesh, slots: torch.Tensor, n: int, chunk: int = 256):
    saved = (n - 1) // chunk * chunk if n % chunk else n  # whole chunks (LMCache: the unfull chunk too, when asked)
    saved = max(saved, chunk)
    chunks = [(s, min(s + chunk, saved)) for s in range(0, saved, chunk)]
    objs = [_memobj(e - s) for s, e in chunks]
    mesh.conn.batched_from_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks],
                               kvcaches=mesh.host, slot_mapping=slots)
    return chunks, objs


def test_owner_rule():
    r = OwnerRule(C=8, bpu=2)
    assert [r.owner_of_column(j) for j in range(6)] == [0, 0, 1, 1, 2, 2]
    assert r.split(r.join(5, 7)) == (5, 7) and r.join(5, 7) % 8 == 5 and r.column_of_position(640) == 10
    assert r.unit_of_column(17) == 8


def test_store_reads_each_page_from_its_owner_and_a_chunk_spans_two_owners():
    mesh = _Mesh()
    n = 512
    toks = prompt(n, seed=11)
    blocks = [3, 9, 14, 20, 27, 31, 40, 55]  # 8 blocks = 4 units -> owners 0, 0, 1, 1, 2, 2, 3, 3
    slots = mesh.prefill(blocks, toks)
    mesh.fake.reads.clear()
    chunks, objs = _store(mesh, slots, n)
    assert chunks == [(0, 256), (256, 512)]
    # every page was read from its unit's owner chip at the mapped local block, once per layer, and from NO other chip
    want = {(name, *mesh.pool.rule.split(mesh.map.physical(b))) for name in LAYER_NAMES for b in blocks}
    assert set(mesh.fake.reads) == want and len(mesh.fake.reads) == len(want)
    owners_of_chunk0 = {mesh.pool.rule.owner_of_column(j) for j in range(4)}
    assert owners_of_chunk0 == {0, 1}, "a 256-token chunk = 2 units spans two owner chips"
    # the blob is in token (unit) order per layer and carries no owner: chip-independent
    for (s, e), obj in zip(chunks, objs):
        for li, name in enumerate(LAYER_NAMES):
            got = obj.tensor[0, li, :, :LATENT_DIM]
            want_rows = fingerprint_rows(layer_index(name), list(range(s, e)), toks[s:e])
            assert torch.equal(got.view(torch.int16), want_rows.view(torch.int16)), name
    assert mesh.pool.stats["located"] == len(blocks) * len(LAYER_NAMES)


def test_load_lands_on_the_owner_chips_after_a_block_remap_bit_exact():
    """store on mesh A; load on a COLD mesh B with another block map (another local block per page, as after a restart
    or an eviction + re-admission) and other logical ids; the decode's view of B equals A's rows"""
    a = _Mesh()
    n = 700
    toks = prompt(n, seed=12)
    blocks_a = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    slots_a = a.prefill(blocks_a, toks)
    chunks, objs = _store(a, slots_a, n)
    saved = chunks[-1][1]
    b = _Mesh(nb_local=NB_LOCAL + 4)  # another pool geometry: the same unit lands on another local block
    b.map.admit([(70, 0), (71, 1)])  # a few local blocks taken on chips 0 by another user -> different local ids
    blocks_b = [21, 33, 2, 44, 60, 19, 77, 8, 50, 61, 39]
    slots_b = slot_mapping(blocks_b, n)
    hit = saved
    b.conn.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=b.host, slot_mapping=slots_b)
    # admitted at their columns -> each page on its unit's owner; the record stream is what T2 consumes
    recs = b.pool.pagein_log
    assert len(recs) == hit // BLOCK_SIZE and all(isinstance(r, PageInRequest) for r in recs)
    for k, r in enumerate(recs):
        assert (r.logical_block, r.column, r.unit, r.owner) == (blocks_b[k], k, k // 2, (k // 2) % C)
        assert r.physical_block == b.map.physical(blocks_b[k]) and r.local_block == r.physical_block // C
        assert r.layers == len(LAYER_NAMES) and r.source == "lmcache_hit"
    assert {(w[1], w[2]) for w in b.fake.writes} == {(r.owner, r.local_block) for r in recs}
    # the continuation's row (keep = the hit) finds the admitted blocks and writes only the suffix
    b.prefill(blocks_b, toks, keep=hit)
    for name in LAYER_NAMES:
        got, want = b.device_tokens(name, blocks_b, n), a.device_tokens(name, blocks_a, n)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name
    # the pages of the hit landed at OTHER local blocks than on A (a remap), yet decode reads them at the right place
    la = [a.map.physical(x) // C for x in blocks_a[: hit // BLOCK_SIZE]]
    lb = [b.map.physical(x) // C for x in blocks_b[: hit // BLOCK_SIZE]]
    assert la != lb
    # a second hit on B (the same chunks again, same blocks: the map reuses the physical blocks in place)
    b.fake.writes.clear()
    phys_before = [b.map.physical(x) for x in blocks_b[: hit // BLOCK_SIZE]]
    b.conn.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=b.host, slot_mapping=slots_b)
    assert [b.map.physical(x) for x in blocks_b[: hit // BLOCK_SIZE]] == phys_before
    assert b.pool.stats["moved"] == 0 and len(b.pool.pagein_log) == 2 * (hit // BLOCK_SIZE)


def test_t2_records_are_written_as_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "t2_pagein.jsonl"
    monkeypatch.setenv("GLM53_T2_PAGEIN_LOG", str(path))
    a = _Mesh()
    toks = prompt(256, seed=3)
    slots = a.prefill([4, 5, 6, 7], toks)
    chunks, objs = _store(a, slots, 256)
    b = _Mesh()
    b.conn.to_gpu(objs[0], 0, 256, kvcaches=b.host, slot_mapping=slot_mapping([30, 31, 32, 33], 256))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["logical_block"], r["owner"], r["unit"]) for r in rows] == [(30, 0, 0), (31, 0, 0), (32, 1, 1), (33, 1, 1)]
    assert all(r["local_block"] == r["physical_block"] // C and r["layers"] == 11 for r in rows)


def test_store_of_a_block_without_device_content_is_refused():
    a = _Mesh()
    slots = a.prefill([4, 5, 6, 7], prompt(256, seed=4))
    a.map.forget(6)  # evicted on its owner (or never written): a STORE must not read garbage
    with pytest.raises(ValueError, match="no device content"):
        _store(a, slots, 256)
    # a LOAD without admission is refused too (the generic path must admit with columns first)
    with pytest.raises(ValueError, match="not admitted"):
        tt_copy_kv_blocks(a.host, a.handles, [66], [66], "h2d")


def test_replicated_path_is_unchanged_without_the_hook():
    backing = {n: torch.zeros(6, 1, BLOCK_SIZE, LATENT_DIM, dtype=torch.bfloat16) for n in LAYER_NAMES}
    model = SimpleNamespace(kv_cache_connector_views=lambda: backing)  # no kv_cache_connector_shard
    views = resolve_connector_kv_views(object(), _kv_cache_config(6), model)
    v = views[LAYER_NAMES[0]].tensor
    assert isinstance(v, torch.Tensor) and not isinstance(v, TTPagedCacheHandle) and tuple(v.shape) == (6, BLOCK_SIZE, LATENT_DIM)
    model2 = SimpleNamespace(kv_cache_connector_views=lambda: backing, kv_cache_connector_shard=lambda: None)
    assert isinstance(resolve_connector_kv_views(object(), _kv_cache_config(6), model2)[LAYER_NAMES[0]].tensor, torch.Tensor)


class _FakeShardDma:
    """The model's LatentHitDma contract on the fake mesh (GLM53_KV_T2): load_many / read_pages in ITS layer order;
    ``owners`` = each page's owner chip and the ids are that chip's LOCAL blocks; owners None = every chip / the save
    chip (the replicated assumption, which the sharded pool must never get: it writes local block b everywhere)."""

    def __init__(self, fake: FakeMeshLatentPool):
        self.fake = fake
        self.layer_names = list(reversed(fake.layer_names))  # a different order than the connector's
        self.loads: list[tuple[list[int], list[int] | None]] = []
        self.reads: list[tuple[list[int], list[int] | None]] = []
        self.waits = 0

    def load_many(self, pages, block_ids, timeout_s: float = 30.0, owners=None):
        self.loads.append((list(block_ids), None if owners is None else list(owners)))
        for name, lp in zip(self.layer_names, pages, strict=True):
            rows = torch.from_numpy(lp).view(pc.LATENT_DTYPE).reshape(len(block_ids), *self.fake.shape[1:])
            for k, b in enumerate(block_ids):
                for c in range(self.fake.C) if owners is None else [owners[k]]:
                    self.fake.chips[name][c][b] = rows[k]
        return len(block_ids)

    def read_pages(self, block_ids, timeout_s: float = 10.0, owners=None):
        self.reads.append((list(block_ids), None if owners is None else list(owners)))
        out = []
        for name in self.layer_names:
            rows = torch.stack([self.fake.chips[name][0 if owners is None else owners[k]][b]
                                for k, b in enumerate(block_ids)])  # fmt: skip
            out.append(rows.contiguous().view(torch.int16).numpy().view("u2").reshape(len(block_ids), BLOCK_SIZE, -1))
        return out

    def wait(self, timeout_s: float = 10.0):
        self.waits += 1


def _dma_mesh(**kw) -> tuple[_Mesh, _FakeShardDma]:
    mesh = _Mesh(**kw)
    dma = _FakeShardDma(mesh.fake)
    mesh.conn.set_hit_dma(dma)
    return mesh, dma


def test_t2_store_reads_each_page_from_its_owner_through_the_page_dma(monkeypatch):
    """Stage 12 (n29h7: GLM53_DSA_SHARD + GLM53_KV_T2): the store's page-DMA read gets each page's OWNER chip and LOCAL
    block (before the fix: the logical ids with owners None = chip 0's local block b); the blob equals the handle
    path's bitwise; the row-major host buffers are never written (they stay uncommitted)."""
    monkeypatch.setenv("GLM53_KV_T2_STORE", "1")
    mesh, dma = _dma_mesh()
    n = 512
    toks = prompt(n, seed=21)
    blocks = [3, 9, 14, 20, 27, 31, 40, 55]
    slots = mesh.prefill(blocks, toks)
    mesh.fake.reads.clear()
    chunks, objs = _store(mesh, slots, n)
    assert mesh.fake.reads == [], "the handle (per-page ttnn reads) is not used: one page-DMA read"
    assert len(dma.reads) == 1
    ids, owners = dma.reads[0]
    want = [mesh.pool.rule.split(mesh.map.physical(b)) for b in blocks]
    assert owners is not None and list(zip(owners, ids)) == want
    assert set(owners) == {0, 1, 2, 3}
    for (s, e), obj in zip(chunks, objs):
        for li, name in enumerate(LAYER_NAMES):
            want_rows = fingerprint_rows(layer_index(name), list(range(s, e)), toks[s:e])
            assert torch.equal(obj.tensor[0, li, :, :LATENT_DIM].view(torch.int16), want_rows.view(torch.int16)), name
            assert torch.count_nonzero(obj.tensor[0, li, :, LATENT_DIM:]) == 0, "reserved pooled-key slice zero"
    assert all(torch.count_nonzero(t) == 0 for t in mesh.host.values()), "host buffers never written"
    assert mesh.conn.stats["d2h_dma_direct_pages"] == len(blocks)
    # a page without device content is refused before any DMA read
    mesh.map.forget(9)
    with pytest.raises(ValueError, match="no device content"):
        _store(mesh, slots, n)
    assert len(dma.reads) == 1


def test_t2_load_admits_then_loads_each_page_on_its_owner_through_the_page_dma():
    """the hit's page-DMA load: the pages are admitted at their columns FIRST, then loaded to (owner, local) only --
    never every chip's local block b; bit-exact on a cold mesh with another block map; host buffers untouched"""
    a, _ = _dma_mesh()
    n = 700
    toks = prompt(n, seed=22)
    blocks_a = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    chunks, objs = _store(a, a.prefill(blocks_a, toks), n)
    hit = chunks[-1][1]
    b, dma = _dma_mesh(nb_local=NB_LOCAL + 4)
    b.map.admit([(70, 0), (71, 1)])  # other users' blocks: another local placement than on A
    blocks_b = [21, 33, 2, 44, 60, 19, 77, 8, 50, 61, 39]
    b.conn.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=b.host,
                          slot_mapping=slot_mapping(blocks_b, n))  # fmt: skip
    assert b.fake.writes == [], "the handle's per-page writes are not used"
    assert len(dma.loads) == 1
    ids, owners = dma.loads[0]
    recs = b.pool.pagein_log
    assert len(recs) == hit // BLOCK_SIZE
    assert owners == [r.owner for r in recs] and ids == [r.local_block for r in recs]
    assert owners == [(k // 2) % C for k in range(hit // BLOCK_SIZE)]
    assert all(torch.count_nonzero(t) == 0 for t in b.host.values()), "whole pages: straight from the objects"
    # no chip other than the owner got the page (the replicated assumption wrote local block b on all 8)
    for name in LAYER_NAMES:
        for (o, lo) in zip(owners, ids):
            for c in range(C):
                if c != o:
                    assert torch.count_nonzero(b.fake.chips[name][c][lo]) == 0 or any(
                        oo == c and ll == lo for oo, ll in zip(owners, ids)), (name, c, lo)  # fmt: skip
    b.prefill(blocks_b, toks, keep=hit)
    for name in LAYER_NAMES:
        got, want = b.device_tokens(name, blocks_b, n), a.device_tokens(name, blocks_a, n)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name


def test_t2_partial_page_load_takes_the_host_path_with_owners():
    """a hit ending inside a page (an unfull chunk): the host-buffer path, still through the page-DMA with owners"""
    a, _ = _dma_mesh()
    n = 300
    toks = prompt(n, seed=23)
    blocks_a = [4, 5, 6, 7, 8]
    slots_a = a.prefill(blocks_a, toks)
    obj = _memobj(n)
    a.conn.from_gpu(obj, 0, n, kvcaches=a.host, slot_mapping=slots_a)
    b, dma = _dma_mesh()
    blocks_b = [40, 41, 42, 43, 44]
    b.conn.to_gpu(obj, 0, n, kvcaches=b.host, slot_mapping=slot_mapping(blocks_b, n))
    assert len(dma.loads) == 1
    ids, owners = dma.loads[0]
    assert owners == [b.pool.rule.split(b.map.physical(x))[0] for x in blocks_b]
    assert ids == [b.pool.rule.split(b.map.physical(x))[1] for x in blocks_b]
    for name in LAYER_NAMES:
        got = b.device_tokens(name, blocks_b, n)
        want = fingerprint_rows(layer_index(name), list(range(n)), toks)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name


def test_host_xfer_buffers_are_lazily_committed(monkeypatch):
    """the connector host's transfer buffers (vLLM's logical pool, 11 KiB per pool token): zero, writable, and
    resident only where written (an anonymous mapping), GLM53_KV_HOST_XFER_LAZY=0 = torch.zeros"""
    from vllm_tt_plugin.kv_connector import _host_zeros

    def rss_kb() -> int:
        with open("/proc/self/status") as f:
            return next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))

    before = rss_kb()
    t = _host_zeros((4096, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16)  # 256 MiB
    assert t.shape == (4096, 1, BLOCK_SIZE, LATENT_DIM) and t.dtype == torch.bfloat16
    assert rss_kb() - before < 32 * 1024, "nothing committed at allocation"
    t[7, 0, 3, 5] = 1.5
    assert float(t[7, 0, 3, 5]) == 1.5 and float(t[8].abs().sum()) == 0.0
    assert rss_kb() - before < 32 * 1024, "one page written, one page committed"
    monkeypatch.setenv("GLM53_KV_HOST_XFER_LAZY", "0")
    z = _host_zeros((2, 3), torch.bfloat16)
    assert torch.equal(z, torch.zeros(2, 3, dtype=torch.bfloat16))


def test_owner_batches_group_pages_by_owner_in_order():
    """the bulk store read's plan (n29h7 M1 2026-10-23): pages grouped by owner chip, at most nb per batch, each item
    keeping its position in the request"""
    from vllm_tt_plugin.kv_connector import _owner_batches

    pairs = [(1, 5), (0, 3), (1, 7), (0, 9), (1, 2)]
    assert _owner_batches(pairs, 2) == [(1, [(0, 5), (2, 7)]), (1, [(4, 2)]), (0, [(1, 3), (3, 9)])]
    assert _owner_batches([], 4) == []


def test_sharded_handle_reads_in_bulk_when_the_io_can():
    """TTShardedPagedCacheHandle.read_pages hands every page's (owner, local) to io.read_locals in one call (the device
    io groups them per owner chip); one page, or an io without it, reads page by page as before"""
    import torch

    from vllm_tt_plugin.kv_connector import TTShardedPagedCacheHandle

    page = (1, 2, 4)

    class _IO:
        def __init__(self):
            self.bulk, self.single = [], []

        def read_local(self, owner, local):
            self.single.append((owner, local))
            return torch.full((1, *page), float(owner * 100 + local))

        def read_locals(self, pairs):
            self.bulk.append(list(pairs))
            return torch.cat([torch.full((1, *page), float(o * 100 + lo)) for o, lo in pairs])

    from types import SimpleNamespace

    pool = SimpleNamespace(locate=lambda b: (b % 2, b // 2))
    io = _IO()
    h = TTShardedPagedCacheHandle(torch.zeros(10, *page), pool, io, num_blocks=20, dtype=torch.float32)
    got = h.read_pages([7, 2, 9])
    assert io.bulk == [[(1, 3), (0, 1), (1, 4)]] and io.single == []
    assert [float(x[0, 0, 0]) for x in got] == [103.0, 1.0, 104.0]
    h.read_pages([6])
    assert io.single == [(0, 3)]
    del _IO.read_locals
    io2 = _IO()
    TTShardedPagedCacheHandle(torch.zeros(10, *page), pool, io2, num_blocks=20, dtype=torch.float32).read_pages([7, 2])
    assert io2.single == [(1, 3), (0, 1)]
