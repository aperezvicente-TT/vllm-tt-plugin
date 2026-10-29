"""Stage 12 lever (c) step 2 (program 2026-10-26T03:59:21Z): GLM53_KV_TILE_OBJECTS=1 -- memory objects in the
device's tile order (whole pages: 64 KB latent tiles + the 4 KB pooled half; a partial last page row-major), stored
from the page-DMA's tile-order read (no untilize) and loaded by descriptors at the objects' own bytes when they live
in the model's pinned pool (LatentHitDma.submit_resident: here a fake that checks the tile order independently),
else untilized onto the row-major submit; the host paths read them through a row-major view. Host only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat

from vllm_tt_plugin.lmcache import connector as C
from vllm_tt_plugin.lmcache import page_contract as pc
from vllm_tt_plugin.lmcache.mock_paged_cache import MockTTPagedLatentCache

from .test_tt_connector_pages import NUM_BLOCKS, _connector, _memobj, _write_fingerprints
from .tt_lmcache_testing import BLOCK_SIZE, LAYER_NAMES, fingerprint_rows, layer_index, pick_blocks, prompt, slot_mapping

T, LB = BLOCK_SIZE, 1024
PAGE = T * (LB + 64)  # 69,632


def _np_tilize(rows: np.ndarray) -> np.ndarray:
    """the device's tile order, written from its definition: element (tr, tc, fr, fc, r, c) of a [64, 512] page =
    row tr 32 + fr 16 + r, column tc 32 + fc 16 + c -> [65,536] bytes"""
    out = np.empty((2, 16, 2, 2, 16, 16), dtype=rows.dtype)
    for tr in range(2):
        for tc in range(16):
            for fr in range(2):
                for fc in range(2):
                    out[tr, tc, fr, fc] = rows[tr * 32 + fr * 16 : tr * 32 + fr * 16 + 16, tc * 32 + fc * 16 : tc * 32 + fc * 16 + 16]
    return out.reshape(-1).view(np.uint8)


class _Tiled:
    """page_pool.TiledPages' contract: .tiles [n, 32 x 1,024] uint16 tile order; result() -> row-major [n, 32,768] int16"""

    def __init__(self, tiles):
        self.tiles = tiles
        self.nbytes = int(tiles.nbytes)

    def result(self):
        n = self.tiles.shape[0]
        rows = self.tiles.reshape(n, 2, 16, 2, 2, 16, 16).transpose(0, 1, 3, 5, 2, 4, 6).reshape(n, -1)
        return torch.from_numpy(np.ascontiguousarray(rows).view(np.int16))


class _FakeTileDma:
    """the model's LatentHitDma contract with tile objects: read_pages(untilize=False) -> tile-order pages, holds(t) =
    t inside the fake pinned pool, submit_resident -> untilizes the objects' pages INDEPENDENTLY (numpy definition)
    into the mock device; submit / submit_objects as the row-major paths"""

    slots = 1 << 20

    def __init__(self, cache: MockTTPagedLatentCache, pool: torch.Tensor | None = None):
        self.cache, self.pool = cache, pool
        self.layer_names = list(reversed(cache.layer_names))
        self.resident, self.objects, self.rowmajor, self.waits, self.reads, self.after = [], [], [], 0, [], []
        self.saved = []

    def read_pages(self, block_ids, untilize: bool = True):
        self.reads.append((list(block_ids), untilize))
        out = []
        for name in self.layer_names:
            dev = self.cache.device_caches[name]
            rows = dev.view(dev.shape[0], -1)[torch.as_tensor(block_ids)].contiguous().view(torch.int16).numpy()
            rows = rows.view(np.uint16).reshape(len(block_ids), 64, 512)
            out.append(rows if untilize else _Tiled(np.stack([_np_tilize(r).view(np.uint16) for r in rows])))
        return out

    def holds(self, t) -> bool:
        if self.pool is None:
            return False
        lo = t.data_ptr() - self.pool.data_ptr()
        hi = lo + sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
        return 0 <= lo and hi <= self.pool.numel()

    def _write(self, name, block_ids, rows):  # rows [n, 64, 512] uint16
        dev = self.cache.device_caches[name]
        dev.view(dev.shape[0], -1)[torch.as_tensor(block_ids)] = torch.from_numpy(rows.view(np.int16)).view(
            pc.LATENT_DTYPE).reshape(len(block_ids), -1)  # fmt: skip

    def submit_resident(self, objects, block_ids, layer_order=None):
        self.resident.append(list(block_ids))
        inv = {}
        for j in range(len(self.layer_names)):
            i = layer_order[j] if layer_order is not None else j
            pages = []
            for r, m in objects:
                assert self.holds(r[i, : m * T])
                lay = r[i, : m * T].reshape(m, PAGE)[:, : T * LB].numpy()
                for p in range(m):  # untilize by the definition: inverse of _np_tilize
                    t = lay[p].view(np.uint16).reshape(2, 16, 2, 2, 16, 16)
                    pages.append(t.transpose(0, 2, 4, 1, 3, 5).reshape(64, 512))
            inv[self.layer_names[j]] = np.stack(pages)
        for name, rows in inv.items():
            self._write(name, block_ids, rows)

    def save_resident(self, objects, block_ids, layer_order=None):
        """the zero-copy store: each page's device rows tilized (the definition) straight into the objects' tile slots"""
        self.saved.append(list(block_ids))
        k = 0
        for r, m in objects:
            assert self.holds(r[:, : m * T])
            for j, name in enumerate(self.layer_names):
                i = layer_order[j] if layer_order is not None else j
                dev = self.cache.device_caches[name]
                lay = r[i, : m * T].reshape(m, PAGE)
                for p in range(m):
                    rows = dev.view(dev.shape[0], -1)[block_ids[k + p]].contiguous().view(torch.int16).numpy()
                    lay[p, : T * LB] = torch.from_numpy(_np_tilize(rows.view(np.uint16).reshape(64, 512)).copy())
            k += m

    def submit_objects(self, objects, block_ids, layer_order=None):
        self.objects.append(list(block_ids))
        for j in range(len(self.layer_names)):
            i = layer_order[j] if layer_order is not None else j
            rows = torch.cat([r[i, : m * T, :LB] for r, m in objects]).contiguous()
            self._write(self.layer_names[j], block_ids, rows.view(torch.int16).numpy().view(np.uint16).reshape(-1, 64, 512))

    def submit(self, pages, block_ids):
        self.rowmajor.append(list(block_ids))
        for name, layer_pages in zip(self.layer_names, pages, strict=True):
            self._write(name, block_ids, np.asarray(layer_pages).view(np.uint16).reshape(len(block_ids), 64, 512))

    def wait(self):
        self.waits += 1
        while self.after:
            self.after.pop(0)()

    def after_wait(self, fn):
        self.after.append(fn)


def _pool_objs(pool: torch.Tensor, chunks):
    alloc = TensorMemoryAllocator(pool, align_bytes=4096)
    out = []
    for s, e in chunks:
        o = alloc.allocate([torch.Size([1, pc.NUM_DSA_LAYERS, e - s, pc.LMCACHE_HIDDEN_ELEMS])], [pc.LATENT_DTYPE],
                           fmt=MemoryFormat.KV_MLA_FMT)  # fmt: skip
        assert o is not None
        out.append(o)
    return out


def _stored(prompt_len, saved, objs_fn, monkeypatch, pooled=None):
    """request A's fingerprints stored (tile mode, the page-DMA read) into objs_fn(chunks) -> (objs, chunks, toks)"""
    monkeypatch.setenv("GLM53_KV_TILE_OBJECTS", "1")
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    slots_a = slot_mapping(pick_blocks(prompt_len, NUM_BLOCKS, seed=1), prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    chunks = [(s, min(s + 256, saved)) for s in range(0, saved, 256)]
    objs = objs_fn(chunks)
    conn = _connector(a)
    dma = _FakeTileDma(a)
    conn.set_hit_dma(dma)
    if pooled is not None:
        conn.enable_pooled_pages(lambda ids: [pooled[n][torch.as_tensor(ids)] for n in dma.layer_names])
    conn.batched_from_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=a.host_list, slot_mapping=slots_a)
    assert dma.reads and all(u is False for _b, u in dma.reads)  # the tile-order read, never untilized
    return objs, chunks, toks, a, slots_a


def _load(objs, chunks, prompt_len, saved, pool=None):
    b = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    blocks_b = pick_blocks(prompt_len, NUM_BLOCKS, seed=2)
    slots_b = slot_mapping(blocks_b, prompt_len)
    conn = _connector(b)
    dma = _FakeTileDma(b, pool)
    conn.set_hit_dma(dma)
    conn.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=b.host_list, slot_mapping=slots_b)
    return b, slots_b, conn, dma, blocks_b


def _check(b, slots_b, toks, n):
    for name in LAYER_NAMES:
        got = b.device_tokens(name, slots_b[:n])
        want = fingerprint_rows(layer_index(name), list(range(n)), toks[:n])
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name


def test_tile_helpers_are_the_device_tile_order():
    rows = torch.randint(0, 256, (3, 64, 1024), dtype=torch.uint8)
    t = C._tilize_page_bytes(rows)
    for k in range(3):
        assert np.array_equal(t[k].numpy(), _np_tilize(rows[k].view(torch.int16).numpy().view(np.uint16).reshape(64, 512)))
    assert torch.equal(C._untilize_page_bytes(t), rows)


def test_tile_store_then_resident_load_is_bit_exact(monkeypatch):
    prompt_len, saved = 600, 576
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    objs, chunks, toks, _a, _s = _stored(prompt_len, saved, lambda ch: _pool_objs(pool, ch), monkeypatch)
    # the stored bytes ARE the tile order: object 0, layer 0, page 0 = the tilized fingerprint page
    lay0 = objs[0].tensor.view(torch.uint8).reshape(pc.NUM_DSA_LAYERS, -1)[0, :PAGE]
    want = fingerprint_rows(layer_index(LAYER_NAMES[0]), list(range(T)), toks[:T]).contiguous()
    assert np.array_equal(lay0[: T * LB].numpy(), _np_tilize(want.view(torch.int16).numpy().view(np.uint16)))
    refs = [o.get_ref_count() for o in objs]
    b, slots_b, conn, dma, blocks_b = _load(objs, chunks, prompt_len, saved, pool)
    assert dma.resident == [blocks_b[: saved // T]] and not dma.objects and not dma.rowmajor
    assert conn.stats.get("h2d_resident_submits") == 1 and conn.stats.get("h2d_resident_async") == 1
    # async: no wait in the connector; the objects are held until the DMA's next wait(), then released
    assert dma.waits == 0 and [o.get_ref_count() for o in objs] == [r + 1 for r in refs]
    dma.wait()
    assert [o.get_ref_count() for o in objs] == refs
    _check(b, slots_b, toks, saved)


def test_tile_pooled_halves_are_views_held_until_released(monkeypatch):
    """GLM53_KV_POOLED_VIEWS (default on in tile mode): a loaded page's pooled entry is a VIEW of its object's 4 KB
    pooled half (no copy on the engine thread), the object refcount-held until release_loaded_pooled; the taken
    slices are the stored pooled keys and survive the release (copies); GLM53_KV_POOLED_VIEWS=0 = the copied keep"""
    prompt_len, saved = 600, 576
    gen = torch.Generator().manual_seed(5)
    pooled = {n: torch.randint(0, 256, (NUM_BLOCKS, 4096), dtype=torch.uint8, generator=gen) for n in LAYER_NAMES}
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    objs, chunks, _toks, _a, slots_a = _stored(prompt_len, saved, lambda ch: _pool_objs(pool, ch), monkeypatch, pooled)
    blocks_a = [int(x) // T for x in slots_a[:saved:T].tolist()]
    refs = [o.get_ref_count() for o in objs]
    for views in ("1", "0"):
        monkeypatch.setenv("GLM53_KV_POOLED_VIEWS", views)
        b = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
        blocks_b = pick_blocks(prompt_len, NUM_BLOCKS, seed=2)
        conn = _connector(b)
        dma = _FakeTileDma(b, pool)
        conn.set_hit_dma(dma)
        conn.enable_pooled_pages(lambda ids: [pooled[n][torch.as_tensor(ids)] for n in dma.layer_names])
        conn.batched_to_gpu(objs, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=b.host_list,
                            slot_mapping=slot_mapping(blocks_b, prompt_len))  # fmt: skip
        dma.wait()  # the async DMA hold released: what is left is the views'
        live = [o for o in objs if o.tensor is not None]
        lo, hi = pool.data_ptr(), pool.data_ptr() + pool.numel()
        in_pool = [lo <= conn.loaded_pooled[p].data_ptr() < hi for p in blocks_b[: saved // T]]
        assert all(in_pool) if views == "1" else not any(in_pool)
        held = [o.get_ref_count() - r for o, r in zip(live, refs)]
        assert held == ([1] * len(live) if views == "1" else [0] * len(live))
        assert conn.stats.get("pooled_view_pages", 0) == (saved // T if views == "1" else 0)
        got = conn.take_loaded_pooled(blocks_b[: saved // T])
        conn.release_loaded_pooled()
        assert [o.get_ref_count() for o in live] == refs and not conn.loaded_pooled
        pool_before = pool.clone()
        pool.zero_()  # the objects' bytes gone: the taken slices are copies
        for j, n in enumerate(dma.layer_names):
            assert torch.equal(got[j], pooled[n][torch.as_tensor(blocks_a)]), (views, n)
        pool.copy_(pool_before)


def test_tile_resident_load_waits_without_the_after_wait_hook(monkeypatch):
    prompt_len, saved = 600, 576
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    objs, chunks, toks, _a, _s = _stored(prompt_len, saved, lambda ch: _pool_objs(pool, ch), monkeypatch)
    monkeypatch.setenv("GLM53_KV_RESIDENT_ASYNC", "0")
    b, slots_b, conn, dma, _blocks_b = _load(objs, chunks, prompt_len, saved, pool)
    assert dma.waits == 1 and not conn.stats.get("h2d_resident_async")
    _check(b, slots_b, toks, saved)


def test_tile_objects_outside_the_pool_load_untilized(monkeypatch):
    prompt_len, saved = 600, 576
    objs, chunks, toks, _a, _s = _stored(prompt_len, saved, lambda ch: [_memobj(e - s) for s, e in ch], monkeypatch)
    b, slots_b, conn, dma, blocks_b = _load(objs, chunks, prompt_len, saved, None)
    assert not dma.resident and (dma.objects or dma.rowmajor)
    assert conn.stats.get("h2d_tile_untilized_pages") == saved // T
    _check(b, slots_b, toks, saved)


def test_tile_store_worker_job_equals_the_direct_store(monkeypatch):
    prompt_len, saved = 700, 640
    rng = torch.Generator().manual_seed(4)
    pooled = {n: torch.randint(0, 256, (NUM_BLOCKS, 4096), generator=rng, dtype=torch.uint8) for n in LAYER_NAMES}
    direct, chunks, _toks, a, slots_a = _stored(prompt_len, saved, lambda ch: [_memobj(e - s) for s, e in ch],
                                                monkeypatch, pooled)  # fmt: skip
    conn = _connector(a)
    conn.initialize_kvcaches_ptr(kvcaches=a.host_list)
    dma = _FakeTileDma(a)
    conn.set_hit_dma(dma)
    conn.enable_pooled_pages(lambda ids: [pooled[n][torch.as_tensor(ids)] for n in dma.layer_names])
    job = conn.prefetch_store_pages(sorted({int(x) // T for x in slots_a[:saved].tolist()}))
    assert job["tile"]
    conn.complete_store_job(job)
    worker = [_memobj(e - s) for s, e in chunks]
    conn.set_store_job(job)
    try:
        conn.batched_from_gpu(worker, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=a.host_list,
                              slot_mapping=slots_a)  # fmt: skip
    finally:
        conn.set_store_job(None)
    for x, y in zip(direct, worker, strict=True):
        assert torch.equal(x.tensor.view(torch.uint8), y.tensor.view(torch.uint8))
    # the loaded pooled halves are the stored pages' pooled keys
    b, slots_b, conn_b, dma_b, blocks_b = _load(worker, chunks, prompt_len, saved, None)
    conn_b.enable_pooled_pages(lambda ids: [pooled[n][torch.as_tensor(ids)] for n in dma_b.layer_names])
    conn_b.batched_to_gpu(worker, [s for s, _ in chunks], [e for _, e in chunks], kvcaches=b.host_list,
                          slot_mapping=slots_b)  # fmt: skip
    blocks_a = [int(x) // T for x in slots_a[:saved:T].tolist()]
    got = conn_b.take_loaded_pooled(blocks_b[: saved // T])
    assert got is not None
    for j, n in enumerate(dma_b.layer_names):  # take_loaded_pooled: the pooled source's (= the hit-DMA's) layer order
        assert torch.equal(got[j], pooled[n][torch.as_tensor(blocks_a)]), n


def test_partial_last_page_roundtrips_row_major(monkeypatch):
    prompt_len, saved = 600, 600  # 9 whole pages + a 24-token page
    objs, chunks, toks, _a, _s = _stored(prompt_len, saved, lambda ch: [_memobj(e - s) for s, e in ch], monkeypatch)
    b, slots_b, conn, dma, _blocks_b = _load(objs, chunks, prompt_len, saved, None)
    assert not dma.resident  # a partial page: the host path (row-major view of the tile objects)
    _check(b, slots_b, toks, saved)


def test_tile_mode_refuses_a_store_without_the_tile_read(monkeypatch):
    monkeypatch.setenv("GLM53_KV_TILE_OBJECTS", "1")
    prompt_len, saved = 300, 256
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    slots_a = slot_mapping(pick_blocks(prompt_len, NUM_BLOCKS, seed=1), prompt_len)
    conn = _connector(a)
    with pytest.raises(RuntimeError, match="page-DMA"):  # no page-DMA at all: the host copy path is refused
        conn.batched_from_gpu([_memobj(saved)], [0], [saved], kvcaches=a.host_list, slot_mapping=slots_a)


@pytest.mark.parametrize("with_pooled", [True, False])
def test_zero_copy_store_equals_the_read_and_fill_store(monkeypatch, with_pooled):
    """GLM53_KV_STORE_ZEROCOPY: objects in the pinned pool are SAVEd into by the page-DMA (no read_pages, no fill) and
    hold exactly the bytes of the read + fill store; the pooled halves come from the pooled source"""
    prompt_len, saved = 600, 512  # whole pages only (8 pages in 2 objects)
    gen = torch.Generator().manual_seed(9)
    pooled = ({n: torch.randint(0, 256, (NUM_BLOCKS, 4096), dtype=torch.uint8, generator=gen) for n in LAYER_NAMES}
              if with_pooled else None)  # fmt: skip
    ref, _c, _t, _a, _s = _stored(prompt_len, saved, lambda ch: [_memobj(e - s) for s, e in ch], monkeypatch, pooled)
    monkeypatch.setenv("GLM53_KV_STORE_ZEROCOPY", "1")
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    slots_a = slot_mapping(pick_blocks(prompt_len, NUM_BLOCKS, seed=1), prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    chunks = [(s0, min(s0 + 256, saved)) for s0 in range(0, saved, 256)]
    objs = _pool_objs(pool, chunks)
    conn = _connector(a)
    dma = _FakeTileDma(a, pool)
    conn.set_hit_dma(dma)
    if pooled is not None:
        conn.enable_pooled_pages(lambda ids: [pooled[n][torch.as_tensor(ids)] for n in dma.layer_names])
    assert conn.zero_copy_store()
    conn.batched_from_gpu(objs, [s0 for s0, _ in chunks], [e for _, e in chunks], kvcaches=a.host_list, slot_mapping=slots_a)
    assert dma.saved and not dma.reads and conn.stats.get("d2h_resident_saves") == 1
    for x, y in zip(objs, ref, strict=True):
        assert torch.equal(x.tensor.view(torch.uint8).reshape(-1), y.tensor.view(torch.uint8).reshape(-1))
    # and the objects load back bitwise (the resident page-in)
    b, slots_b, _conn, _dma, _blocks_b = _load(objs, chunks, prompt_len, saved, pool)
    _check(b, slots_b, toks, saved)


def test_zero_copy_store_falls_back_for_partial_pages_and_outside_objects(monkeypatch):
    monkeypatch.setenv("GLM53_KV_TILE_OBJECTS", "1")
    monkeypatch.setenv("GLM53_KV_STORE_ZEROCOPY", "1")
    prompt_len = 600
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    slots_a = slot_mapping(pick_blocks(prompt_len, NUM_BLOCKS, seed=1), prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    for saved, objs_fn in ((600, lambda ch: _pool_objs(pool, ch)),  # a partial last page
                           (512, lambda ch: [_memobj(e - s0) for s0, e in ch])):  # objects outside the pool
        chunks = [(s0, min(s0 + 256, saved)) for s0 in range(0, saved, 256)]
        conn = _connector(a)
        dma = _FakeTileDma(a, pool)
        conn.set_hit_dma(dma)
        conn.batched_from_gpu(objs_fn(chunks), [s0 for s0, _ in chunks], [e for _, e in chunks], kvcaches=a.host_list,
                              slot_mapping=slots_a)  # fmt: skip
        assert not dma.saved and dma.reads, saved  # the read + fill path


def test_the_host_runs_zero_copy_stores_on_the_flushing_thread(monkeypatch):
    """kv_connector._save_pages: with zero-copy stores the store worker is bypassed (the pages are valid at the flush)"""
    from types import SimpleNamespace

    from vllm_tt_plugin import kv_connector as K

    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    submitted, synced = [], []
    host.store_worker = SimpleNamespace(submit=lambda meta, job: submitted.append(job))
    host._generic_sync = False
    host._metadata = lambda so: SimpleNamespace(requests=[])
    host._store_page_ids = lambda meta: [1, 2]
    host.tt_gpu_connector = SimpleNamespace(zero_copy_store=lambda: True, kvcaches=object(),
                                            prefetch_store_pages=lambda pages: {"pages": pages})  # fmt: skip
    host._step_context = lambda mi: (None, {})
    host.views = {}
    host._lock = __import__("threading").Lock()
    host.connector = SimpleNamespace(bind_connector_metadata=lambda m: synced.append("bind"),
                                     wait_for_save=lambda: synced.append("wait"),
                                     clear_connector_metadata=lambda: synced.append("clear"))  # fmt: skip
    host._save_pages(SimpleNamespace(), SimpleNamespace())
    assert submitted == [] and synced == ["bind", "wait", "clear"]
    host.tt_gpu_connector.zero_copy_store = lambda: False
    host._save_pages(SimpleNamespace(), SimpleNamespace())
    assert submitted == [{"pages": [1, 2]}]


def test_zero_copy_store_on_the_sharded_pool_passes_owner_local_blocks(monkeypatch):
    """GLM53_DSA_SHARD: the zero-copy store's descriptors address each page's LOCAL block on its OWNER chip (the
    sharded pool's locate), exactly as the read path does -- save_resident receives owners= and local ids"""
    monkeypatch.setenv("GLM53_KV_TILE_OBJECTS", "1")
    monkeypatch.setenv("GLM53_KV_STORE_ZEROCOPY", "1")
    prompt_len, saved = 600, 512
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    slots_a = slot_mapping(pick_blocks(prompt_len, NUM_BLOCKS, seed=1), prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    chunks = [(s0, min(s0 + 256, saved)) for s0 in range(0, saved, 256)]
    objs = _pool_objs(pool, chunks)
    conn = _connector(a)
    dma = _FakeTileDma(a, pool)
    got = {}
    base_save = dma.save_resident

    def save_resident(objects, block_ids, layer_order=None, owners=None):
        got["ids"], got["owners"] = list(block_ids), list(owners)
        # the fake device is replicated: translate (owner, local) back to the logical page to write its content
        base_save(objects, [2 * lo + ow for ow, lo in zip(owners, block_ids)], layer_order=layer_order)

    dma.save_resident = save_resident
    conn.set_hit_dma(dma)
    conn.sharded_pool = SimpleNamespace(locate=lambda p: (int(p) % 2, int(p) // 2))
    conn.batched_from_gpu(objs, [s0 for s0, _ in chunks], [e for _, e in chunks], kvcaches=a.host_list, slot_mapping=slots_a)
    pages = [int(x) // T for x in slots_a[:saved:T].tolist()]
    assert got["owners"] == [p % 2 for p in pages] and got["ids"] == [p // 2 for p in pages]
    ref, _c, _t, _a, _s = _stored(prompt_len, saved, lambda ch: [_memobj(e - s0) for s0, e in ch], monkeypatch)
    for x, y in zip(objs, ref, strict=True):  # the same bytes as the replicated read + fill store
        lat = lambda o: o.tensor.view(torch.uint8).reshape(pc.NUM_DSA_LAYERS, -1, PAGE)[..., : T * LB]  # noqa: E731
        assert torch.equal(lat(x), lat(y))


def test_a_hit_ending_on_a_partial_chunk_keeps_its_whole_objects_on_the_resident_dma(monkeypatch):
    """14's SHOULD 3: pool objects [0, 256), [256, 512) and a partial tail [512, 600) (one whole page + 24 tokens): the
    two whole objects load by descriptors at their own bytes, only the tail takes the host path; bitwise either way"""
    prompt_len, saved = 600, 600
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    objs, chunks, toks, _a, _s = _stored(prompt_len, saved, lambda ch: _pool_objs(pool, ch), monkeypatch)
    b, slots_b, conn, dma, blocks_b = _load(objs, chunks, prompt_len, saved, pool)
    assert dma.resident == [blocks_b[:8]] and dma.rowmajor == [blocks_b[8:10]]
    assert conn.stats["h2d_split_partial"] == 1
    _check(b, slots_b, toks, saved)
    monkeypatch.setenv("GLM53_KV_SPLIT_PARTIAL", "0")  # the old path: every object through the host buffers
    b, slots_b, conn, dma, blocks_b = _load(objs, chunks, prompt_len, saved, pool)
    assert not dma.resident and dma.rowmajor == [blocks_b[:10]]
    _check(b, slots_b, toks, saved)


class _FakeOneDma(_FakeTileDma):
    """+ store path P2's save_resident_pooled: the latent SAVEs and the pooled reads in ONE wait"""

    def __init__(self, cache, pool, pooled):
        super().__init__(cache, pool)
        self.pooled_dev, self.one = pooled, []

    def save_resident_pooled(self, objects, block_ids, targets, layer_order=None):
        self.save_resident(objects, block_ids, layer_order=layer_order)
        self.one.append(list(targets))
        self.wait()
        return [("raw", self.pooled_dev[n][torch.as_tensor(targets)]) for n in self.layer_names]


@pytest.mark.parametrize("one_dma", ["0", "1"])
def test_one_dma_store_equals_the_two_wait_store(monkeypatch, one_dma):
    """store path P2 (GLM53_KV_STORE_ONE_DMA=1): with the model's pooled plan the zero-copy store reads its pooled
    pages in the same page-DMA wait as its latent pages; the objects hold exactly the two-wait store's bytes. Off (or
    without the plan / the DMA method): the two-wait path, unchanged"""
    prompt_len, saved = 600, 512
    gen = torch.Generator().manual_seed(9)
    pooled = {n: torch.randint(0, 256, (NUM_BLOCKS, 4096), dtype=torch.uint8, generator=gen) for n in LAYER_NAMES}
    ref, _c, _t, _a, _s = _stored(prompt_len, saved, lambda ch: [_memobj(e - s) for s, e in ch], monkeypatch, pooled)
    monkeypatch.setenv("GLM53_KV_STORE_ZEROCOPY", "1")
    monkeypatch.setenv("GLM53_KV_STORE_ONE_DMA", one_dma)
    pool = torch.zeros(64 << 20, dtype=torch.uint8)
    toks, positions = prompt(prompt_len), list(range(prompt_len))
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch")
    slots_a = slot_mapping(pick_blocks(prompt_len, NUM_BLOCKS, seed=1), prompt_len)
    _write_fingerprints(a, slots_a, positions, toks)
    chunks = [(s0, min(s0 + 256, saved)) for s0 in range(0, saved, 256)]
    objs = _pool_objs(pool, chunks)
    conn = _connector(a)
    dma = _FakeOneDma(a, pool, pooled)
    conn.set_hit_dma(dma)
    source_calls = []

    def source(ids):
        source_calls.append(list(ids))
        return [pooled[n][torch.as_tensor(ids)] for n in dma.layer_names]

    def plan(ids):  # the model's split: targets = the pages, finish = unwrap the device arrays
        return list(ids), lambda got: [g[1] for g in got]

    conn.enable_pooled_pages(source, plan=plan)
    conn.batched_from_gpu(objs, [s0 for s0, _ in chunks], [e for _, e in chunks], kvcaches=a.host_list, slot_mapping=slots_a)
    for x, y in zip(objs, ref, strict=True):
        assert torch.equal(x.tensor.view(torch.uint8).reshape(-1), y.tensor.view(torch.uint8).reshape(-1))
    if one_dma == "1":
        assert dma.one and not source_calls and dma.waits == 1 and conn.stats["d2h_one_dma_saves"] == 1
    else:
        assert not dma.one and source_calls and dma.waits == 1 and "d2h_one_dma_saves" not in conn.stats
