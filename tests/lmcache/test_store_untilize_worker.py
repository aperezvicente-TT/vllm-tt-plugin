"""Stage 12 W4 F3a (2026-10-24, doc/ttnn/12-fast-start/w4_host): GLM53_KV_STORE_UNTILIZE_WORKER=1 asks the model's
page-DMA read for the ring's tile-order pages (read_pages(untilize=False): objects whose result() untilizes) so the
store worker, not the engine thread, runs the host untilize (56 % of the engine thread's store read on cf3h-d)."""

import threading

import torch

from vllm_tt_plugin.lmcache.mock_paged_cache import MockTTPagedLatentCache

from .test_tt_connector_pages import NUM_BLOCKS, _connector, _FakeReadDma, _write_fingerprints
from .tt_lmcache_testing import pick_blocks, prompt, slot_mapping


class _FakeTiledReadDma(_FakeReadDma):
    """read_pages(untilize=False): each layer as an object whose result() gives the rows (here: the row-major bytes the
    untilized read returns, as torch int16 [n, page elements]) on the calling thread"""

    def __init__(self, cache, log):
        super().__init__(cache)
        self.log, self.kwargs = log, []

    def read_pages(self, block_ids, untilize=True):
        self.kwargs.append(untilize)
        rows = super().read_pages(block_ids)
        if untilize:
            return rows
        log = self.log

        class Tiled:
            nbytes = int(rows[0].nbytes)

            def __init__(self, r):
                self.r = r

            def result(self):
                log.append(threading.current_thread().name)
                return torch.from_numpy(self.r.view("i2").reshape(self.r.shape[0], -1).copy())

        return [Tiled(r) for r in rows]


def _cache():
    a = MockTTPagedLatentCache(NUM_BLOCKS, backend="torch", seed=9)
    n = 400
    blocks = pick_blocks(n, NUM_BLOCKS, seed=4)
    _write_fingerprints(a, slot_mapping(blocks, n), list(range(n)), prompt(n))
    return a, blocks[: n // 64]


def test_untilize_worker_defers_the_untilize_to_the_store_worker(monkeypatch):
    a, ids = _cache()
    log: list = []
    conn = _connector(a)
    conn.initialize_kvcaches_ptr(kvcaches=a.host_list)
    dma = _FakeTiledReadDma(a, log)
    conn.set_hit_dma(dma)
    monkeypatch.setenv("GLM53_KV_STORE_UNTILIZE_WORKER", "0")
    want = conn.prefetch_store_pages(ids)
    assert dma.kwargs == [True] and all(isinstance(t, torch.Tensor) for t in want["bytes"])
    monkeypatch.setenv("GLM53_KV_STORE_UNTILIZE_WORKER", "1")
    job = conn.prefetch_store_pages(ids)
    assert dma.kwargs == [True, False] and not any(isinstance(t, torch.Tensor) for t in job["bytes"])
    assert job["nbytes"] == want["nbytes"] and log == []  # nothing untilized on the engine thread
    th = threading.Thread(target=conn.complete_store_job, args=(job,), name="tt-kv-store-worker")
    th.start()
    th.join()
    assert log == ["tt-kv-store-worker"] * len(job["bytes"])
    for x, y in zip(job["bytes"], want["bytes"], strict=True):
        assert torch.equal(x, y)


def test_untilize_worker_falls_back_on_a_model_without_the_tiled_read(monkeypatch):
    """a model tree whose read_pages has no untilize parameter: the untilized read, as before (no TypeError)"""
    a, ids = _cache()
    conn = _connector(a)
    conn.initialize_kvcaches_ptr(kvcaches=a.host_list)
    conn.set_hit_dma(_FakeReadDma(a))
    monkeypatch.setenv("GLM53_KV_STORE_UNTILIZE_WORKER", "1")
    job = conn.prefetch_store_pages(ids)
    assert all(isinstance(t, torch.Tensor) for t in job["bytes"]) and job["nbytes"] > 0
