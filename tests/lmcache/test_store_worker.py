# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GLM53_KV_STORE_WORKER (Stage 12 cf3h, program 2026-10-23T21:13:14Z): LMCache's page stores off the engine
thread. The engine thread reads the stored pages (device -> host) into a job snapshot and hands (metadata, snapshot)
to one store worker; the worker runs LMCache's store loop with that metadata (thread-local, the engine thread keeps
binding its own) and the GPU connector copies from the snapshot. No device, no vLLM engine."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch

import vllm_tt_plugin.kv_connector as K


class _Impl:
    def __init__(self):
        self._parent = None
        self._lmcache_chunk_size = 128
        self.calls: list = []
        self.gate = threading.Event()
        self.gate.set()

    def wait_for_save(self):
        self.gate.wait(10)
        self.calls.append((threading.current_thread().name, self._parent._get_connector_metadata()))


class _Conn:
    def __init__(self):
        self._lmcache_engine = _Impl()
        self._lmcache_engine._parent = self
        self.bound = None
        self.shut = False

    def _get_connector_metadata(self):
        return self.bound

    def bind_connector_metadata(self, m):
        self.bound = m

    def clear_connector_metadata(self):
        self.bound = None

    def save_kv_layer(self, *a):
        pass

    def wait_for_save(self):
        self._lmcache_engine.wait_for_save()

    def shutdown(self):
        self.shut = True


class _GPU:
    def __init__(self):
        self.prefetch: list = []
        self._tl = threading.local()
        self.seen_jobs: list = []

    def prefetch_store_pages(self, pages):
        self.prefetch.append((threading.current_thread().name, list(pages)))
        return {"row": {p: i for i, p in enumerate(pages)}, "bytes": [], "pooled": None, "nbytes": 10 * len(pages)}

    def set_store_job(self, job):
        self._tl.job = job

    def current_store_job(self):
        return getattr(self._tl, "job", None)


def _host(monkeypatch, cap_gb="8"):
    monkeypatch.setenv("GLM53_KV_STORE_WORKER", "1")
    monkeypatch.setenv("GLM53_KV_STORE_WORKER_MAX_GB", cap_gb)
    h = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    h.connector, h.tt_gpu_connector = _Conn(), _GPU()
    h.pooled_pages, h._generic_sync = False, False
    h.views = {"l0": SimpleNamespace(block_size=64)}
    h._lock, h._metadata = threading.Lock(), (lambda so: so)
    h._step_context = lambda mi: (None, {"l0": None})
    h.connector_caches = {"l0": None}
    h._deferred_save, h._held_pages = None, []
    h._enable_store_worker(h.connector)
    assert h.store_worker is not None
    return h


def _meta(skip, slots, can_save=True):
    return SimpleNamespace(requests=[SimpleNamespace(
        save_spec=SimpleNamespace(can_save=can_save, skip_leading_tokens=skip), slot_mapping=torch.tensor(slots))])


def test_store_runs_on_the_worker_with_its_own_metadata(monkeypatch):
    h = _host(monkeypatch)
    impl = h.connector._lmcache_engine
    impl.gate.clear()  # the worker's store blocks: the engine thread must not
    m = _meta(128, list(range(256)))
    t0 = time.perf_counter()
    h._save_pages(m, None)
    assert time.perf_counter() - t0 < 1.0 and impl.calls == []
    assert h.tt_gpu_connector.prefetch == [(threading.current_thread().name, [2, 3])]  # pages past the saved prefix
    h.connector.bind_connector_metadata("engine-step")  # the engine thread binds its own next step meanwhile
    impl.gate.set()
    h.store_worker.drain()
    assert impl.calls == [("tt-kv-store-worker", m)] and h.connector.bound == "engine-step"
    assert h.tt_gpu_connector.current_store_job() is None and h.store_worker.stats["done"] == 1


def test_a_step_without_a_store_stays_inline(monkeypatch):
    h = _host(monkeypatch)
    m = _meta(0, list(range(64)), can_save=False)
    h._save_pages(m, None)
    assert h.tt_gpu_connector.prefetch == []
    assert h.connector._lmcache_engine.calls == [(threading.current_thread().name, m)]  # unpins, on this thread


def test_back_pressure_blocks_the_engine_only_past_the_byte_cap(monkeypatch):
    h = _host(monkeypatch, cap_gb=str(25 / 2**30))  # 25 bytes in flight at most; a 2-page job = 20 bytes
    impl = h.connector._lmcache_engine
    impl.gate.clear()
    h._save_pages(_meta(0, list(range(128))), None)  # in flight: 20
    done = threading.Event()
    th = threading.Thread(target=lambda: (h._save_pages(_meta(0, list(range(128, 256))), None), done.set()))
    th.start()
    assert not done.wait(0.5)  # 20 + 20 > 25: the second submit waits
    impl.gate.set()
    assert done.wait(5)
    th.join()
    h.store_worker.drain()
    assert len(impl.calls) == 2 and h.store_worker.stats["submit_wait_s"] > 0.3


def test_lookup_flush_drains_the_queue_unless_hits_cannot_resume(monkeypatch):
    h = _host(monkeypatch)
    impl = h.connector._lmcache_engine
    impl.gate.clear()
    h._save_pages(_meta(0, list(range(128))), None)
    h.checkpoints_off = True
    h.flush_deferred_save()  # no hit can resume: the lookup does not wait for the queue
    assert impl.calls == []
    h.checkpoints_off = False
    done = threading.Event()
    th = threading.Thread(target=lambda: (h.flush_deferred_save(), done.set()))
    th.start()
    assert not done.wait(0.3)  # a lookup that may hit waits for every queued store
    impl.gate.set()
    assert done.wait(5)
    th.join()
    h.shutdown()
    assert h.connector.shut and not h.store_worker.thread.is_alive()


def test_store_page_ids_follow_the_save_specs():
    h = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    h.connector = _Conn()
    h.views = {"l0": SimpleNamespace(block_size=64)}
    m = SimpleNamespace(requests=[
        SimpleNamespace(save_spec=SimpleNamespace(can_save=True, skip_leading_tokens=200),  # aligned down to 128
                        slot_mapping=torch.tensor([64 * 10 + i for i in range(64)] + [64 * 4 + i for i in range(64)]
                                                  + [64 * 7 + i for i in range(64)] + [64 * 9 + i for i in range(64)])),
        SimpleNamespace(save_spec=None, slot_mapping=torch.tensor([0, 1])),
        SimpleNamespace(save_spec=SimpleNamespace(can_save=False, skip_leading_tokens=0), slot_mapping=torch.tensor([5])),
    ])
    assert sorted(h._store_page_ids(m)) == [7, 9]


def test_the_worker_completes_enqueued_reads_before_the_store(monkeypatch):
    """the job's enqueued page reads (GLM53_KV_STORE_ASYNC_READ) are completed on the worker, before LMCache's store"""
    h = _host(monkeypatch)
    order: list = []
    h.tt_gpu_connector.complete_store_job = lambda job: order.append(("complete", threading.current_thread().name))
    impl = h.connector._lmcache_engine
    orig = impl.wait_for_save
    impl.wait_for_save = lambda: (order.append(("store", threading.current_thread().name)), orig())
    h._save_pages(_meta(0, list(range(128))), None)
    h.store_worker.drain()
    assert order == [("complete", "tt-kv-store-worker"), ("store", "tt-kv-store-worker")]


# --- store path P3 (GLM53_KV_STORE_ZC_WORKER, program 2026-10-26T23:48:49Z): the zero-copy store whole on the worker


class _ZcGPU(_GPU):
    def zero_copy_store(self):
        return True


def _zc_host(monkeypatch, safe=True, leave=True, paged=True, pooled=True):
    monkeypatch.setenv("GLM53_KV_STORE_ZC_WORKER", "1")
    monkeypatch.setenv("GLM53_KV_LEAVE_DRAIN", "1" if leave else "0")
    monkeypatch.setenv("GLM53_DSA_PAGED_POOLS", "1" if paged else "0")
    h = _host(monkeypatch)
    h.tt_gpu_connector = _ZcGPU()
    h.hit_dma = SimpleNamespace(thread_safe=safe)
    h.pooled_pages = pooled
    h._page_columns = lambda mi: {"step": mi}
    seen = []
    impl = h.connector._lmcache_engine
    orig = impl.wait_for_save

    def wait_for_save():  # the store sees the job's pooled map on its own thread
        seen.append(getattr(h._store_meta_tl, "pooled_map", None))
        orig()

    impl.wait_for_save = wait_for_save
    return h, seen


def _mi(rids):
    return SimpleNamespace(row_req_ids=list(rids))


def test_a_zero_copy_store_runs_whole_on_the_worker(monkeypatch):
    h, seen = _zc_host(monkeypatch)
    impl = h.connector._lmcache_engine
    impl.gate.clear()
    m = _meta(128, list(range(256)))
    t0 = time.perf_counter()
    h._save_pages(m, _mi(["a"]))
    assert time.perf_counter() - t0 < 1.0 and impl.calls == []  # the engine thread does not wait for the store
    assert h.tt_gpu_connector.prefetch == []  # no engine-thread read: the worker's store reads through the page-DMA
    assert h._inflight_rid_counts == {"a": 1}  # M1: the request's blocks stay held
    impl.gate.set()
    h.store_worker.drain()
    assert impl.calls == [("tt-kv-store-worker", m)] and seen == [{"step": _mi(["a"])}]
    assert h.__dict__.get("_inflight_rid_counts", {}) == {} and h.zc_worker_jobs == 1
    assert h.tt_gpu_connector.current_store_job() is None


def test_a_preempted_requests_job_lands_before_begin_step_returns(monkeypatch):
    h, _seen = _zc_host(monkeypatch)
    impl = h.connector._lmcache_engine
    impl.gate.clear()
    h._save_pages(_meta(128, list(range(256))), _mi(["a"]))
    threading.Timer(0.3, impl.gate.set).start()
    t0 = time.perf_counter()
    h._zc_drain_for({"a"})  # begin_step's drain: "a" was preempted this step
    assert time.perf_counter() - t0 >= 0.25 and impl.calls and h.zc_preempt_drains == 1
    h._zc_drain_for({"b"})  # nothing of "b" in flight: no wait
    assert h.zc_preempt_drains == 1


def test_zero_copy_worker_needs_held_blocks_a_safe_dma_and_paged_pools(monkeypatch):
    for kw in ({"safe": False}, {"leave": False}, {"paged": False}):
        h, _seen = _zc_host(monkeypatch, **kw)
        m = _meta(128, list(range(256)))
        h._save_pages(m, _mi(["a"]))
        assert h.connector._lmcache_engine.calls == [(threading.current_thread().name, m)], kw  # inline, as before
    h, _seen = _zc_host(monkeypatch, paged=False, pooled=False)  # no pooled pages: slot units do not matter
    h._save_pages(_meta(128, list(range(256))), _mi(["a"]))
    h.store_worker.drain()
    assert h.connector._lmcache_engine.calls[0][0] == "tt-kv-store-worker"


def test_shutdown_drains_the_store_worker_before_the_dma_closes(monkeypatch):
    """P3 at shutdown: TTWorker.shutdown releases the model (its page-DMA closes) before the connector's shutdown, so
    drain_store_workers() runs first -- a queued zero-copy job lands while the DMA is open, later stores run inline"""
    h, _seen = _zc_host(monkeypatch)
    impl = h.connector._lmcache_engine
    impl.gate.clear()
    h._save_pages(_meta(128, list(range(256))), _mi(["a"]))
    threading.Timer(0.3, impl.gate.set).start()
    K.drain_store_workers()
    assert impl.calls and impl.calls[0][0] == "tt-kv-store-worker"  # landed before the drain returned
    m = _meta(128, list(range(256)))
    h._save_pages(m, _mi(["b"]))
    assert impl.calls[-1] == (threading.current_thread().name, m)  # after the drain: inline


def test_the_shutdown_drain_is_bounded(monkeypatch):
    """a wedged store never keeps the mesh from closing: drain_store_workers waits at most GLM53_KV_SHUTDOWN_DRAIN_S"""
    h, _seen = _zc_host(monkeypatch)
    impl = h.connector._lmcache_engine
    impl.gate.clear()  # the store never finishes (until released below)
    h._save_pages(_meta(128, list(range(256))), _mi(["a"]))
    monkeypatch.setenv("GLM53_KV_SHUTDOWN_DRAIN_S", "0.3")
    t0 = time.perf_counter()
    K.drain_store_workers()
    assert 0.25 <= time.perf_counter() - t0 < 3.0
    impl.gate.set()
    h.store_worker.drain()


def test_the_counters_line_sums_every_source(monkeypatch):
    """GLM53_KV_COUNTERS_LOG_S: one INFO line of the Stage 12 counters from the host, the scheduler-side connector and
    the scheduler (registered at construction); printed when something changed, forced at shutdown"""
    h, _seen = _zc_host(monkeypatch)
    K._COUNTER_SOURCES["host"].add(h)
    look = K.TTPageAlignedConnector(SimpleNamespace(), 64)
    look.__dict__["preempt_exact_hits"] = 3
    h.zc_worker_jobs = 2
    class _Sched:  # (a weak-referenceable stand-in of TTScheduler)
        admissions_deferred_kv = 0

    sched = _Sched()
    sched.admissions_deferred_kv = 5
    K._COUNTER_SOURCES["sched"].add(sched)
    snap = K.counters_snapshot()
    assert snap["preempt_exact_hits"] == 3 and snap["zc_worker_jobs"] >= 2 and snap["admissions_deferred_kv"] == 5
    monkeypatch.setattr(K, "_COUNTERS_LAST", {"t": 0.0, "snap": None})
    lines = []
    monkeypatch.setattr(K.logger, "info", lambda fmt, *a: lines.append(fmt % a))  # (the plugin logger does not propagate)
    K.log_counters(force=True)
    assert any(l.startswith("TT KV connector counters") and '"preempt_exact_hits": 3' in l for l in lines), lines
    K.log_counters()  # unchanged and inside the period: no second line
    assert len(lines) == 1
