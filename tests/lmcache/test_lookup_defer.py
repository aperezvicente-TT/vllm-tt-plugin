"""Stage 12 (program 2026-10-26T07:23:04Z): GLM53_KV_LOOKUP_DEFER=1 -- a connector lookup whose prompt shares the
first LMCache chunk of a host's deferred / held / queued page stores no longer DRAINS the store worker on the engine
thread (a back-to-back repeat paid 0.53 / 1.57 s at 16K / 32K): the held stores go to the worker as before, and while
they are in flight the lookup is reported undetermined (None: vLLM skips the request this step and retries); other
prompts never touch the host; past GLM53_KV_LOOKUP_DEFER_MAX_S the old flush + drain runs."""

import threading
import time
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import kv_connector as K

from .test_lookup_flush_prefix import CHUNK, NXT, _host, _step

A = list(range(100, 1100))


class _Worker:
    """the store worker's surface the host and the lookup use: busy(), drain (counted), wait_idle (busy until released)"""

    def __init__(self):
        self.drains, self.waits, self.inflight = 0, 0, True

    def busy(self):
        return self.inflight

    def drain(self):
        self.drains += 1
        self.inflight = False

    def wait_idle(self, timeout_s):
        self.waits += 1
        return not self.inflight


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(K, "_SAVE_HOSTS", set())
    monkeypatch.setattr(K, "_LOOKUP_DEFERRED", {})
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER", "1")
    monkeypatch.delenv("GLM53_KV_LOOKUP_FLUSH", raising=False)  # (DEFER implies the prefix filter)
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER",
              "GLM53_KV_STORE_FT_GRACE", "GLM53_KV_LOOKUP_DEFER_MAX_S", "GLM53_KV_LOOKUP_DEFER_WAIT_MS"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)


def _worker_host():
    """the stub host of test_lookup_flush_prefix with a store worker: a flushed held step becomes a queued job of its
    request (queued heads +1) -- the worker 'finishes' them when the test releases it"""
    host, pages, _drains = _host({"A": A, "C": [5, 6, 7, 8, 9]})
    w = host.store_worker = _Worker()
    jobs = []

    def save_pages(so, mi):
        pages.append(mi.tag)
        meta = SimpleNamespace(requests=[SimpleNamespace(req_id=r) for r in mi.row_req_ids])
        jobs.append((meta, host._queue_heads(meta, +1)))

    host._save_pages = save_pages

    def release():
        for meta, heads in jobs:
            host._queue_heads(meta, -1, heads)
        jobs.clear()
        w.inflight = False

    return host, pages, w, release


def _hold_a(host):
    for tag, s0, e in (("a1", 0, 128), ("a2", 128, 256)):
        host.save(*_step(tag, ("A", s0, e)))
        host.begin_step(NXT)
    host.save(*_step("a3", ("A", 256, 384)))  # deferred (pending) at the lookup


class _Inner:
    def __init__(self):
        self.calls = 0

    def get_num_new_matched_tokens(self, request, n):
        self.calls += 1
        return 0, False


def _req(tokens, rid="B"):
    return SimpleNamespace(request_id=rid, prompt_token_ids=tokens, num_prompt_tokens=len(tokens),
                           sampling_params=SimpleNamespace(extra_args=None), kv_transfer_params=None)  # fmt: skip


def test_a_repeat_defers_while_its_stores_are_in_flight_then_looks_up_without_a_drain():
    host, pages, w, release = _worker_host()
    _hold_a(host)
    inner = _Inner()
    c = K.TTPageAlignedConnector(inner, 128, None)
    r = _req(A[:CHUNK] + [1, 2, 3])
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)  # undetermined: vLLM retries next step
    assert pages == ["a1", "a2", "a3"] and w.drains == 0 and inner.calls == 0  # handed to the worker, never drained
    assert c.get_num_new_matched_tokens(r, 0) == (None, False) and pages == ["a1", "a2", "a3"]  # nothing re-submitted
    release()  # the worker landed A's stores
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)
    assert inner.calls == 1 and w.drains == 0 and c.lookups_deferred == 2 and "B" not in K._LOOKUP_DEFERRED


def test_the_deferral_and_its_release_are_logged():
    import logging

    lines = []
    h = logging.Handler()
    h.emit = lambda rec: lines.append(rec.getMessage())
    K.logger.addHandler(h)
    old = K.logger.level
    K.logger.setLevel(logging.INFO)
    try:
        host, _pages, _w, release = _worker_host()
        _hold_a(host)
        c = K.TTPageAlignedConnector(_Inner(), 128, None)
        r = _req(A[:CHUNK], rid="Bx")
        c.get_num_new_matched_tokens(r, 0)
        c.get_num_new_matched_tokens(r, 0)
        release()
        c.get_num_new_matched_tokens(r, 0)
    finally:
        K.logger.removeHandler(h)
        K.logger.setLevel(old)
    text = "\n".join(lines)
    assert text.count("lookup of Bx deferred") == 1 and "lookup of Bx released after" in text
    assert "0 external tokens" in text and "Bx" not in K._LOOKUP_RELEASED


def test_another_prompt_neither_defers_nor_flushes_the_host():
    host, pages, w, _release = _worker_host()
    _hold_a(host)
    inner = _Inner()
    c = K.TTPageAlignedConnector(inner, 128, None)
    assert c.get_num_new_matched_tokens(_req([7, 8, 9, 10, 11]), 0) == (0, False)
    assert pages == [] and w.drains == 0 and inner.calls == 1 and w.waits == 0


def test_a_short_worker_wait_avoids_the_deferral_when_the_stores_land(monkeypatch):
    host, pages, w, release = _worker_host()
    _hold_a(host)

    def wait_idle(timeout_s):
        release()  # the stores land during the bounded wait
        return True

    w.wait_idle = wait_idle
    inner = _Inner()
    c = K.TTPageAlignedConnector(inner, 128, None)
    assert c.get_num_new_matched_tokens(_req(A[:CHUNK]), 0) == (0, False)
    assert pages == ["a1", "a2", "a3"] and w.drains == 0 and inner.calls == 1


def test_past_the_cap_the_old_flush_and_drain_run(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER_MAX_S", "0.05")
    host, pages, w, _release = _worker_host()
    _hold_a(host)
    inner = _Inner()
    c = K.TTPageAlignedConnector(inner, 128, None)
    r = _req(A[:CHUNK])
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)
    time.sleep(0.08)
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)  # progress guaranteed: drain, then the lookup
    assert w.drains == 1 and inner.calls == 1


def test_knob_off_drains_at_the_lookup_as_before(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER", "0")
    host, pages, w, _release = _worker_host()
    _hold_a(host)
    inner = _Inner()
    c = K.TTPageAlignedConnector(inner, 128, None)
    assert c.get_num_new_matched_tokens(_req(A[:CHUNK]), 0) == (0, False)
    assert pages == ["a1", "a2", "a3"] and w.drains == 1


def test_store_worker_wait_idle_is_bounded():
    host = SimpleNamespace(_run_store_job=lambda meta, job: time.sleep(0.2))
    w = K._StoreWorker(host, 1 << 30)
    try:
        assert w.wait_idle(0.01)  # idle
        w.submit(None, {"nbytes": 10})
        t0 = time.perf_counter()
        assert not w.wait_idle(0.02) and time.perf_counter() - t0 < 0.15
        assert w.wait_idle(2.0)
    finally:
        w.close()


def _real_save_host():
    """the stub host with the REAL _save_pages worker branch (prefetch -> heads -> submit); the j3cngx2 mock
    (2026-10-26 09:48Z) failed exactly there: heads were registered only under GLM53_KV_LOOKUP_FLUSH=prefix"""
    host, _pages, _drains = _host({"A": A})
    host._save_pages = K.TTKVConnectorHost._save_pages.__get__(host)
    w = host.store_worker = _Worker()
    jobs = []
    w.submit = lambda meta, job: jobs.append((meta, job))
    host._generic_sync = False
    host._metadata = lambda so: SimpleNamespace(requests=[SimpleNamespace(req_id="A")])
    host._store_page_ids = lambda meta: [1, 2]
    host.tt_gpu_connector = SimpleNamespace(kvcaches=object(), prefetch_store_pages=lambda pages: {"pages": pages})
    host.pooled_pages = False
    K._SAVE_HOSTS.add(host)
    return host, w, jobs


def test_the_real_store_path_registers_heads_for_the_deferral_under_the_served_flush_default():
    """GLM53_KV_LOOKUP_FLUSH unset (= all, as served) + DEFER: the real worker branch records the job's request heads,
    so a repeat's lookup waits for A's queued store instead of looking up a partial store"""
    host, w, jobs = _real_save_host()
    host._save_pages(SimpleNamespace(), SimpleNamespace())
    assert len(jobs) == 1 and jobs[0][1].get("heads") == [tuple(A[:CHUNK])]
    c = K.TTPageAlignedConnector(_Inner(), 128, None)
    r = _req(A[:CHUNK] + [1, 2])
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)  # waits for A's store
    meta, job = jobs[0]
    host._queue_heads(meta, -1, job["heads"])  # the worker finished the job
    w.inflight = False
    assert c.get_num_new_matched_tokens(r, 0) == (0, False) and w.drains == 0


def test_without_defer_the_served_default_records_no_heads(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER", "0")
    host, w, jobs = _real_save_host()
    host._save_pages(SimpleNamespace(), SimpleNamespace())
    assert "heads" not in jobs[0][1]
