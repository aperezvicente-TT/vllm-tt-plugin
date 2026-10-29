"""GLM53_KV_LOOKUP_ADMIT_FIRST=1 (program 2026-10-28T05:07:22Z (3), the k4 fix item): under GLM53_KV_LOOKUP_DEFER a
lookup waits only for an in-flight store that writes INSIDE the common prefix of its request and the new prompt. The
k4d5 warm 16,100 repeat waited 225 ms behind page stores that could not change its LMCache match. A queued job records
(its request's history, the chunk-aligned start of the span it writes); a job past the common prefix no longer defers
the lookup (counted lookups_admitted_first), a job inside it still does, and a job of unknown span keeps the old rule."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin import kv_connector as K

from .test_lookup_defer import _Inner, _req, _Worker
from .test_lookup_flush_prefix import CHUNK, NXT, _host, _step

A = list(range(100, 1100))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(K, "_SAVE_HOSTS", set())
    monkeypatch.setattr(K, "_LOOKUP_DEFERRED", {})
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_LOOKUP_DEFER", "1")
    monkeypatch.setenv("GLM53_KV_LOOKUP_ADMIT_FIRST", "1")
    monkeypatch.delenv("GLM53_KV_LOOKUP_FLUSH", raising=False)
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER",
              "GLM53_KV_STORE_FT_GRACE", "GLM53_KV_LOOKUP_DEFER_MAX_S", "GLM53_KV_LOOKUP_DEFER_WAIT_MS"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)


def _span_host(spec=True):
    """the stub host with a store worker whose jobs carry their spans; jobs land one at a time (land(start))"""
    host, pages, _drains = _host({"A": A})
    w = host.store_worker = _Worker()
    jobs = []

    def save_pages(so, mi):
        pages.append(mi.tag)
        reqs = []
        for rid, s0, _e in mi.rows:
            ss = SimpleNamespace(skip_leading_tokens=s0, can_save=True) if spec else None
            reqs.append(SimpleNamespace(req_id=rid, save_spec=ss))
        meta = SimpleNamespace(requests=reqs)
        job = {"heads": host._queue_heads(meta, +1)}
        host._queue_spans(meta, +1, job)
        jobs.append((meta, job, mi.rows[0][1]))

    host._save_pages = save_pages

    def land(below):
        """the worker finishes every job whose span starts before `below`"""
        for item in [j for j in jobs if j[2] < below]:
            meta, job, _s = item
            host._queue_heads(meta, -1, job["heads"])
            host._queue_spans(meta, -1, job)
            jobs.remove(item)
        w.inflight = bool(jobs)

    return host, pages, w, land


def _hold_a(host):
    for tag, s0, e in (("a1", 0, 128), ("a2", 128, 256)):
        host.save(*_step(tag, ("A", s0, e)))
        host.begin_step(NXT)
    host.save(*_step("a3", ("A", 256, 384)))  # deferred (pending) at the lookup


def test_a_store_past_the_common_prefix_does_not_defer_the_lookup():
    host, pages, w, land = _span_host()
    _hold_a(host)
    c = K.TTPageAlignedConnector(_Inner(), 128, None)
    r = _req(A[:200] + [1, 2, 3])  # shares A's first 200 tokens
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)  # a1 [0, 128) / a2 [128, 256) write inside the prefix
    land(256)  # a1 and a2 landed; a3 writes [256, 384): past the 200-token common prefix
    assert w.inflight and host.writes_into_prefix(r.prompt_token_ids) is False
    assert host.serves_prefix(r.prompt_token_ids)  # (the old rule: chunk 0 shared -> it would defer)
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)  # admitted now, a3 still in flight
    assert host.lookups_admitted_first == 1 and w.drains == 0


def test_a_store_inside_the_common_prefix_still_defers():
    host, _pages, w, land = _span_host()
    _hold_a(host)
    c = K.TTPageAlignedConnector(_Inner(), 128, None)
    r = _req(A[:300] + [1])  # the common prefix reaches into a3's span [256, 384)
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)  # (hands the held stores to the worker)
    land(256)
    assert host.writes_into_prefix(r.prompt_token_ids) is True
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)
    land(10_000)
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)


def test_an_unknown_span_keeps_the_old_rule():
    host, _pages, _w, land = _span_host(spec=False)
    _hold_a(host)
    c = K.TTPageAlignedConnector(_Inner(), 128, None)
    r = _req(A[:200] + [1, 2, 3])
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)
    land(256)
    assert host.writes_into_prefix(r.prompt_token_ids) is True  # no save_spec: a span of unknown start
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)


def test_off_the_knob_the_same_case_defers(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LOOKUP_ADMIT_FIRST", "0")
    host, _pages, w, land = _span_host()
    _hold_a(host)
    c = K.TTPageAlignedConnector(_Inner(), 128, None)
    r = _req(A[:200] + [1, 2, 3])
    assert c.get_num_new_matched_tokens(r, 0) == (None, False)
    land(256)  # (off the knob no spans are recorded: land() only lowers the queued heads)
    assert w.inflight and c.get_num_new_matched_tokens(r, 0) == (None, False)
    assert not getattr(host, "lookups_admitted_first", 0) and not getattr(host, "_queued_spans", None)


def test_a_different_first_chunk_never_touches_the_host():
    host, pages, w, _land = _span_host()
    _hold_a(host)
    c = K.TTPageAlignedConnector(_Inner(), 128, None)
    assert c.get_num_new_matched_tokens(_req([7, 8, 9, 10, 11]), 0) == (0, False)
    assert pages == [] and w.waits == 0 and CHUNK == 4
