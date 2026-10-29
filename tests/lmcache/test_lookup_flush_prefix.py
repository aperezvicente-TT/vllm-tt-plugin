"""Stage 12 W4 F2b (2026-10-24, doc/ttnn/12-fast-start/w4_host): GLM53_KV_LOOKUP_FLUSH=prefix -- a connector lookup
flushes (and drains) only the held / deferred / queued page stores of requests that share the new prompt's first
LMCache chunk (LMCache's chunk keys chain from chunk 0: no other store can serve its hit). A prompt arriving while a
long prompt prefills no longer pays every held step's page reads on its own first token."""

import threading
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import kv_connector as K

CHUNK = 4


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(K, "_SAVE_HOSTS", set())
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_LOOKUP_FLUSH", "prefix")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER",
              "GLM53_KV_STORE_FT_GRACE"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)


def _host(histories):
    """a stub host: histories {req_id: tokens}; LMCache chunk = CHUNK tokens; a store worker whose drains are counted"""
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host._deferred_save, host.saves_deferred = None, 0
    pages = []
    host._save_state = lambda mi, out=None: None
    host._save_pages = lambda so, mi: pages.append(mi.tag)
    host._save_now = lambda so, mi, out=None: pages.append(mi.tag)
    host._tokens_of = lambda rid: histories.get(rid)
    host._head_of_fn = lambda rid, n: None if rid not in histories else histories[rid][:n]
    host._lmcache_chunk = lambda: CHUNK
    host._prefill_rows = lambda mi: ([(i, rid, s0, e) for i, (rid, s0, e) in enumerate(mi.rows)]
                                     if mi.prompt_lens is not None else [])  # fmt: skip
    host.connector = SimpleNamespace(bind_connector_metadata=lambda m: None, clear_connector_metadata=lambda: None,
                                     handle_preemptions=lambda m: None)  # fmt: skip
    host._metadata, host._lock = (lambda so: None), threading.Lock()
    drains = []
    host.store_worker = SimpleNamespace(drain=lambda: drains.append(1))
    return host, pages, drains


def _step(tag, *rows):
    so = SimpleNamespace(finished_req_ids=set())
    mi = SimpleNamespace(prompt_lens=[e for _r, _s, e in rows], row_req_ids=[r for r, _s, _e in rows], rows=list(rows),
                         tag=tag)  # fmt: skip
    return so, mi


A = list(range(100, 1100))  # a long prompt, mid-prefill
NXT = SimpleNamespace(finished_req_ids=set())


def _hold_a(host):
    for tag, s0, e in (("a1", 0, 128), ("a2", 128, 256)):
        host.save(*_step(tag, ("A", s0, e)))
        host.begin_step(NXT)
    host.save(*_step("a3", ("A", 256, 384)))  # deferred (pending) at the lookup


def test_a_lookup_for_another_prompt_leaves_the_held_stores_alone():
    host, pages, drains = _host({"A": A})
    _hold_a(host)
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=[7, 8, 9, 10, 11]))
    assert pages == [] and drains == [] and host.lookup_flushes_skipped == 1
    assert host._deferred_save is not None  # its checkpoints run at the next begin_step, as before
    host.begin_step(NXT)
    assert pages == [] and [mi.tag for _so, mi in host._held_pages] == ["a1", "a2", "a3"]


def test_a_lookup_sharing_the_first_chunk_flushes_and_drains_as_before():
    """a multi-turn follow-up (A's prompt + more) or a repeat: its hit needs A's stores, so everything flushes"""
    host, pages, drains = _host({"A": A})
    _hold_a(host)
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=A[:CHUNK] + [1, 2, 3]))
    assert pages == ["a1", "a2", "a3"] and drains == [1]


def test_a_short_prompt_matching_the_first_tokens_flushes():
    host, pages, _drains = _host({"A": A})
    _hold_a(host)
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=A[:2]))  # shorter than a chunk: compared on its length
    assert pages == ["a1", "a2", "a3"]


def test_an_unknown_history_flushes():
    host, pages, _drains = _host({"A": A})
    _hold_a(host)
    host._head_of_fn = lambda rid, n: None  # the runner no longer knows A
    # ... and no head was kept from its stored step (since 2026-10-26 22:40:55Z (b) a held step keeps its request's head:
    # test_leave_drain.py::test_a_gone_request_keeps_its_head_so_an_unrelated_lookup_does_not_flush_it)
    host.__dict__.pop("_rid_heads", None)
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=[7, 8, 9, 10]))
    assert pages == ["a1", "a2", "a3"]


def test_queued_worker_stores_count_until_done():
    """stores already handed to the store worker: a sharing lookup drains them, another prompt's does not"""
    host, pages, drains = _host({"A": A, "C": [5, 6, 7, 8, 9]})
    K._SAVE_HOSTS.add(host)
    host._held_pages = []
    meta = SimpleNamespace(requests=[SimpleNamespace(req_id="C")])
    heads = host._queue_heads(meta, +1)
    assert heads == [(5, 6, 7, 8)]
    assert host.serves_prefix([5, 6, 7, 8, 1]) and not host.serves_prefix([1, 2, 3, 4])
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=[1, 2, 3, 4]))
    assert drains == []
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=[5, 6, 7, 8, 1]))
    assert drains == [1]
    host._queue_heads(meta, -1, heads)
    assert not host.serves_prefix([5, 6, 7, 8, 1]) and host._queued_heads == {}


def test_knob_off_flushes_everything_at_any_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_KV_LOOKUP_FLUSH", "all")
    host, pages, drains = _host({"A": A})
    _hold_a(host)
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=[7, 8, 9, 10, 11]))
    assert pages == ["a1", "a2", "a3"] and drains == [1]


def test_the_lookup_passes_its_request(monkeypatch):
    got = []
    monkeypatch.setattr(K, "flush_deferred_saves", lambda request=None: got.append(request))

    class Inner:
        def get_num_new_matched_tokens(self, request, n):
            return 0, False

    c = K.TTPageAlignedConnector(Inner(), 128, None)
    r = SimpleNamespace(request_id="r", prompt_token_ids=[1, 2, 3], num_prompt_tokens=3,
                        sampling_params=SimpleNamespace(extra_args=None), kv_transfer_params=None)  # fmt: skip
    try:
        c.get_num_new_matched_tokens(r, 0)
    except Exception:  # noqa: BLE001 - only the flush call's argument matters here
        pass
    assert got and got[0] is r


def test_runner_head_reads_only_the_first_tokens():
    runner = SimpleNamespace(requests={"r": SimpleNamespace(prompt_token_ids=[1, 2], output_token_ids=[3, 4, 5])})
    assert K._runner_request_head(runner, "r", 4) == [1, 2, 3, 4]
    assert K._runner_request_head(runner, "r", 1) == [1]
    assert K._runner_request_head(runner, "x", 4) is None


def test_a_slow_lookup_flush_is_logged(monkeypatch):
    import logging
    import time as _t

    monkeypatch.setenv("GLM53_KV_PHASE_LOG_MS", "1")
    host, pages, drains = _host({"A": A})
    host.store_worker.drain = lambda: (_t.sleep(0.01), drains.append(1))
    _hold_a(host)
    lines = []
    h = logging.Handler()
    h.emit = lambda rec: lines.append(rec.getMessage())
    K.logger.addHandler(h)
    old = K.logger.level
    K.logger.setLevel(logging.INFO)
    try:
        K.flush_deferred_saves(SimpleNamespace(request_id="R1", prompt_token_ids=A[:CHUNK]))
        K.flush_deferred_saves(SimpleNamespace(request_id="R2", prompt_token_ids=[7, 8, 9, 10]))  # nothing flushed
    finally:
        K.logger.removeHandler(h)
        K.logger.setLevel(old)
    got = [x for x in lines if "lookup flush" in x]
    assert len(got) == 1 and "(R1)" in got[0] and "1 host(s)" in got[0]
