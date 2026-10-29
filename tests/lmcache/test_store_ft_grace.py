"""Stage 12 W4 (2026-10-24, doc/ttnn/12-fast-start/w4_host): the held page stores off the first-token path.

vLLM's batch queue runs step N + 1 (begin_step, forward, save) before it pops step N's output, so a page store flushed
there delays the first token of a prompt whose last chunk ran in N. GLM53_KV_STORE_FT_GRACE=1 keeps every page store
held from that begin_step until the next one; GLM53_KV_STORE_HOLD_OVER=excess flushes only the steps above
GLM53_KV_STORE_HOLD_MAX instead of all of them."""

import threading
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import kv_connector as K


@pytest.fixture(autouse=True)
def _own_save_hosts(monkeypatch):
    """flush_deferred_saves walks the module's registry of hosts with deferred saves: this module's stub hosts only
    (a full-suite run leaves real hosts of earlier modules there)"""
    monkeypatch.setattr(K, "_SAVE_HOSTS", set())


def _host(prompt_len):
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host._deferred_save, host.saves_deferred = None, 0
    pages = []
    host._save_state = lambda mi, out=None: None
    host._save_pages = lambda so, mi: pages.append(mi.tag)
    host._save_now = lambda so, mi, out=None: pages.append(mi.tag)
    host._tokens_of = lambda rid: list(range(prompt_len))
    host._prefill_rows = lambda mi: [(0, "r", mi.start, mi.end)] if mi.prompt_lens is not None else []
    host.connector = SimpleNamespace(bind_connector_metadata=lambda m: None, clear_connector_metadata=lambda: None,
                                     handle_preemptions=lambda m: None)  # fmt: skip
    host._metadata, host._lock = (lambda so: None), threading.Lock()
    return host, pages


NXT = SimpleNamespace(finished_req_ids=set())


def _pre(tag, s0, e):
    return NXT, SimpleNamespace(prompt_lens=[e], row_req_ids=["r"], start=s0, end=e, tag=tag)


def _dec(tag):
    return NXT, SimpleNamespace(prompt_lens=None, row_req_ids=["o"], tag=tag)


def _chunks(host, spans):
    for tag, s0, e in spans:
        host.save(*_pre(tag, s0, e))
        host.begin_step(NXT)


def test_grace_keeps_the_page_stores_off_the_step_that_returns_the_first_token(monkeypatch):
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_FT_GRACE", "1")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER"):
        monkeypatch.delenv(k, raising=False)
    host, pages = _host(400)  # chunks end 128, 256, 384, 400
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256), ("c3", 256, 384)])
    assert pages == []  # mid-prefill: held
    host.save(*_pre("c4", 384, 400))  # the last chunk: the first token is sampled in this step
    host.begin_step(NXT)  # step N + 1 (its output pops only after this step): nothing flushes
    assert pages == []
    host.save(*_dec("d1"))  # step N + 1's own save (a decode step): itself only
    assert pages == ["d1"]
    host.begin_step(NXT)  # step N + 2: the first token was popped -> one held step per engine step, oldest first
    assert pages == ["d1", "c1"]
    host.save(*_dec("d2"))
    assert pages[-2:] == ["c2", "d2"]
    host.begin_step(NXT)
    assert pages[-1] == "c3" and [t for _so, t in ((so, mi.tag) for so, mi in host._held_pages)] == ["c4"]


def test_grace_never_holds_a_leaving_request_or_a_lookup(monkeypatch):
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_FT_GRACE", "1")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER"):
        monkeypatch.delenv(k, raising=False)
    host, pages = _host(300)
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256)])
    host.save(*_pre("c3", 256, 300))
    host.begin_step(SimpleNamespace(finished_req_ids={"r"}))  # leaving in the grace step: its blocks are reused next
    assert pages == ["c1", "c2", "c3"] and not host._held_pages
    host2, pages2 = _host(300)
    _chunks(host2, [("e1", 0, 128), ("e2", 128, 256)])
    host2.save(*_pre("e3", 256, 300))
    host2.begin_step(NXT)
    assert pages2 == []
    K.flush_deferred_saves()  # a lookup in the grace step: every held store (a hit must see them)
    assert pages2 == ["e1", "e2", "e3"]


def test_grace_off_is_the_budgeted_flush_at_the_first_token_step(monkeypatch):
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    for k in ("GLM53_KV_STORE_FT_GRACE", "GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX",
              "GLM53_KV_STORE_HOLD_OVER"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    host, pages = _host(400)
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256), ("c3", 256, 384)])
    host.save(*_pre("c4", 384, 400))
    host.begin_step(NXT)
    assert pages == ["c1"]  # today: the oldest held step flushes on the first token's step


def test_hold_over_excess_flushes_only_the_steps_above_hold_max(monkeypatch):
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_HOLD_MAX", "2")
    monkeypatch.setenv("GLM53_KV_STORE_HOLD_OVER", "excess")
    for k in ("GLM53_KV_STORE_FT_GRACE", "GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER"):
        monkeypatch.delenv(k, raising=False)
    host, pages = _host(1000)  # a long prompt, every chunk intermediate below
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256)])
    assert pages == []
    _chunks(host, [("c3", 256, 384)])  # 3 held > 2: only the oldest one
    assert pages == ["c1"] and len(host._held_pages) == 2
    _chunks(host, [("c4", 384, 512)])
    assert pages == ["c1", "c2"] and len(host._held_pages) == 2
    monkeypatch.setenv("GLM53_KV_STORE_HOLD_OVER", "all")  # today: past HOLD_MAX everything at once
    _chunks(host, [("c5", 512, 640)])
    assert pages == ["c1", "c2", "c3", "c4", "c5"] and not host._held_pages


def test_hold_over_excess_with_grace_flushes_only_the_excess_on_the_first_token_step(monkeypatch):
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_HOLD_MAX", "2")
    monkeypatch.setenv("GLM53_KV_STORE_HOLD_OVER", "excess")
    monkeypatch.setenv("GLM53_KV_STORE_FT_GRACE", "1")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER"):
        monkeypatch.delenv(k, raising=False)
    host, pages = _host(400)
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256), ("c3", 256, 384)])
    assert pages == ["c1"]
    host.save(*_pre("c4", 384, 400))
    host.begin_step(NXT)  # 3 held: the one above HOLD_MAX, nothing else (grace)
    assert pages == ["c1", "c2"]
    host.begin_step(NXT)
    assert pages == ["c1", "c2", "c3"]


def _multi_host(lens):
    """a host whose steps carry rows of several requests: mi.rows = [(req_id, start, end)]"""
    host, pages = _host(0)
    host._tokens_of = lambda rid: list(range(lens[rid]))
    host._prefill_rows = lambda mi: ([(i, rid, s0, e) for i, (rid, s0, e) in enumerate(mi.rows)]
                                     if mi.prompt_lens is not None else [])  # fmt: skip
    return host, pages


def _step(tag, *rows, finished=()):
    so = SimpleNamespace(finished_req_ids=set(finished))
    mi = SimpleNamespace(prompt_lens=[e for _r, _s, e in rows], row_req_ids=[r for r, _s, _e in rows], rows=list(rows),
                         tag=tag)  # fmt: skip
    return so, mi


def test_grace_with_concurrent_requests(monkeypatch):
    """A's last chunk runs in step N while B prefills and C decodes around it: nothing flushes on A's first-token step
    (B's prefill save and C's decode save included); B's later chunks hold everything (B has no first token yet); after
    B's last chunk the grace repeats; then one held step per engine step, oldest first, in step order"""
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_FT_GRACE", "1")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER"):
        monkeypatch.delenv(k, raising=False)
    host, pages = _multi_host({"A": 300, "B": 260})
    nxt = SimpleNamespace(finished_req_ids=set())
    for tag, s0, e in (("a1", 0, 128), ("a2", 128, 256)):
        host.save(*_step(tag, ("A", s0, e)))
        host.begin_step(nxt)
    host.save(*_step("a3", ("A", 256, 300)))  # step N: A's last chunk
    host.begin_step(nxt)  # N + 1 begins: A's first token is still in the batch queue
    host.save(nxt, SimpleNamespace(prompt_lens=None, row_req_ids=["C"], tag="c-dec"))  # C decodes in N + 1
    assert pages == ["c-dec"]
    host.save(*_step("b1", ("B", 0, 128)))  # B prefills in N + 1 too: deferred, nothing of A's flushes
    assert pages == ["c-dec"]
    host.begin_step(nxt)  # N + 2: b1 is the last held step and B is mid-prefill -> everything stays held
    assert pages == ["c-dec"]
    host.save(*_step("b2", ("B", 128, 260)))  # B's last chunk (step M)
    host.begin_step(nxt)  # M + 1: B's first token in the batch queue -> grace again
    assert pages == ["c-dec"]
    host.begin_step(nxt)  # M + 2: one held step per engine step, step order
    assert pages == ["c-dec", "a1"]
    host.save(nxt, SimpleNamespace(prompt_lens=None, row_req_ids=["C"], tag="c-dec2"))
    assert pages[-2:] == ["a2", "c-dec2"]
    host.begin_step(SimpleNamespace(finished_req_ids={"B"}))  # B leaves: every held step at once
    assert pages[-3:] == ["a3", "b1", "b2"] and not host._held_pages


def test_grace_on_a_mixed_step_ending_one_prompt_while_another_starts(monkeypatch):
    """one batched prefill step carries A's last chunk and B's first: A's first token is in the batch queue (grace) and
    B is mid-prefill (hold) -> nothing flushes until B's last chunk's step has been popped too"""
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_FT_GRACE", "1")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER"):
        monkeypatch.delenv(k, raising=False)
    host, pages = _multi_host({"A": 200, "B": 300})
    nxt = SimpleNamespace(finished_req_ids=set())
    host.save(*_step("a1", ("A", 0, 128)))
    host.begin_step(nxt)
    host.save(*_step("ab", ("A", 128, 200), ("B", 0, 128)))  # mixed
    host.begin_step(nxt)
    assert pages == []
    host.begin_step(nxt)
    assert pages == []  # B still mid-prefill
    host.save(*_step("b2", ("B", 128, 300)))
    host.begin_step(nxt)
    assert pages == []  # B's first token in the batch queue
    host.begin_step(nxt)
    assert pages == ["a1"]


# ---------------------------------------------------------------- F2c (2026-10-25): the empty step and the full worker
class _Worker:
    """a store worker stub: has_room() as set, drain() counted"""

    def __init__(self, room=True):
        self.room, self.drains, self.inflight = room, 0, 0

    def busy(self):
        return self.inflight > 0

    def has_room(self):
        return self.room

    def drain(self):
        self.drains += 1


def _nofwd_host(monkeypatch, prompt_len):
    host, pages = _host(prompt_len)
    host._step_context = lambda mi: (None, None)
    host._collect = lambda so: None
    host.connector.start_load_kv = lambda fc: None
    host.store_worker = _Worker()
    monkeypatch.setattr(K, "ModelRunnerOutput", SimpleNamespace(with_kv_conn_output_only=lambda o: o))
    return host, pages


def _f2c_env(monkeypatch, **extra):
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_FT_GRACE", "1")
    for k in ("GLM53_KV_STORE_PRIORITY", "GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER",
              "GLM53_KV_NOFWD_FLUSH", "GLM53_KV_STORE_NONBLOCK"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


@pytest.mark.parametrize("mode", ["budget", "all"])
def test_the_empty_step_after_the_last_chunk(monkeypatch, mode):
    """the runner step right after a prompt's last chunk has no model input and calls no_forward (the cf3h-g run's
    engine stall stack, 0.44 s after the last prefill). 'all' (the default, cf3h-g): every held page store flushes and the worker drains on that
    step -- the stored 1M needle's 51.4 s before its first token. 'budget' (F2c): the hold / grace rules apply (nothing
    flushes while the first token is in the queue) and nothing drains."""
    _f2c_env(monkeypatch, GLM53_KV_NOFWD_FLUSH=mode)
    host, pages = _nofwd_host(monkeypatch, 400)
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256), ("c3", 256, 384)])
    host.save(*_pre("c4", 384, 400))  # the last chunk
    host.begin_step(NXT)  # the empty step N + 1: its begin_step (grace) ...
    host.no_forward(NXT)  # ... then no_forward
    if mode == "budget":
        assert pages == [] and host.store_worker.drains == 0
        host.begin_step(NXT)  # step N + 2: one held step per engine step, oldest first
        assert pages == ["c1"]
    else:
        assert pages == ["c1", "c2", "c3", "c4"] and host.store_worker.drains == 1


def test_the_budgeted_flush_never_blocks_on_a_full_worker(monkeypatch):
    """GLM53_KV_STORE_NONBLOCK=1: a budgeted flush waits for a later step while the worker is at its in-flight cap;
    a held request leaving still flushes everything (its blocks are reused next)"""
    _f2c_env(monkeypatch, GLM53_KV_NOFWD_FLUSH="budget", GLM53_KV_STORE_NONBLOCK="1")
    host, pages = _nofwd_host(monkeypatch, 400)
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256), ("c3", 256, 384)])
    host.save(*_pre("c4", 384, 400))
    host.begin_step(NXT)  # grace
    host.store_worker.room = False
    host.begin_step(NXT)  # full: nothing, no blocking submit
    assert pages == []
    host.store_worker.room = True
    host.begin_step(NXT)
    assert pages == ["c1"]
    host.store_worker.room = False
    host.begin_step(SimpleNamespace(finished_req_ids={"r"}))  # the held request leaves: everything, full or not
    assert pages == ["c1", "c2", "c3", "c4"]


def test_nonblock_is_off_by_default(monkeypatch):
    _f2c_env(monkeypatch)
    host, pages = _nofwd_host(monkeypatch, 400)
    host.store_worker.room = False
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256), ("c3", 256, 384)])
    host.save(*_pre("c4", 384, 400))
    host.begin_step(NXT)  # grace
    host.begin_step(NXT)  # off: the budget flushes one held step even with the worker full (submit would block)
    assert pages == ["c1"]


def test_a_finished_requests_queued_stores_keep_the_host_registered_for_the_next_lookup(monkeypatch):
    """the cf3h-i hit-depth mock (2026-10-25 15:32Z): with F2c the held steps trickle out after the first token; the
    prompt finishes, the 'leaving' begin_step hands every remaining held step to the store worker WITHOUT a drain, and
    the host used to leave _SAVE_HOSTS right there -- the repeat's lookup then skipped the drain and saw LMCache short
    (cached 0, a provenance restore of the stored chunks). The host now stays registered while its stores are in
    flight, so the lookup drains them first."""
    _f2c_env(monkeypatch, GLM53_KV_NOFWD_FLUSH="budget", GLM53_KV_LOOKUP_FLUSH="prefix")
    host, pages = _nofwd_host(monkeypatch, 400)
    worker = host.store_worker
    host._head_of_fn = lambda rid, n: None  # the finished request's history is gone from the runner: unknown head
    host._lmcache_chunk = lambda: 128
    _chunks(host, [("c1", 0, 128), ("c2", 128, 256), ("c3", 256, 384)])
    host.save(*_pre("c4", 384, 400))
    host.begin_step(NXT)  # grace
    host.no_forward(NXT)  # F2c: nothing
    worker.inflight = 3  # the leaving flush below hands three steps to the worker (still in flight after it); the
    host.__dict__["_queued_heads"] = {None: 3}  # real submit records their heads (None: the request is gone)
    host.begin_step(SimpleNamespace(finished_req_ids={"r"}))  # the prompt finished: everything to the worker, no drain
    assert pages == ["c1", "c2", "c3", "c4"] and host.store_worker.drains == 0
    assert host in K._SAVE_HOSTS  # still registered: its stores are in flight
    K.flush_deferred_saves(SimpleNamespace(prompt_token_ids=list(range(400))))  # the repeat's lookup
    assert host.store_worker.drains == 1  # ... drains them before LMCache is read
    worker.inflight = 0
    host.__dict__["_queued_heads"] = {}  # the jobs finished (a prefix lookup now skips the host: nothing of it pending)
    host.begin_step(NXT)  # the next engine step's flush finds nothing held and nothing in flight
    assert host not in K._SAVE_HOSTS  # landed: the host leaves the registry
