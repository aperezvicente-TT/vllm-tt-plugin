"""Segmented prefill traces (doc/ttnn/14b-persistent-stack/segtrace/runner_design.md section 4, item 12): the connector
host parks a segmented chunk's save at its first step and runs it at the release step.

The first step of a segmented chunk (set_tt_segmented {req_id: S}) runs only segment 0 of the chunk's layers, so the
chunk's DSA pages past it and its KDA state are not written yet. Its page store, KDA checkpoint and in-replay snapshot
wait for the request's next prefill row (the 1-token tail, segment S - 1) and run right before that step's own save. A
request that leaves inside the hold drops it, and its preempt save is skipped."""

import pickle
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import kv_connector as K
from vllm_tt_plugin.scheduler import get_tt_segmented, set_tt_segmented


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(K, "_SAVE_HOSTS", set())
    for k in ("GLM53_KV_SAVE_DEFER", "GLM53_KV_STORE_HOLD", "GLM53_KV_STORE_PRIORITY", "GLM53_KV_STORE_FT_GRACE",
              "GLM53_KV_STORE_HOLD_MAX", "GLM53_KV_STORE_HOLD_OVER", "GLM53_PREEMPT_SAVE"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)


def _host():
    """the host's save path with the device / LMCache ends stubbed: every KDA export and page store is logged"""
    host = K.TTKVConnectorHost.__new__(K.TTKVConnectorHost)
    host._deferred_save, host.saves_deferred = None, 0
    log = []
    host._save_state = lambda mi, out=None: log.append(("state", mi.tag))
    host._save_pages = lambda so, mi: log.append(("pages", mi.tag))
    host._tokens_of = lambda rid: list(range(4096))
    host._prefill_rows = lambda mi: [(i, rid, s0, e) for i, (rid, s0, e) in enumerate(mi.rows)] if mi.rows else []
    host.connector = SimpleNamespace(bind_connector_metadata=lambda m: None, clear_connector_metadata=lambda: None,
                                     handle_preemptions=lambda m: None)  # fmt: skip
    host._metadata, host._lock = (lambda so: None), threading.Lock()
    return host, log


def _so(segmented=None, finished=(), preempted=()):
    so = SimpleNamespace(finished_req_ids=set(finished), preempted_req_ids=set(preempted))
    if segmented:
        set_tt_segmented(so, segmented)
    return so


def _pre(tag, *rows):
    return SimpleNamespace(prompt_lens=[e for _r, _s, e in rows], row_req_ids=[r for r, _s, _e in rows], rows=rows,
                           tag=tag)  # fmt: skip


def _dec(tag):
    return SimpleNamespace(prompt_lens=None, row_req_ids=["o"], rows=(), tag=tag)


def _s3_chunk(host, s0=0, e=2048):
    """S = 3: step 0 (the chunk minus its last token, the runner runs [s0, e) segment 0), step 1 (decode + segment 1),
    step 2 (the release: the 1-token tail [e - 1, e), segment 2 + the head)"""
    host.save(_so({"r": 3}), _pre("c0", ("r", s0, e)))
    host.begin_step(_so())
    host.save(_so(), _dec("d1"))
    host.begin_step(_so())
    host.save(_so(), _pre("rel", ("r", e - 1, e)))


def test_stamp_round_trips_through_pickle():
    so = _so({"r": 3, 7: 2})
    assert get_tt_segmented(pickle.loads(pickle.dumps(so))) == {"r": 3, "7": 2}
    assert get_tt_segmented(_so()) == {}


@pytest.mark.parametrize("defer", ["1", "0"])
def test_the_chunk_saves_at_the_release_in_step_order(monkeypatch, defer):
    monkeypatch.setenv("GLM53_KV_SAVE_DEFER", defer)
    host, log = _host()
    host.save(_so({"r": 3}), _pre("c0", ("r", 0, 2048)))
    host.begin_step(_so())
    assert ("state", "c0") not in log and ("pages", "c0") not in log  # segment 0 only: nothing of the chunk yet
    host.save(_so(), _dec("d1"))
    host.begin_step(_so())
    assert [t for _k, t in log] == ["d1", "d1"]  # the decode step saved itself only
    host.save(_so(), _pre("rel", ("r", 2047, 2048)))
    if defer == "1":
        assert ("state", "c0") not in log  # the release's save is deferred too: the chunk with it
        host.begin_step(_so())
    K.flush_deferred_saves()
    chunk = [e for e in log if e[1] in ("c0", "rel")]
    assert [e[1] for e in chunk if e[0] == "state"] == ["c0", "rel"]
    assert [e[1] for e in chunk if e[0] == "pages"] == ["c0", "rel"]
    assert host.segment_saves_parked == 1 and host.segment_saves_released == 1
    assert not host._segment_parked


def test_an_unstamped_or_single_segment_chunk_saves_as_today():
    host, log = _host()
    host.save(_so({"r": 1}), _pre("c0", ("r", 0, 2048)))
    host.begin_step(_so())
    assert ("state", "c0") in log
    host.save(_so({"x": 3}), _pre("c1", ("r", 2048, 4096)))  # another request's stamp
    host.begin_step(_so())
    assert ("state", "c1") in log and not getattr(host, "segment_saves_parked", 0)


@pytest.mark.parametrize("how", ["preempted", "finished"])
def test_a_request_leaving_inside_the_hold_drops_its_chunk(how):
    host, log = _host()
    host.save(_so({"r": 3}), _pre("c0", ("r", 0, 2048)))
    host.begin_step(_so(**{how: {"r"}}))
    assert host.segment_saves_dropped == 1 and not host._segment_parked
    host.save(_so(), _pre("again", ("r", 0, 2048)))  # its recompute later: itself only
    host.begin_step(_so())
    K.flush_deferred_saves()
    assert all(t != "c0" for _k, t in log) and ("state", "again") in log


def test_the_release_then_finish_keeps_the_chunk():
    """a final chunk with max_tokens 1: the request finishes in the step after its release -- the release's flush in
    that begin_step saves the chunk before the drop looks"""
    host, log = _host()
    _s3_chunk(host)
    host.begin_step(_so(finished={"r"}))
    assert ("state", "c0") in log and ("pages", "c0") in log
    assert not getattr(host, "segment_saves_dropped", 0)


def test_a_preempt_save_inside_the_hold_is_skipped(monkeypatch):
    monkeypatch.setenv("GLM53_PREEMPT_SAVE", "1")
    monkeypatch.setattr(K, "_PREEMPT_SAVES", {"r": {"n": 2047}})
    monkeypatch.setattr(K, "_PREEMPT_RESUME", {"r": None})
    host, _log = _host()
    host._preempt_store_one = lambda so, rid, rec: pytest.fail("a mid-hold request has no consistent [0, n) to save")
    host.save(_so({"r": 3}), _pre("c0", ("r", 0, 2048)))
    host.begin_step(_so(preempted={"r"}))
    assert host.preempt_saves_segment_held == 1 and "r" not in K._PREEMPT_RESUME


def test_a_stamped_step_with_another_prefill_row_is_refused():
    host, _log = _host()
    with pytest.raises(RuntimeError, match="only prefill row"):
        host.save(_so({"r": 3}), _pre("c0", ("r", 0, 2047), ("q", 0, 300)))


def test_a_second_first_step_before_the_release_is_refused():
    host, _log = _host()
    host.save(_so({"r": 3}), _pre("c0", ("r", 0, 2048)))
    host.begin_step(_so())
    with pytest.raises(RuntimeError, match="already parked"):
        host.save(_so({"r": 3}), _pre("c1", ("r", 2048, 4096)))


def test_back_to_back_segmented_chunks_each_save_once(monkeypatch):
    host, log = _host()
    _s3_chunk(host, 0, 2048)
    host.begin_step(_so())  # the next chunk's step 0 follows the release at once
    host.save(_so({"r": 3}), _pre("c1", ("r", 2048, 4096)))
    assert ("state", "c0") in log and ("state", "c1") not in log
    host.begin_step(_so())
    host.save(_so(), _pre("rel1", ("r", 4095, 4096)))
    host.begin_step(_so())
    K.flush_deferred_saves()
    states = [t for k, t in log if k == "state"]
    assert states == ["d1", "c0", "rel", "c1", "rel1"]  # the hold's decode step saves before the chunk


def test_shutdown_drops_an_unreleased_chunk():
    host, log = _host()
    host.save(_so({"r": 3}), _pre("c0", ("r", 0, 2048)))
    host.connector.shutdown = lambda: None
    host.shutdown()
    assert host.segment_saves_dropped == 1 and not log


def test_the_counters_line_carries_the_segment_counters():
    for k in ("segment_saves_parked", "segment_saves_released", "segment_saves_dropped", "segment_slots_moved",
              "preempt_saves_segment_held"):
        assert k in K._COUNTER_KEYS["host"]


@dataclass(frozen=True)
class _MI:
    """a frozen model input like TTModelInput: the slot refresh goes through dataclasses.replace"""

    tag: str
    rows: tuple
    prefill_empty_slots: list
    prompt_lens: list = field(default_factory=lambda: [0])

    @property
    def row_req_ids(self):
        return [r for r, _s, _e in self.rows]


@pytest.mark.parametrize("moved", [False, True])
def test_the_parked_save_reads_the_release_steps_slot(moved):
    host, log = _host()
    seen = []
    host._save_state = lambda mi, out=None: seen.append((mi.tag, list(mi.prefill_empty_slots)))
    host.save(_so({"r": 3}), _MI("c0", (("r", 0, 2048),), [5]))
    host.begin_step(_so())
    host.save(_so(), _MI("rel", (("r", 2047, 2048),), [6 if moved else 5]))
    host.begin_step(_so())
    want = 6 if moved else 5
    assert seen == [("c0", [want]), ("rel", [want])]
    assert getattr(host, "segment_slots_moved", 0) == int(moved)
