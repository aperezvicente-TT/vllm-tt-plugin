# SPDX-License-Identifier: Apache-2.0
"""Stage 12 (program 2026-10-26T17:09:24Z, a k3 MUST): a request whose prefill cannot PLAN on the model is a
REQUEST error, not a latched device fault. The model runner registers the model's host-only plan check
(vllm_tt_plugin.plan_check); the scheduler finishes a running request whose remaining prefill does not plan (finish
reason abort, stop_reason 'plan: ...', its client notified) while the others keep stepping; the connector steps a prefix
hit down to a checkpoint whose continuation plans. Fakes only (no engine, no device), as test_request_timeout.py."""

from types import SimpleNamespace

import pytest
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from vllm_tt_plugin import plan_check
from vllm_tt_plugin.kv_connector import TTPageAlignedConnector
from vllm_tt_plugin.scheduler import TTScheduler

CACHE = 66048


def _check(start, n):
    """the model's cache-end rule in miniature: the last chunk's 2,048 bucket must fit"""
    last = start + ((n - 1) // 2048) * 2048
    return None if last + 2048 <= CACHE else f"chunk [{last}, {last + 2048}) past the cache ({CACHE} tokens)"


@pytest.fixture(autouse=True)
def _registered():
    plan_check.register(_check)
    yield
    plan_check.register(None)


class _Req:
    def __init__(self, rid, computed, prompt, client=0, status=RequestStatus.RUNNING):
        self.request_id, self.num_computed_tokens, self.num_prompt_tokens = rid, computed, prompt
        self.client_index, self.status, self.stop_reason, self.trace_headers = client, status, None, None
        self.prompt_token_ids = list(range(prompt))

    def is_finished(self):
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self):
        return RequestStatus.get_finished_reason(self.status)

    def take_events(self):
        return None


def _sched(running):
    finished = []

    def finish_requests(ids, status):
        out = [r for r in running if r.request_id in set(ids) and not r.is_finished()]
        for r in out:
            r.status = status
        finished.append((sorted(ids), status))
        return out

    return SimpleNamespace(running=list(running), waiting=[], finish_requests=finish_requests, _finished=finished)


def test_a_waiting_request_resuming_past_the_cache_is_refused_too():
    w = _Req("w", 64256, 65256, status=RequestStatus.WAITING)
    sched = _sched([w])
    sched.running, sched.waiting = [], [w]
    TTScheduler._refuse_unplannable(sched)
    assert sched._finished == [(["w"], RequestStatus.FINISHED_ABORTED)]


def test_the_registry_is_advisory():
    assert plan_check.plan_error(0, 100) is None
    assert "past the cache" in plan_check.plan_error(64256, 1000)
    plan_check.register(lambda s, n: 1 / 0)  # a broken check never refuses
    assert plan_check.plan_error(64256, 1000) is None
    plan_check.register(None)
    assert plan_check.plan_error(64256, 1000) is None


def test_an_unplannable_running_request_is_refused_and_the_rest_keep_stepping():
    bad, good, done = _Req("bad", 64256, 65256, client=1), _Req("good", 2048, 9000), _Req("decoding", 500, 500)
    sched = _sched([bad, good, done])
    TTScheduler._refuse_unplannable(sched)
    assert sched._finished == [(["bad"], RequestStatus.FINISHED_ABORTED)]
    assert good.status == done.status == RequestStatus.RUNNING
    assert bad.stop_reason.startswith("plan: chunk [64256, 66304) past the cache")
    outputs = {0: EngineCoreOutputs()}
    TTScheduler._emit_refused(sched, outputs)
    rows = [(c, o.request_id, o.finish_reason, o.new_token_ids) for c, v in outputs.items() for o in v.outputs]
    assert rows == [(1, "bad", FinishReason.ABORT, [])]
    TTScheduler._emit_refused(sched, outputs)  # emitted once
    TTScheduler._refuse_unplannable(sched)  # nothing new to refuse
    assert len(sched._finished) == 1


def test_no_registered_check_refuses_nothing():
    plan_check.register(None)
    bad = _Req("bad", 64256, 65256)
    sched = _sched([bad])
    TTScheduler._refuse_unplannable(sched)
    assert sched._finished == [] and bad.status == RequestStatus.RUNNING


def test_a_hit_whose_continuation_does_not_plan_steps_down_to_one_that_does():
    conn = TTPageAlignedConnector.__new__(TTPageAlignedConnector)
    conn._inner = SimpleNamespace()  # (its __getattr__ delegates to the inner connector)
    conn.aligned_hit = lambda toks, total: (min(int(total), len(toks)) // 128) * 128  # every 128 checkpointed
    req = _Req("r", 0, 65300)
    # a hit at 64,256: [64256, 65300) ends past the cache in its 2,048 bucket -> steps down to 64,000 (64,000 + 2,048)
    assert TTPageAlignedConnector._plannable_hit(conn, req, 0, 64256) == 64000
    assert conn.hits_replanned == 2
    assert TTPageAlignedConnector._plannable_hit(conn, req, 0, 4096) == 4096  # a plannable hit stays


def test_a_device_prefix_hit_whose_continuation_does_not_plan_steps_down():
    """M2-1 (program 17:23:42Z): the DEVICE prefix cache's hit (the oracle-truncated local page hit) gets the same
    plan check as an LMCache hit -- a continuation near the cap re-plans from a lower snapshotted boundary"""

    class _Oracle:
        boundary = 128

        def longest_cached(self, ids, max_tokens=None):  # every 128 snapshotted
            return (int(max_tokens) // 128) * 128

    sched = SimpleNamespace(__dict__={})
    req = _Req("r", 0, 65300)
    keep = TTScheduler._plannable_prefix_hit(sched, req, _Oracle(), req.prompt_token_ids, 64256, 64)
    assert keep == 64000 and sched._prefix_hits_replanned == 2
    assert TTScheduler._plannable_prefix_hit(sched, req, _Oracle(), req.prompt_token_ids, 4096, 64) == 4096
