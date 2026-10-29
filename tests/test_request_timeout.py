# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request wall-clock timeouts: config parsing and the scheduler-step expiry.

The expiry runs on a fake TTScheduler (no engine, no device): requests with an
``arrival_time``, a driven clock, and a recording ``finish_requests``.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.config import get_tt_request_timeout_s
from vllm_tt_plugin.scheduler import TTScheduler


def _cfg(tt):
    return SimpleNamespace(additional_config={"tt": tt})


def test_request_timeout_config_default_and_validation():
    assert get_tt_request_timeout_s(_cfg({})) == 0.0
    assert get_tt_request_timeout_s(_cfg({"request_timeout_s": 30})) == 30.0
    assert get_tt_request_timeout_s(_cfg({"request_timeout_s": 2.5})) == 2.5
    for bad in (-1, True, "30"):
        with pytest.raises(ValueError, match="request_timeout_s"):
            get_tt_request_timeout_s(_cfg({"request_timeout_s": bad}))


class _Req:
    def __init__(self, rid, arrival, client_index=0, status=RequestStatus.RUNNING):
        self.request_id = rid
        self.arrival_time = arrival
        self.client_index = client_index
        self.status = status
        self.stop_reason = None
        self.num_output_tokens = 3
        self.trace_headers = None

    def is_finished(self):
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self):
        return RequestStatus.get_finished_reason(self.status)

    def take_events(self):
        return None


def _fake_scheduler(timeout, now, running, waiting=()):
    finished = []

    def finish_requests(ids, status):
        out = []
        for r in list(running) + list(waiting):
            if r.request_id in set(ids) and not r.is_finished():
                r.status = status
                out.append(r)
        finished.append((sorted(ids), status))
        return out

    return SimpleNamespace(
        _request_timeout_s=timeout,
        _tt_clock=lambda: now,
        running=list(running),
        waiting=list(waiting),
        finish_requests=finish_requests,
        _finished=finished,
    )


def test_expiry_finishes_only_over_budget_requests_and_notifies_their_client():
    fresh, old, queued = (
        _Req("fresh", 95.0),
        _Req("old", 60.0, 1),
        _Req("q", 50.0, 0, RequestStatus.WAITING),
    )
    sched = _fake_scheduler(30.0, 100.0, [fresh, old], [queued])
    outputs = {0: EngineCoreOutputs()}  # client 0 already has an output frame this step
    TTScheduler._expire_timed_out_requests(sched, outputs)
    assert sched._finished == [(["old", "q"], RequestStatus.FINISHED_ABORTED)]
    assert fresh.status == RequestStatus.RUNNING
    assert old.status == RequestStatus.FINISHED_ABORTED and old.stop_reason == "timeout"
    # one finish row per expired request, routed to its own client, no tokens
    by_client = {
        c: [
            (o.request_id, o.finish_reason, o.stop_reason, o.new_token_ids)
            for o in v.outputs
        ]
        for c, v in outputs.items()
    }
    assert by_client == {
        0: [("q", FinishReason.ABORT, "timeout", [])],
        1: [("old", FinishReason.ABORT, "timeout", [])],
    }


def test_expiry_is_a_no_op_at_or_under_budget():
    r = _Req("r", 70.0)
    sched = _fake_scheduler(30.0, 100.0, [r])
    outputs = {}
    TTScheduler._expire_timed_out_requests(sched, outputs)
    assert sched._finished == [] and outputs == {} and r.status == RequestStatus.RUNNING
