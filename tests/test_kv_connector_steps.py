# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""KV connector steps, host only: decode-join on vLLM's scheduler and the runner's
connector lifecycle around a step.

A request whose KV a connector receives waits in ``kv_holding_waiting`` as
WAITING_FOR_REMOTE_KVS (it holds the blocks being written) until the connector
reports it in a step output's ``finished_recving``.
"""

from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.request import RequestStatus

from tests.test_block_scheduler import _request, _runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler
from vllm_tt_plugin import kv_transfer
from vllm_tt_plugin.scheduler import TTSchedulingMode


class _Meta:
    def __init__(self):
        self.reqs_to_recv = {}


class _SchedulerConnector:
    """The scheduler-side connector surface the base scheduler calls in these steps."""

    supports_divergent_local_hybrid_hits = False

    def on_new_request(self, request):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def has_pending_block_frees(self):
        return False

    def build_connector_meta(self, scheduler_output):
        return _Meta()

    def update_connector_output(self, kv_connector_output):
        pass

    def get_kv_connector_stats(self):
        return None

    def take_events(self):
        return ()


def _scheduler_with_running_decode(max_num_seqs=2, async_scheduling=False):
    """A scheduler with a connector, whose request ``A`` is decoding."""
    scheduler = _real_scheduler(
        output_width=1, max_num_seqs=max_num_seqs, async_scheduling=async_scheduling
    )
    scheduler.add_request(_request(4, request_id="A"))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, [7]))
    scheduler.connector = _SchedulerConnector()
    return scheduler


def _received(scheduler, request_id, *, arrived=True, computed=0, **sampling):
    """A request whose whole prompt KV is (or is being) received from a prefill node.

    ``computed``: tokens short of the whole prompt the prefill node computed.
    """
    request = _request(4, request_id=request_id)
    for name, value in sampling.items():
        setattr(request.sampling_params, name, value)
    scheduler.add_request(request)
    scheduler.waiting.remove_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = request.num_tokens - computed
    scheduler.kv_holding_waiting.add_request(request)
    if arrived:
        scheduler.finished_recving_kv_req_ids.add(request_id)
    return request


def _step_output(scheduler_output, tokens, kv_connector_output=None):
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[tokens[r]] for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_connector_output,
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_a_received_request_joins_the_decode_step(async_scheduling):
    scheduler = _scheduler_with_running_decode(async_scheduling=async_scheduling)
    _received(scheduler, "B")

    assert not scheduler._has_pending_prefill()
    step = scheduler.schedule()

    # B's last prompt token runs as a decode row next to A.
    assert step.num_scheduled_tokens == {"A": 1, "B": 1}
    assert [r.req_id for r in step.scheduled_new_reqs] == ["B"]
    assert step.scheduled_new_reqs[0].num_computed_tokens == 31
    assert not scheduler.kv_holding_waiting


def test_a_request_still_receiving_is_not_prefill_work():
    scheduler = _scheduler_with_running_decode()
    held = _received(scheduler, "B", arrived=False)

    assert not scheduler._has_pending_prefill()
    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"A": 1}
    assert list(scheduler.kv_holding_waiting) == [held]


@pytest.mark.parametrize(
    "sampling",
    [{"seed": 5}, {"prompt_logprobs": 1}, {"repetition_penalty": 1.1}],
)
def test_a_received_request_that_needs_its_prefill_keeps_it(sampling):
    # The prefill program initialises the request's device sampling state (or
    # returns prompt logprobs), so the request does not join decode.
    scheduler = _scheduler_with_running_decode()
    _received(scheduler, "B", **sampling)

    assert scheduler._has_pending_prefill()
    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"B": 1}


def test_a_prefill_step_leaves_joining_requests_to_decode():
    scheduler = _scheduler_with_running_decode(max_num_seqs=3)
    joining = _received(scheduler, "B")
    scheduler.add_request(_request(4, request_id="C"))

    prefill = scheduler.schedule()

    assert set(prefill.num_scheduled_tokens) == {"C"}
    assert list(scheduler.kv_holding_waiting) == [joining]

    scheduler.set_forced_mode(TTSchedulingMode.DECODE_ONLY)
    scheduler.update_from_output(prefill, _runner_output(prefill, [9]))
    decode = scheduler.schedule()

    assert decode.num_scheduled_tokens.get("B") == 1
    assert not scheduler.kv_holding_waiting


def test_join_eligibility():
    scheduler = _scheduler_with_running_decode()
    ok = _received(scheduler, "B")
    assert scheduler._joins_decode(ok)

    assert not scheduler._joins_decode(_received(scheduler, "C", arrived=False))
    failed = _received(scheduler, "D")
    scheduler.failed_recving_kv_req_ids.add("D")
    assert not scheduler._joins_decode(failed)

    # State models: the prefill node computes all but the last prompt token.
    assert scheduler._joins_decode(_received(scheduler, "E", computed=1))
    # Anything earlier is prompt work for a prefill step.
    assert not scheduler._joins_decode(_received(scheduler, "F", computed=2))

    mm = _received(scheduler, "G")
    mm.mm_features = [object()]
    assert not scheduler._joins_decode(mm)

    # A request that is not waiting for remote KV (a preempted one in the same
    # queue) never joins.
    ok.status = RequestStatus.PREEMPTED
    assert not scheduler._joins_decode(ok)


def test_a_replay_window_keeps_received_requests_on_the_prefill_path():
    """vLLM rewinds a received request by the replay window of its SWA groups, so
    it resumes with more than one token left."""
    scheduler = _scheduler_with_running_decode()
    received = _received(scheduler, "B")
    scheduler.prefix_replay_tokens = 8

    assert not scheduler._joins_decode(received)
    assert scheduler._has_pending_prefill()


def test_recurrent_state_joins_only_when_the_model_imports_it(monkeypatch):
    scheduler = _scheduler_with_running_decode()
    received = _received(scheduler, "B", computed=1)
    monkeypatch.setattr(scheduler, "_has_recurrent_state", lambda: True)

    scheduler._imports_state = False
    assert not scheduler._joins_decode(received)
    scheduler._imports_state = True
    assert scheduler._joins_decode(received)


class _LoadOp:
    """An asynchronous copy op whose loads complete when the test says so."""

    def __init__(self):
        self.ticket, self.done = None, set()

    def take_load_ticket(self):
        ticket, self.ticket = self.ticket, None
        return ticket

    def load_done(self, ticket):
        return ticket in self.done


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_a_request_is_scheduled_only_after_its_load_completes(
    monkeypatch, async_scheduling
):
    """The connector reports B received while the copy op's host -> device load is
    still running. The gate keeps B out of the step output until the load is done,
    so no step (also under async scheduling, where the scheduler runs a step ahead
    of the outputs it has seen) schedules B before its KV is on the device."""
    op = _LoadOp()
    monkeypatch.setattr(kv_transfer, "_copy_op", op)
    monkeypatch.setattr(kv_transfer, "_held_recvs", [])
    scheduler = _scheduler_with_running_decode(async_scheduling=async_scheduling)
    receiving = _received(scheduler, "B", arrived=False)

    # Step 1: the connector reports B; its load (ticket 1) is still running.
    step1 = scheduler.schedule()
    kv_out = KVConnectorOutput(finished_recving={"B"})
    op.ticket = 1
    kv_transfer.gate_received(kv_out)
    assert kv_out.finished_recving is None
    scheduler.update_from_output(step1, _step_output(step1, {"A": 8}, kv_out))

    step2 = scheduler.schedule()
    assert "B" not in step2.num_scheduled_tokens
    assert receiving.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Step 2's output is built after the load completed: B is released then.
    op.done.add(1)
    kv_out = KVConnectorOutput()
    kv_transfer.release_ready(kv_out)
    assert kv_out.finished_recving == {"B"}
    scheduler.update_from_output(step2, _step_output(step2, {"A": 9}, kv_out))

    step3 = scheduler.schedule()
    assert step3.num_scheduled_tokens.get("B") == 1


# The runner's connector lifecycle around a step.


class _StepOp:
    """Asynchronous saves, completed only when waited; records the step hooks."""

    def __init__(self, events):
        self.events, self.issued, self.waits = events, 0, []

    def save(self):
        self.issued += 1

    def wait_saves(self):
        self.waits.append(self.issued)

    def begin_step(self):
        self.events.append("begin_step")

    def note_block_ids(self, block_ids):
        self.events.append(("note", block_ids))


def _lifecycle_runner(monkeypatch, body, *, op_cls=_StepOp):
    import vllm.forward_context
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorModelRunnerMixin,
    )

    from vllm_tt_plugin.model_runner import TTModelRunner

    events = []
    op = op_cls(events)
    monkeypatch.setattr(kv_transfer, "_copy_op", op)
    monkeypatch.setattr(kv_transfer, "_held_recvs", [])
    monkeypatch.setattr(kv_transfer, "_unwaited_saves", set())

    @contextmanager
    def lifecycle(scheduler_output):
        events.append("start_loads")
        # A request whose transfer finished: the output is not empty.
        yield KVConnectorOutput(finished_sending={"sent"})
        op.save()  # wait_for_save: the connector issues the step's save here
        events.append("save")

    monkeypatch.setattr(
        KVConnectorModelRunnerMixin,
        "_get_kv_connector_output",
        staticmethod(lifecycle),
    )
    monkeypatch.setattr(
        vllm.forward_context, "set_forward_context", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(
        kv_transfer, "handle_preemptions", lambda out: events.append("preempt")
    )
    monkeypatch.setattr(
        TTModelRunner, "_execute_model_body", lambda self, out: body(self, events)
    )
    runner = SimpleNamespace(vllm_config=None, _pending_samples=deque(), requests={})
    return TTModelRunner, runner, op, events


def _step(new_block_ids=()):
    return SimpleNamespace(
        kv_connector_metadata=None,
        scheduled_new_reqs=[
            SimpleNamespace(req_id=f"r{i}", num_computed_tokens=0, block_ids=ids)
            for i, ids in enumerate(new_block_ids)
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], num_computed_tokens=[], new_block_ids=None
        ),
        num_scheduled_tokens={f"r{i}": 1 for i in range(len(new_block_ids))},
        finished_req_ids=set(),
        preempted_req_ids=None,
    )


def _forward_body(self, events):
    events.append("forward")
    self._pending_samples.append(
        lambda grammar: ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0})
    )
    return None


def test_step_hooks_run_before_the_loads_and_the_forward(monkeypatch):
    cls, runner, op, events = _lifecycle_runner(monkeypatch, _forward_body)
    assert cls._execute_model_with_kv_connector(runner, _step([([3, 4],)])) is None
    out = runner._pending_samples.popleft()(None)

    # begin_step comes before the connector starts this step's loads, which the
    # copy op then orders after the work already enqueued, not after this forward.
    assert events == [
        "preempt",
        ("note", ([3, 4],)),
        "begin_step",
        "start_loads",
        "forward",
        "save",
    ]
    assert op.waits == [1]  # the save landed before the output was returned
    assert out.kv_connector_output.finished_sending == {"sent"}


def test_the_step_hooks_are_optional(monkeypatch):
    class _PlainOp:
        def __init__(self, events):
            self.events = events

        def save(self):
            pass

    cls, runner, _, events = _lifecycle_runner(
        monkeypatch, _forward_body, op_cls=_PlainOp
    )
    cls._execute_model_with_kv_connector(runner, _step([([3],)]))
    runner._pending_samples.popleft()(None)
    assert events == ["preempt", "start_loads", "forward", "save"]


def test_empty_step_still_reports_transfers(monkeypatch):
    cls, runner, op, _ = _lifecycle_runner(
        monkeypatch, lambda self, events: EMPTY_MODEL_RUNNER_OUTPUT
    )
    out = cls._execute_model_with_kv_connector(runner, _step())

    assert out is not EMPTY_MODEL_RUNNER_OUTPUT
    assert out.kv_connector_output.finished_sending == {"sent"}
    assert op.waits == [1]


def test_lane_data_parallelism_with_a_connector_is_refused(monkeypatch):
    from vllm_tt_plugin import model_runner

    cls, runner, _, _ = _lifecycle_runner(monkeypatch, lambda self, events: None)
    monkeypatch.setattr(model_runner, "get_tt_step_plan", lambda out: object())
    with pytest.raises(NotImplementedError, match="lane data parallelism"):
        cls._execute_model_with_kv_connector(runner, _step())


def test_note_block_ids_reads_every_multi_connector_child(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorMetadata,
    )

    noted = []
    monkeypatch.setattr(
        kv_transfer, "_copy_op", SimpleNamespace(note_block_ids=noted.append)
    )
    recv = SimpleNamespace(reqs_to_recv={"b": SimpleNamespace(local_block_ids=[[5]])})
    save = SimpleNamespace(reqs_to_save={"c": SimpleNamespace(local_block_ids=[[6]])})
    step = _step([([1],)])
    step.kv_connector_metadata = MultiKVConnectorMetadata(metadata=(recv, save))

    kv_transfer.note_block_ids(step)

    assert len(noted) == 3
    assert all(ids in noted for ids in (([1],), [[5]], [[6]]))
