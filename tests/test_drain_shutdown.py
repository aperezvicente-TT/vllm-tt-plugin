# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Drain-mode shutdown (vllm_tt_plugin.drain), host-only units: the engine-side drain budget aborts through the
engine's abort path between steps, the API side reports engine exit statuses, and the TT exit status tells an
orderly drain (0) from a dead engine (70)."""

from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

from vllm_tt_plugin import drain


def _cfg(timeout, tt=None):
    return SimpleNamespace(shutdown_timeout=timeout, additional_config={"tt": tt or {}})


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(drain.DRAIN_TIMEOUT_ENV, raising=False)
    drain._engine_exits.clear()
    yield
    drain._engine_exits.clear()


def test_drain_budget_leaves_the_kill_timer_a_margin(monkeypatch):
    assert drain.drain_budget_s(_cfg(0)) is None  # abort mode: upstream aborts at once
    assert drain.drain_budget_s(_cfg(900)) == 780.0  # 120 s margin
    assert drain.drain_budget_s(_cfg(30)) == 22.5  # 25 %
    assert drain.drain_budget_s(_cfg(8)) == 3.0  # 5 s floor
    assert drain.drain_budget_s(_cfg(900, {"drain_timeout_s": 60})) == 60.0
    assert drain.drain_budget_s(_cfg(900, {"drain_timeout_s": 5000})) == 780.0  # never past the margin
    monkeypatch.setenv(drain.DRAIN_TIMEOUT_ENV, "45")
    assert drain.drain_budget_s(_cfg(900, {"drain_timeout_s": 60})) == 45.0


def test_describe_exit():
    assert "orderly" in drain.describe_exit(0)
    assert "SIGBUS handler" in drain.describe_exit(int(signal.SIGBUS))
    assert "SIGKILL" in drain.describe_exit(-int(signal.SIGKILL))
    assert drain.describe_exit(None) == "still running"


class _Sched:
    def __init__(self, n):
        self.live = [f"r{i}" for i in range(n)]
        self.finish_calls = []

    def get_num_unfinished_requests(self):
        return len(self.live)

    def has_requests(self):
        return bool(self.live)

    def finish_requests(self, ids, status):
        self.finish_calls.append((ids, status))
        out, self.live = self.live, []
        return out


def _engine(core, timeout, n):
    eng = core.EngineCoreProc.__new__(core.EngineCoreProc)
    eng.vllm_config = _cfg(timeout)
    eng.scheduler = _Sched(n)
    eng.engines_running = False
    eng.batch_queue = None
    eng.sent_aborts = []
    eng._send_abort_outputs = eng.sent_aborts.append
    eng.shutdown_state = core.EngineShutdownState.REQUESTED
    return eng


def test_engine_drain_aborts_between_steps_at_the_budget(monkeypatch):
    import vllm.v1.engine.core as core
    from vllm.v1.request import RequestStatus

    drain.install_engine_drain_patch()
    drain.install_engine_drain_patch()  # idempotent
    assert core.EngineCoreProc._handle_shutdown._tt_drain_patch
    clock = [1000.0]
    monkeypatch.setattr(drain.time, "monotonic", lambda: clock[0])
    eng = _engine(core, 900, 2)
    assert eng._handle_shutdown() is True  # drain starts: keep stepping
    assert eng._tt_drain_budget == 780.0 and eng.scheduler.finish_calls == []
    clock[0] += 779.0
    assert eng._handle_shutdown() is True and eng.scheduler.finish_calls == []  # still draining
    clock[0] += 1.0
    assert eng._handle_shutdown() is True  # budget: abort now, loop runs once more for the abort outputs
    assert eng.scheduler.finish_calls == [(None, RequestStatus.FINISHED_ABORTED)]
    assert eng.sent_aborts == [["r0", "r1"]]
    assert eng._handle_shutdown() is False  # no work left: orderly exit -> engine_core.shutdown() closes the mesh
    assert len(eng.scheduler.finish_calls) == 1


def test_engine_drain_that_finishes_in_time_aborts_nothing(monkeypatch):
    import vllm.v1.engine.core as core

    drain.install_engine_drain_patch()
    monkeypatch.setattr(drain.time, "monotonic", lambda: 0.0)
    eng = _engine(core, 900, 1)
    assert eng._handle_shutdown() is True
    eng.scheduler.live = []  # the request finished by itself
    assert eng._handle_shutdown() is False
    assert eng.scheduler.finish_calls == [] and eng.sent_aborts == []


def test_abort_mode_is_upstream(monkeypatch):
    import vllm.v1.engine.core as core

    drain.install_engine_drain_patch()
    eng = _engine(core, 0, 2)
    assert eng._handle_shutdown() is False  # upstream aborted everything at once
    assert eng._tt_drain_budget is None and len(eng.scheduler.finish_calls) == 1


def test_process_manager_join_reports_engine_exit_status(monkeypatch):
    import vllm.v1.engine.utils as engine_utils

    lines = []

    class Rec:
        def info(self, msg, *a):
            lines.append(msg % a)

        error = info

    monkeypatch.setattr(drain, "logger", Rec())

    drain.install_api_drain_patches()
    drain.install_api_drain_patches()
    assert engine_utils.shutdown._tt_drain_patch

    class P:
        def __init__(self, pid, code):
            self.pid, self.exitcode, self.name, self.sentinel = pid, code, "EngineCore", None

        def is_alive(self):
            return False

    engine_utils.shutdown([P(11, 0)], timeout=1)
    assert not drain.abnormal_engine_exit()
    engine_utils.shutdown([P(12, int(signal.SIGBUS))], timeout=1)
    assert drain.abnormal_engine_exit() and drain.engine_exits() == {11: 0, 12: 7}
    text = "\n".join(lines)
    assert "EngineCore pid 11 exit 0 (orderly)" in text
    assert "EngineCore pid 12 exit status 7" in text and "mesh was NOT closed" in text


def test_exit_report_reads_the_mesh_close_marker(monkeypatch, tmp_path):
    """a non-zero exit after TTWorker closed the mesh (the fault-path close) is not reported as an open mesh"""
    import vllm.v1.engine.utils as engine_utils

    lines = []

    class Rec:
        def info(self, msg, *a):
            lines.append(msg % a)

        error = info

    monkeypatch.setattr(drain, "logger", Rec())
    monkeypatch.setenv("GLM53_TT_MESH_MARKER_DIR", str(tmp_path))
    drain.install_api_drain_patches()

    class P:
        def __init__(self, pid, code):
            self.pid, self.exitcode, self.name, self.sentinel = pid, code, "EngineCore", None

        def is_alive(self):
            return False

    open(drain.mesh_close_marker(31), "w").write("1\n")
    engine_utils.shutdown([P(31, -int(signal.SIGTERM)), P(32, -int(signal.SIGTERM))], timeout=1)
    text = "\n".join(lines)
    assert "pid 31" in text and "had closed its mesh before the exit" in text
    assert "pid 32" in text and "mesh was NOT closed by TTWorker" in text and "reset policy decide" in text


def test_tt_exit_status_is_not_errored_after_an_orderly_drain():
    """``errored`` is True after every AsyncLLM.shutdown; only a death report or an abnormal exit is a death."""
    from vllm_tt_plugin.platform import _tt_engine_died

    engine = SimpleNamespace(errored=True, engine_core=SimpleNamespace(resources=SimpleNamespace(engine_dead=False)))
    assert _tt_engine_died(engine) is False  # the orderly drain with an open stream: exit 0, not 70
    engine.engine_core.resources.engine_dead = True
    assert _tt_engine_died(engine) is True  # ENGINE_CORE_DEAD / liveness monitor
    engine.engine_core.resources.engine_dead = False
    drain._engine_exits[99] = 7
    assert _tt_engine_died(engine) is True  # died during the drain (silent exit)
    assert _tt_engine_died(None) is False
