# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A device step exception latches the runner unhealthy: no further steps, no retry."""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin.model_runner import (
    TTDeviceFaultError,
    TTModelRunner,
    _latch_device_faults,
)
from vllm_tt_plugin.worker import TTWorker


class _Runner:
    """Only the latch state plus one guarded step method."""

    def __init__(self):
        self.calls = 0

    _latch_device_fault = TTModelRunner._latch_device_fault
    _raise_if_device_faulted = TTModelRunner._raise_if_device_faulted
    check_health = TTModelRunner.check_health
    device_fault = TTModelRunner.device_fault

    @_latch_device_faults
    def step(self, fail=None):
        self.calls += 1
        if fail is not None:
            raise fail
        return "ok"


def test_healthy_runner_steps_and_reports_healthy():
    r = _Runner()
    assert r.step() == "ok" and r.device_fault is None
    r.check_health()


def test_step_exception_latches_and_later_steps_refuse_without_running():
    r = _Runner()
    boom = RuntimeError("Completion Timeout on chip 3")
    with pytest.raises(RuntimeError, match="Completion Timeout"):
        r.step(fail=boom)
    assert r.device_fault is boom and r.calls == 1
    with pytest.raises(
        TTDeviceFaultError, match="no retry re-opens the devices"
    ) as exc:
        r.check_health()
    assert exc.value.__cause__ is boom
    # the next step is refused before the method body runs: no new device call
    with pytest.raises(TTDeviceFaultError):
        r.step()
    assert r.calls == 1
    # a second fault does not replace the first (the root cause stays visible)
    r._latch_device_fault(ValueError("later"))
    assert r.device_fault is boom


def test_worker_health_probe_reports_the_runner_fault():
    w = SimpleNamespace(model_runner=_Runner())
    TTWorker.check_health(w)  # healthy
    with pytest.raises(RuntimeError):
        w.model_runner.step(fail=RuntimeError("x"))
    with pytest.raises(TTDeviceFaultError):
        TTWorker.check_health(w)
    TTWorker.check_health(SimpleNamespace())  # before init_device: nothing to probe


def test_the_first_fault_calls_the_close_hook_once_and_a_failing_hook_does_not_mask_it():
    """the worker's close hook (TTWorker._close_mesh_after_fault) runs at the latch, once, before the fault propagates"""
    r = _Runner()
    seen = []
    r._on_device_fault = seen.append
    boom = RuntimeError("Completion Timeout on chip 3")
    with pytest.raises(RuntimeError):
        r.step(fail=boom)
    assert seen == [boom]
    r._latch_device_fault(ValueError("later"))  # a second fault: no second close
    assert seen == [boom]

    r2 = _Runner()

    def bad(_exc):
        raise OSError("close failed")

    r2._on_device_fault = bad
    with pytest.raises(RuntimeError, match="original"):
        r2.step(fail=RuntimeError("original"))
    assert isinstance(r2.device_fault, RuntimeError)


def test_worker_close_after_fault_shuts_down_once_and_marks_the_close(monkeypatch, tmp_path):
    """TTWorker.shutdown closes the mesh once and writes the marker the API's exit report reads"""
    import os

    from vllm_tt_plugin import drain, worker as W

    monkeypatch.setenv("GLM53_TT_MESH_MARKER_DIR", str(tmp_path))
    closed = []
    monkeypatch.setattr(W, "close_mesh_device", lambda m, cfg: closed.append(m))
    monkeypatch.setattr(W, "get_tt_config", lambda cfg: {})
    monkeypatch.setattr(W, "ensure_tt_kv_transfer_shutdown", lambda: None)
    runner = SimpleNamespace(shutdown=lambda: None)
    w = SimpleNamespace(model_runner=runner, mesh_device="mesh", vllm_config=None)
    w.shutdown = lambda: TTWorker.shutdown(w)
    TTWorker._close_mesh_after_fault(w, RuntimeError("x"))
    TTWorker.shutdown(w)  # the engine's finally-shutdown later: nothing to close again
    assert closed == ["mesh"]
    assert os.path.exists(drain.mesh_close_marker(os.getpid()))
