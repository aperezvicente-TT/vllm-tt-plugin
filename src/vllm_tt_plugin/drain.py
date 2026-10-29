# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Bounded, observable drain-mode shutdown for TT engines (vLLM 0.26 ``--shutdown-timeout N > 0``).

Upstream's drain has three gaps that matter when the engine process owns an open TT mesh:

1. ``EngineCoreProc._handle_shutdown`` (drain mode) has no deadline of its own: it steps until the
   in-flight requests finish. The only bound is the process manager's ``shutdown(procs, timeout)``
   (``vllm/v1/utils.py``), which SIGKILLs the engine process at ``shutdown_timeout`` -- a device
   process killed mid-step, mesh left open. A generation longer than the timeout always ends that
   way. :func:`install_engine_drain_patch` gives the engine its own budget, shorter than the kill
   timer: at the budget it aborts the remaining requests through the engine's own abort path (the
   same ``finish_requests(None, FINISHED_ABORTED)`` + ``_send_abort_outputs`` upstream's abort mode
   runs), between two steps -- a running step is never interrupted -- and the busy loop then exits
   the orderly way, so ``TTWorker.shutdown`` closes the mesh.
2. The process manager never reports how the engine process ended: ``monitor_engine_liveness``
   skips ``failed_proc_name`` once ``manager_stopped`` is set, and ``shutdown(procs)`` only logs a
   force kill. An engine process that dies during the drain (e.g. tt-umd's SIGBUS handler, which
   ``_exit(SIGBUS)``-s -- status 7, no Python output -- when a device reset zapped its BAR
   mappings) leaves no trace. :func:`install_api_drain_patches` logs every engine process's exit
   status after the join and records abnormal ones.
3. After that death the API server's output handler is cancelled by ``AsyncLLM.shutdown``, so the
   streams still open never get an error and hang until the client times out (the HTTP server waits
   for them). The patched ``AsyncLLM.shutdown`` ends them with ``EngineDeadError``.

``abnormal_engine_exit()`` is what the TT exit-status patch (platform.py) uses instead of
``engine_client.errored``: ``errored`` is True after every ``AsyncLLM.shutdown`` (the output handler
is cancelled), so an orderly drain with a stream still open exited 70 ("device step failed").
"""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

# Engine-side drain budget override (seconds); else additional_config["tt"]["drain_timeout_s"].
DRAIN_TIMEOUT_ENV = "VLLM_TT_DRAIN_TIMEOUT_S"
# tt-umd silicon_tlb_window.cpp sigbus_handler: ``_exit(sig)`` outside a guarded TLB access.
UMD_SIGBUS_EXIT_STATUS = int(signal.SIGBUS)

_lock = threading.Lock()
_engine_exits: dict[int, int | None] = {}  # pid -> exitcode as seen after the process-manager join


def drain_budget_s(vllm_config: Any) -> float | None:
    """Seconds the engine may drain before it aborts what is left; None = upstream behaviour.

    Always leaves the process manager's kill timer (``shutdown_timeout``) a margin for the abort,
    the step in flight and the mesh close: min(120 s, max(5 s, 25 % of the timeout)).
    """
    timeout = float(getattr(vllm_config, "shutdown_timeout", 0) or 0)
    if timeout <= 0:
        return None  # abort mode: upstream aborts at once
    margin = min(120.0, max(5.0, 0.25 * timeout))
    cap = max(0.0, timeout - margin)
    want = os.environ.get(DRAIN_TIMEOUT_ENV, "")
    if not want:
        extra = getattr(vllm_config, "additional_config", None) or {}
        want = (extra.get("tt") or {}).get("drain_timeout_s", "") if isinstance(extra, dict) else ""
    if want not in ("", None):
        return min(float(want), cap)
    return cap


def mesh_close_marker(pid: int) -> str:
    """The file TTWorker writes once it has closed its mesh (engine process ``pid``). The API process reads it to
    report how an engine ended: an engine killed by a signal after it closed its mesh needs no reset for the mesh.
    In ``GLM53_TT_MESH_MARKER_DIR``, else next to ``GLM53_VLLM_EVENTS`` (the run dir), else the temp dir."""
    base = os.environ.get("GLM53_TT_MESH_MARKER_DIR")
    if not base and os.environ.get("GLM53_VLLM_EVENTS"):
        base = os.path.dirname(os.environ["GLM53_VLLM_EVENTS"])
    if not base:
        import tempfile  # noqa: PLC0415

        base = tempfile.gettempdir()
    return os.path.join(base, f"tt_mesh_closed.{int(pid)}")


def describe_exit(code: int | None) -> str:
    if code is None:
        return "still running"
    if code == 0:
        return "exit 0 (orderly)"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        return f"killed by {name}"
    if code == UMD_SIGBUS_EXIT_STATUS:
        return (
            f"exit status {code}: tt-umd's SIGBUS handler _exit(SIGBUS) -- a device BAR mapping was "
            "invalidated under the live process (a device reset / tt-kmd mapping zap)"
        )
    return f"exit status {code}"


def abnormal_engine_exit() -> bool:
    with _lock:
        return any(code not in (0, None) for code in _engine_exits.values())


def engine_exits() -> dict[int, int | None]:
    with _lock:
        return dict(_engine_exits)


def install_engine_drain_patch() -> None:
    """Engine process: bound the drain and abort through the engine's own abort path."""
    import vllm.v1.engine.core as core
    from vllm.v1.request import RequestStatus

    cls = core.EngineCoreProc
    if getattr(cls._handle_shutdown, "_tt_drain_patch", False):
        return
    original = cls._handle_shutdown
    State = core.EngineShutdownState

    def _handle_shutdown(self) -> bool:
        if self.shutdown_state == State.REQUESTED:
            self._tt_drain_t0 = time.monotonic()
            self._tt_drain_budget = drain_budget_s(self.vllm_config)
            self._tt_drain_aborted = False
            if self._tt_drain_budget is not None:
                logger.info(
                    "[shutdown] TT drain: in-flight requests may finish for %.0f s (kill timer %ss); "
                    "the rest are then aborted between steps and the mesh closes in TTWorker.shutdown",
                    self._tt_drain_budget,
                    self.vllm_config.shutdown_timeout,
                )
        keep_running = original(self)
        budget = getattr(self, "_tt_drain_budget", None)
        if (
            keep_running
            and budget is not None
            and self.shutdown_state == State.SHUTTING_DOWN
            and not getattr(self, "_tt_drain_aborted", True)
            and time.monotonic() - self._tt_drain_t0 >= budget
        ):
            # Top of a busy-loop iteration: the previous step returned; nothing runs on the device
            # from this thread. Upstream's abort mode, applied late.
            self._tt_drain_aborted = True
            n = self.scheduler.get_num_unfinished_requests()
            logger.warning(
                "[shutdown] TT drain budget %.0f s reached: aborting %d in-flight request(s) through "
                "the engine's abort path",
                budget,
                n,
            )
            aborted = self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
            self._send_abort_outputs(aborted)
        return keep_running

    _handle_shutdown._tt_drain_patch = True
    cls._handle_shutdown = _handle_shutdown


def install_api_drain_patches() -> None:
    """API process: report how each engine process ended; end streams a dead engine left open."""
    import vllm.v1.engine.utils as engine_utils

    if not getattr(engine_utils.shutdown, "_tt_drain_patch", False):
        original_shutdown = engine_utils.shutdown

        def shutdown(procs, timeout=None):
            original_shutdown(procs, timeout=timeout)
            for proc in procs:
                code = proc.exitcode
                with _lock:
                    _engine_exits[proc.pid] = code
                closed = os.path.exists(mesh_close_marker(proc.pid))
                log = logger.info if code == 0 else logger.error
                log(
                    "[shutdown] TT: %s pid %s %s%s",
                    proc.name,
                    proc.pid,
                    describe_exit(code),
                    ""
                    if code == 0
                    else (
                        "; TTWorker had closed its mesh before the exit (no reset needed for the mesh)"
                        if closed
                        # review round 13 O4: whether a reset is needed is the reset policy's call (device-node
                        # holders after the exit, chip health), not this marker's: UMD may still close the devices
                        else "; it did not reach TTWorker.shutdown: its mesh was NOT closed by TTWorker (whether a "
                        "reset is needed: the device-node holders after the exit and the reset policy decide)"
                    ),
                )

        shutdown._tt_drain_patch = True
        engine_utils.shutdown = shutdown

    from vllm.v1.engine.async_llm import AsyncLLM

    if getattr(AsyncLLM.shutdown, "_tt_drain_patch", False):
        return
    original_async_shutdown = AsyncLLM.shutdown

    def async_shutdown(self, timeout: float | None = None) -> None:
        handler = getattr(self, "output_handler", None)
        loop = None
        if handler is not None:
            try:
                loop = handler.get_loop()
            except Exception:
                loop = None
        original_async_shutdown(self, timeout=timeout)
        processor = getattr(self, "output_processor", None)
        if not abnormal_engine_exit() or processor is None or loop is None:
            return
        if not processor.has_unfinished_requests() or getattr(self, "_tt_dead_streams_ended", False):
            return
        self._tt_dead_streams_ended = True
        from vllm.v1.engine.exceptions import EngineDeadError

        logger.error(
            "[shutdown] TT: the engine process died with %d request(s) in flight: ending their streams "
            "with EngineDeadError",
            processor.get_num_unfinished_requests(),
        )
        try:
            loop.call_soon_threadsafe(processor.propagate_error, EngineDeadError())
        except RuntimeError:  # loop already closed
            pass

    async_shutdown._tt_drain_patch = True
    AsyncLLM.shutdown = async_shutdown
