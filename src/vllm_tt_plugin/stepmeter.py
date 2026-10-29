# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Step meter: the EngineCore's liveness and step counters for an external supervisor.

Stdlib-only (importing ``vllm`` here would re-enter plugin loading). Disabled unless
``GLM53_STEP_STATUS`` names a file; then the runner's step boundaries are recorded and
an atomic JSON snapshot of the counters is written there (at most every
``GLM53_STEP_STATUS_EVERY_S`` seconds, and at once on errors, readiness and shutdown):

    {"pid", "ready", "shutting_down", "in_flight", "kind", "step_begin", "last_step",
     "steps": {"prefill", "decode", "mixed", "lane"}, "step_ms_last", "device_errors",
     "last_error", "spec_accepted_tokens", "spec_verify_rows", "updated"}

A supervisor polls the file instead of the API server: a device hang shows as
``in_flight`` with a stale ``step_begin`` even when the HTTP front end still answers
``/health``. When ``PROMETHEUS_MULTIPROC_DIR`` is set (serve_window.sh), the same
counters are exported as multiprocess Prometheus metrics (``glm53_steps_total{kind}``,
``glm53_device_errors_total``, ``glm53_last_step_timestamp_seconds``,
``glm53_spec_accepted_tokens_total``, ``glm53_spec_verify_rows_total``) so the API
server's ``/metrics`` carries them next to ``glm53_startup_seconds``.
"""

from __future__ import annotations

import json
import os
import threading
import time

STATUS_ENV = "GLM53_STEP_STATUS"
EVERY_ENV = "GLM53_STEP_STATUS_EVERY_S"
KINDS = ("prefill", "decode", "mixed", "lane")


class StepMeter:
    """Counters for one EngineCore process; every method is cheap and never raises."""

    def __init__(self, path: str | None, every_s: float = 1.0) -> None:
        self.path = path
        self.every_s = every_s
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self.ready = False
        self.shutting_down = False
        self.in_flight = False
        self.kind = ""
        self.step_begin = 0.0
        self.last_step = 0.0
        self.steps = {k: 0 for k in KINDS}
        self.step_ms_last = 0.0
        self.device_errors = 0
        self.last_error = ""
        self.spec_accepted_tokens = 0
        self.spec_verify_rows = 0
        self._prom = None
        if path and os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            self._prom = _prometheus_metrics()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    # -- step boundaries ---------------------------------------------------
    def begin(self, kind: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            self.in_flight = True
            self.kind = kind
            self.step_begin = time.time()
        self._maybe_flush()

    def set_kind(self, kind: str) -> None:
        if self.enabled:
            with self._lock:
                self.kind = kind

    def cancel(self) -> None:
        """The step scheduled nothing: no device work was submitted."""
        if self.enabled:
            with self._lock:
                self.in_flight = False
                self.kind = ""

    def end(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            now = time.time()
            if self.in_flight:
                self.step_ms_last = (now - self.step_begin) * 1000.0
            kind = self.kind if self.kind in self.steps else "lane"
            self.steps[kind] += 1
            self.in_flight = False
            self.last_step = now
            if self._prom:
                self._prom["steps"].labels(kind=kind).inc()
                self._prom["last_step"].set(now)
        self._maybe_flush()

    def error(self, exc: BaseException | str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.device_errors += 1
            self.last_error = (f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc))[:400]
            self.in_flight = False
            if self._prom:
                self._prom["errors"].inc()
        self.flush()

    def spec_accepted(self, accepted_tokens: int, rows: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.spec_accepted_tokens += int(accepted_tokens)
            self.spec_verify_rows += int(rows)
            if self._prom:
                self._prom["spec_accepted"].inc(int(accepted_tokens))
                self._prom["spec_rows"].inc(int(rows))

    def mark_ready(self) -> None:
        if self.enabled:
            with self._lock:
                self.ready = True
            self.flush()

    def mark_shutting_down(self) -> None:
        if self.enabled:
            with self._lock:
                self.shutting_down = True
            self.flush()

    # -- snapshot ----------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pid": os.getpid(),
                "ready": self.ready,
                "shutting_down": self.shutting_down,
                "in_flight": self.in_flight,
                "kind": self.kind,
                "step_begin": self.step_begin,
                "last_step": self.last_step,
                "steps": dict(self.steps),
                "step_ms_last": round(self.step_ms_last, 3),
                "device_errors": self.device_errors,
                "last_error": self.last_error,
                "spec_accepted_tokens": self.spec_accepted_tokens,
                "spec_verify_rows": self.spec_verify_rows,
                "updated": time.time(),
            }

    def _maybe_flush(self) -> None:
        if time.time() - self._last_flush >= self.every_s:
            self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            snap = self.snapshot()
            tmp = f"{self.path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, self.path)
            self._last_flush = time.time()
        except OSError:
            pass  # the meter never disturbs the step


class _NullMeter(StepMeter):
    def __init__(self) -> None:
        super().__init__(None)


def _prometheus_metrics() -> dict | None:
    try:
        from prometheus_client import Counter, Gauge
    except ImportError:
        return None
    try:
        return {
            "steps": Counter("glm53_steps", "GLM-5.3-Flash engine steps by kind", ["kind"]),
            "errors": Counter("glm53_device_errors", "GLM-5.3-Flash engine steps that raised"),
            "last_step": Gauge(
                "glm53_last_step_timestamp_seconds",
                "Unix time of the last completed engine step",
                multiprocess_mode="max",
            ),
            "spec_accepted": Counter(
                "glm53_spec_accepted_tokens", "Tokens committed by speculative verify steps"
            ),
            "spec_rows": Counter("glm53_spec_verify_rows", "Rows committed by speculative verify steps"),
        }
    except ValueError:
        return None  # registered twice in one process (tests): keep the file channel only


_METER: StepMeter | None = None
_METER_LOCK = threading.Lock()


def get_step_meter() -> StepMeter:
    """The process-wide meter: a file-backed one when GLM53_STEP_STATUS is set, else a no-op."""
    global _METER
    if _METER is None:
        with _METER_LOCK:
            if _METER is None:
                path = os.environ.get(STATUS_ENV)
                try:
                    every = float(os.environ.get(EVERY_ENV, "1.0"))
                except ValueError:
                    every = 1.0
                _METER = StepMeter(path, every) if path else _NullMeter()
    return _METER


def reset_step_meter() -> None:
    """Tests only: drop the singleton so the next call re-reads the environment."""
    global _METER
    with _METER_LOCK:
        _METER = None
