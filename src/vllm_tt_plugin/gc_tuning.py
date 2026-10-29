# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""v3b5 engine GC (program 2026-10-29T00:11:38Z / 00:12:19Z, operator priority: blk0's events.jsonl showed
gc_pause_s_total 77.8 s and 1.4-s engine gaps between decode steps; on the mesh f2i3b4 window 9 gen-2 collections took
11.6 s, up to 2.41 s each).

vLLM freezes the startup heap ONCE (EngineCore.__init__ -> freeze_gc_heap, after model load and warm-up). Everything that
becomes long-lived AFTER it -- the first-use builds (prefill programs, slot tables, host shards), the KV connector's and
LMCache's state, the per-request history -- sits in the oldest generation, and every full collection scans it again,
stopping every thread for seconds. Three remedies, each a knob (read once, in the engine process):

  GLM53_GC_THRESHOLD = "g0,g1,g2"  gc.set_threshold at engine start (default 50000,20,100; "off" keeps CPython's
                                  700,10,10). Fewer young collections, and a full one only every g1 x g2 young ones.
  GLM53_GC_REFREEZE = 1           (default) a daemon thread re-freezes the survivors while the engine is IDLE: once
                                  >= GLM53_GC_REFREEZE_STEPS (64) model steps ran since the last freeze and no step has
                                  run for GLM53_GC_IDLE_S (2) seconds, gc.collect() + gc.freeze() -- the one pause it
                                  costs falls where nothing is live, and later full collections scan only what is new.
                                  At most every GLM53_GC_REFREEZE_MIN_S (60) seconds.
  stats                           per generation: collections and pause seconds IN SERVICE (the counters line,
                                  "runner" source: gc_collections_g{0,1,2}, gc_pause_s_g{0,1,2}, gc_pause_s_max), and
                                  the idle refreeze apart (gc_refreezes, gc_refreeze_pause_s, gc_frozen_objects).

A frozen object is never scanned again, so cyclic garbage frozen with it is not collected (reference counting still
frees everything acyclic, which is nearly all per-request state). The refreeze only runs at idle and only after new
steps, so its cost is bounded by what the engine actually allocated."""

from __future__ import annotations

import gc
import os
import threading
import time

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


def _threshold() -> tuple | None:
    v = os.environ.get("GLM53_GC_THRESHOLD", "50000,20,100").strip().lower()
    if v in ("", "off", "0"):
        return None
    try:
        g = tuple(int(x) for x in v.split(","))
        return g if len(g) == 3 and all(x > 0 for x in g) else None
    except ValueError:
        return None


class GCTuning:
    """the engine process's GC policy and accounting (one per process: install())"""

    def __init__(self):
        self.gc_collections_g0 = self.gc_collections_g1 = self.gc_collections_g2 = 0
        self.gc_pause_s_g0 = self.gc_pause_s_g1 = self.gc_pause_s_g2 = 0.0
        self.gc_pause_s_max = 0.0
        self.gc_refreezes = 0
        self.gc_refreeze_pause_s = 0.0  # the refreeze's own collections, at idle (not in gc_pause_s_g*)
        self._in_refreeze = False
        self.gc_frozen_objects = 0
        self._t0: float | None = None
        self._lock = threading.Lock()
        self.steps = 0  # model steps (execute_model calls) since the process started
        self._steps_at_freeze = 0
        self._in_step = 0
        self._last_step_end = time.monotonic()
        self._last_freeze = 0.0
        self.refreeze_steps = int(os.environ.get("GLM53_GC_REFREEZE_STEPS", "64") or 64)
        self.idle_s = float(os.environ.get("GLM53_GC_IDLE_S", "2") or 2)
        self.min_s = float(os.environ.get("GLM53_GC_REFREEZE_MIN_S", "60") or 60)
        self.threshold = _threshold()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- accounting (gc.callbacks: every generation) ----------------------------------------------------------------
    def _callback(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
            return
        if self._t0 is None:
            return
        dt, self._t0 = time.perf_counter() - self._t0, None
        if self._in_refreeze:  # the idle refreeze's own collection: nothing was live
            self.gc_refreeze_pause_s += dt
            return
        g = int(info.get("generation", 0))
        if g == 0:
            self.gc_collections_g0 += 1
            self.gc_pause_s_g0 += dt
        elif g == 1:
            self.gc_collections_g1 += 1
            self.gc_pause_s_g1 += dt
        else:
            self.gc_collections_g2 += 1
            self.gc_pause_s_g2 += dt
        if dt > self.gc_pause_s_max:
            self.gc_pause_s_max = dt

    def pause_total(self) -> float:
        return self.gc_pause_s_g0 + self.gc_pause_s_g1 + self.gc_pause_s_g2

    def report(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()
                if k.startswith("gc_")} | {"gc_pause_s_total": round(self.pause_total(), 4), "steps": self.steps}  # fmt: skip

    # -- the step boundary (the runner) -------------------------------------------------------------------------------
    def step_begin(self) -> None:
        with self._lock:
            self._in_step += 1

    def step_end(self) -> None:
        with self._lock:
            self._in_step = max(0, self._in_step - 1)
            self.steps += 1
            self._last_step_end = time.monotonic()

    # -- the idle refreeze --------------------------------------------------------------------------------------------
    def refreeze_due(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            return (self._in_step == 0 and self.steps - self._steps_at_freeze >= self.refreeze_steps
                    and now - self._last_step_end >= self.idle_s and now - self._last_freeze >= self.min_s)  # fmt: skip

    def refreeze(self) -> dict:
        t = time.perf_counter()
        self._in_refreeze = True
        try:
            gc.collect()
        finally:
            self._in_refreeze = False
        before = gc.get_freeze_count()
        gc.freeze()
        frozen = gc.get_freeze_count() - before
        with self._lock:
            self._steps_at_freeze = self.steps
            self._last_freeze = time.monotonic()
        self.gc_refreezes += 1
        self.gc_frozen_objects = gc.get_freeze_count()
        out = {"frozen_new": frozen, "frozen_total": self.gc_frozen_objects, "s": round(time.perf_counter() - t, 3)}
        logger.info("TT gc: refreeze %d at idle (%d steps): %s", self.gc_refreezes, self.steps, out)
        return out

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            try:
                if self.refreeze_due():
                    self.refreeze()
            except Exception as exc:  # noqa: BLE001 -- the GC policy must never stop the engine
                logger.warning("TT gc: refreeze failed: %s", exc)

    def start(self) -> None:
        if self.threshold is not None:
            gc.set_threshold(*self.threshold)
        gc.callbacks.append(self._callback)
        if os.environ.get("GLM53_GC_REFREEZE", "1") == "1":
            self._thread = threading.Thread(target=self._loop, name="glm53-gc-refreeze", daemon=True)
            self._thread.start()
        logger.info("TT gc: threshold %s (CPython default 700,10,10), idle refreeze %s (after %d steps, %.1f s idle, "
                    "every >= %.0f s)", self.threshold or "default", self._thread is not None, self.refreeze_steps,
                    self.idle_s, self.min_s)  # fmt: skip

    def stop(self) -> None:
        self._stop.set()


_GC: GCTuning | None = None


def install() -> GCTuning:
    """once per engine process (the worker's init_device): the thresholds, the accounting and the idle refreeze; the
    counters line reads it as a "runner" source"""
    global _GC
    if _GC is None:
        _GC = GCTuning()
        _GC.start()
        try:
            from vllm_tt_plugin.kv_connector import _COUNTER_SOURCES  # noqa: PLC0415

            _COUNTER_SOURCES["runner"].add(_GC)
        except Exception:  # noqa: BLE001
            pass
    return _GC


def get() -> GCTuning | None:
    return _GC
