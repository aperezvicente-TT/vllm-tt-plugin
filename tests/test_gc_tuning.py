# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""v3b5 engine GC policy (program 2026-10-29T00:12:19Z (2)): the threshold knob, the idle-refreeze rule, the
refreeze itself (survivors frozen, its own pause kept apart from the in-service pauses), the counters line."""

import gc

import pytest

from vllm_tt_plugin import gc_tuning


@pytest.fixture
def fresh(monkeypatch):
    for k in ("GLM53_GC_THRESHOLD", "GLM53_GC_REFREEZE", "GLM53_GC_REFREEZE_STEPS", "GLM53_GC_IDLE_S",
              "GLM53_GC_REFREEZE_MIN_S"):  # fmt: skip
        monkeypatch.delenv(k, raising=False)
    old = gc.get_threshold()
    yield
    gc.set_threshold(*old)
    gc.unfreeze()


@pytest.mark.parametrize("v,want", [(None, (50000, 20, 100)), ("off", None), ("0", None), ("1000,5,7", (1000, 5, 7)),
                                    ("bad", None), ("1,2", None), ("1,0,3", None)])  # fmt: skip
def test_threshold_knob(fresh, monkeypatch, v, want):
    if v is not None:
        monkeypatch.setenv("GLM53_GC_THRESHOLD", v)
    assert gc_tuning._threshold() == want


def test_refreeze_waits_for_steps_idle_and_no_step_in_flight(fresh, monkeypatch):
    monkeypatch.setenv("GLM53_GC_REFREEZE_STEPS", "3")
    monkeypatch.setenv("GLM53_GC_IDLE_S", "2")
    monkeypatch.setenv("GLM53_GC_REFREEZE_MIN_S", "0")
    g = gc_tuning.GCTuning()
    t = g._last_step_end
    assert not g.refreeze_due(t + 10)  # no steps yet
    for _ in range(3):
        g.step_begin()
        g.step_end()
    t = g._last_step_end
    assert not g.refreeze_due(t + 1)  # not idle long enough
    assert g.refreeze_due(t + 2.5)
    g.step_begin()
    assert not g.refreeze_due(t + 10)  # a step in flight
    g.step_end()


def test_refreeze_freezes_survivors_and_keeps_its_pause_apart(fresh, monkeypatch):
    monkeypatch.setenv("GLM53_GC_REFREEZE_MIN_S", "0")
    g = gc_tuning.GCTuning()
    gc.callbacks.append(g._callback)
    try:
        keep = [[i] for i in range(200000)]  # long-lived, created after any earlier freeze
        before = gc.get_freeze_count()
        g2, p2 = g.gc_collections_g2, g.gc_pause_s_g2  # (allocating `keep` may have run automatic collections)
        out = g.refreeze()
        assert out["frozen_new"] >= len(keep) and gc.get_freeze_count() >= before + len(keep)
        assert g.gc_refreezes == 1 and g.gc_refreeze_pause_s > 0
        assert (g.gc_collections_g2, g.gc_pause_s_g2) == (g2, p2)  # the idle collection is not an in-service pause
        gc.collect(2)  # an in-service full collection after it: counted
        assert g.gc_collections_g2 == g2 + 1
    finally:
        gc.callbacks.remove(g._callback)


def test_the_counters_line_reports_the_gc_policy(fresh):
    from vllm_tt_plugin.kv_connector import _COUNTER_SOURCES, counters_snapshot

    g = gc_tuning.GCTuning()
    g.gc_refreezes, g.gc_pause_s_g2, g.gc_collections_g2 = 2, 1.23456, 3
    _COUNTER_SOURCES["runner"].add(g)
    snap = counters_snapshot()
    assert snap["gc_refreezes"] >= 2 and snap["gc_collections_g2"] >= 3 and snap["gc_pause_s_g2"] >= 1.235
