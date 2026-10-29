"""The plugin hook module (src/vllm_tt_plugin/stepmeter.py), imported without vLLM."""

import importlib.util
import json
import os
import sys
import time

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src", "vllm_tt_plugin", "stepmeter.py")


def load(monkeypatch, path):
    monkeypatch.setenv("GLM53_STEP_STATUS", path)
    monkeypatch.setenv("GLM53_STEP_STATUS_EVERY_S", "0")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    spec = importlib.util.spec_from_file_location("stepmeter_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("GLM53_STEP_STATUS", raising=False)
    spec = importlib.util.spec_from_file_location("stepmeter_off", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.get_step_meter()
    assert not m.enabled
    m.begin("decode"); m.set_kind("prefill"); m.end(); m.error("x"); m.spec_accepted(2); m.mark_ready(); m.flush()
    assert m.steps == {"prefill": 0, "decode": 0, "lane": 0}


def test_counters_and_snapshot_file(monkeypatch, tmp_path):
    path = str(tmp_path / "step_status.json")
    mod = load(monkeypatch, path)
    m = mod.get_step_meter()
    assert m.enabled
    m.begin()
    m.set_kind("prefill")
    time.sleep(0.01)
    m.end()
    m.begin("decode")
    m.end()
    m.begin("decode")
    m.cancel()  # scheduled nothing
    m.spec_accepted(3)
    m.mark_ready()
    snap = json.load(open(path))
    assert snap["steps"] == {"prefill": 1, "decode": 1, "lane": 0}
    assert snap["ready"] is True and snap["in_flight"] is False
    assert snap["step_ms_last"] >= 0 and snap["last_step"] > 0
    assert snap["spec_accepted_tokens"] == 3 and snap["spec_verify_rows"] == 1
    assert snap["pid"] == os.getpid()


def test_in_flight_visible_and_error_counted(monkeypatch, tmp_path):
    path = str(tmp_path / "s.json")
    mod = load(monkeypatch, path)
    m = mod.get_step_meter()
    m.begin("decode")
    m.flush()
    snap = json.load(open(path))
    assert snap["in_flight"] is True and snap["kind"] == "decode" and snap["step_begin"] > 0
    try:
        raise RuntimeError("Device timeout detected")
    except RuntimeError as e:
        m.error(e)
    snap = json.load(open(path))
    assert snap["device_errors"] == 1 and "Device timeout" in snap["last_error"] and snap["in_flight"] is False
    m.mark_shutting_down()
    assert json.load(open(path))["shutting_down"] is True


def test_unwritable_path_never_raises(monkeypatch):
    mod = load(monkeypatch, "/nonexistent-dir/x/step.json")
    m = mod.get_step_meter()
    m.begin("decode"); m.end(); m.error("boom"); m.mark_ready()
    assert m.steps["decode"] == 1


def test_model_runner_hook_sites_present():
    """The runner carries every hook (the plugin tests need vLLM, so this is a static check)."""
    src = open(os.path.join(os.path.dirname(SRC), "model_runner.py")).read()
    for needle in (
        "from vllm_tt_plugin.stepmeter import get_step_meter",
        "meter.begin()",
        "return self._execute_model_impl(scheduler_output)",
        'get_step_meter().set_kind("decode" if is_decode else "prefill")',
        'get_step_meter().set_kind("lane")',
        "get_step_meter().cancel()",
        "meter.end()",
        "get_step_meter().mark_ready()",
        "get_step_meter().mark_shutting_down()",
        "get_step_meter().spec_accepted(len(block))",
    ):
        assert needle in src, needle
    assert src.count("meter.error(exc)") == 2  # execute_model and sample_tokens
