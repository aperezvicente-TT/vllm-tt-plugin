"""The one-replay store through the RUNNER's path (review round 19 O4; program 2026-10-25T07:25:33Z: a host test that
drives one storing prefill through the runner, not only the connector methods). 3e8687d made the connector's load()
return the step's model input -- a dataclasses.replace copy carrying `kda_snapshot_at` (TTModelInput is frozen) -- and
the runner rebind it: the single-kind step (`model_input = kv_host.load(...) or model_input`) and the mixed step's
prefill part (`replace(mixed, prefill=kv_host.load(...) or mixed.prefill)`). These tests run the runner's real
_execute_model_impl / _execute_mixed_step with the connector's real snapshot marking (_mark_kda_snapshots on a real
frozen TTModelInput) inside a stand-in load(), and check that the forward AND the step's save (wrap_finish /
wrap_finish_mixed) both get the copy that carries the cut -- then export the snapshot from what the save got."""

from collections import deque
from dataclasses import MISSING, fields, replace
from functools import partial
from types import SimpleNamespace

import torch

from vllm_tt_plugin import kv_connector as K
from vllm_tt_plugin.model_input import TTMixedModelInput, TTModelInput
from vllm_tt_plugin.model_runner import TTModelRunner


def _model_input(**kw):
    base = {f.name: None for f in fields(TTModelInput) if f.default is MISSING and f.default_factory is MISSING}
    base.update(perform_device_sampling=False)
    base.update(kw)
    return TTModelInput(**base)


def _snap_host(tmp_path, prompt, exports):
    store = K.KDACheckpointStore(str(tmp_path / "snap"), min_free_bytes=0)

    def export(kv, slot, at=None):
        exports.append((slot, at, "snapshot"))
        return [{"L": 0, "S": torch.zeros(2)}]

    host = SimpleNamespace(kda_store=store, _kda_export=export, _kv_cache_object=None,
                           views={"l": SimpleNamespace(block_size=64)}, _tokens_of=lambda rid: prompt, _ckpt_grid=128,
                           _slot_for=lambda *a: 3, checkpoints_off=False,
                           _kda_snapshot_ready=lambda kv, slot, cut: True)  # fmt: skip
    host._prefill_rows = lambda mi: K.TTKVConnectorHost._prefill_rows(host, mi)
    return host, store


def _connector(host, events):
    """the runner-facing connector surface; load() runs the connector's real snapshot marking (the step that replaced
    the frozen input in 3e8687d) and returns its value, as TTKVConnectorHost.load does"""
    return SimpleNamespace(
        begin_step=lambda so: events.append("begin_step"),
        load=lambda so, mi: K.TTKVConnectorHost._mark_kda_snapshots(host, mi),
        wrap_finish=lambda so, mi, finish: events.append(("save", mi)) or finish,
        wrap_finish_mixed=lambda so, pre, dec, finish: events.append(("save_mixed", pre, dec)) or finish,
    )


def _runner(model_input, kv_host, events):
    async_decode = SimpleNamespace(
        can_attempt_steady_decode_from_scheduler=lambda _so: False,
        must_drain_pending_async_steps=lambda *_a: False,
        wait_for_all_pending_async_steps=lambda: events.append("drain"),
        note_prefill_submitted=lambda: events.append("prefill_submitted"),
    )
    r = SimpleNamespace(
        kv_connector_host=kv_host,
        async_decode=async_decode,
        async_decode_scheduling=True,
        _pending_samples=deque(),
        _req_state_slot={},
        build_model_input=lambda so, grammar: model_input,
        _forward_with_model_input=lambda mi: events.append(("forward", mi)) or SimpleNamespace(model_input=mi),
        _finish_front_packed_sync=lambda *a, **k: None,
        _finish_mixed_sync=lambda *a, **k: None,
        _decode_layout_changed_since_last_decode=False,
    )
    r._execute_mixed_step = partial(TTModelRunner._execute_mixed_step, r)
    return r


def _setup(tmp_path, monkeypatch, n=2048):
    monkeypatch.setenv("GLM53_KDA_CKPT_ASYNC", "0")
    monkeypatch.setenv("GLM53_KDA_CKPT_IN_REPLAY", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_EVERY", "1048576")
    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "1024")  # the served store floor
    monkeypatch.setattr(TTModelRunner, "_raise_if_device_grammar_poisoned", staticmethod(lambda _r: None))
    prompt = list(range(n))
    exports, events = [], []
    host, store = _snap_host(tmp_path, prompt, exports)
    pre = _model_input(prompt_lens=[n], row_req_ids=["r"], input_positions=torch.tensor([0]),
                       input_tokens=torch.tensor([prompt]), prefill_empty_slots=[5])  # fmt: skip
    return prompt, host, store, exports, events, pre


def test_single_kind_storing_prefill_forwards_and_saves_the_replaced_input(tmp_path, monkeypatch):
    """a storing prefill of 2,048 tokens: the cut (1,920) rides the copy load() returned into the forward and the save;
    the built input stays untouched (frozen); the snapshot then exports at the cut from what the save got"""
    prompt, host, store, exports, events, pre = _setup(tmp_path, monkeypatch)
    runner = _runner(pre, _connector(host, events), events)
    assert TTModelRunner._execute_model_impl(runner, SimpleNamespace()) is None
    fwd = [e[1] for e in events if isinstance(e, tuple) and e[0] == "forward"]
    saved = [e[1] for e in events if isinstance(e, tuple) and e[0] == "save"]
    assert len(fwd) == 1 and len(saved) == 1
    assert fwd[0].kda_snapshot_at == [1920] and saved[0] is fwd[0]
    assert pre.kda_snapshot_at is None  # the frozen built input was replaced, never assigned
    K.TTKVConnectorHost._export_kda_snapshots(host, saved[0])
    assert exports == [(3, 1920, "snapshot")] and store.has(prompt, 1920)


def test_mixed_step_prefill_part_forwards_and_saves_the_replaced_input(tmp_path, monkeypatch):
    """the mixed step's prefill part: forwarded with the cut; wrap_finish_mixed's prefill input carries it too (and the
    decode part goes through unchanged)"""
    prompt, host, store, exports, events, pre = _setup(tmp_path, monkeypatch)
    dec = _model_input(prompt_lens=None, row_req_ids=["d"], input_positions=torch.tensor([7]),
                       input_tokens=torch.tensor([[1]]))  # fmt: skip
    mixed = TTMixedModelInput(prefill=pre, decode=dec, req_ids=("r", "d"))
    runner = _runner(mixed, _connector(host, events), events)
    monkeypatch.setattr("vllm_tt_plugin.model_runner.is_tt_mixed_step", lambda so: True)
    assert TTModelRunner._execute_model_impl(runner, SimpleNamespace()) is None
    fwd = [e[1] for e in events if isinstance(e, tuple) and e[0] == "forward"]
    assert [f.kda_snapshot_at for f in fwd] == [[1920], None]  # the prefill part, then the decode part
    (save,) = [e for e in events if isinstance(e, tuple) and e[0] == "save_mixed"]
    assert save[1].kda_snapshot_at == [1920] and save[2] is dec
    assert pre.kda_snapshot_at is None
    K.TTKVConnectorHost._export_kda_snapshots(host, save[1])
    assert exports == [(3, 1920, "snapshot")] and store.has(prompt, 1920)


def test_below_the_store_floor_the_runner_forwards_the_built_input_itself(tmp_path, monkeypatch):
    """a 900-token prefill (under the 1,024 store floor; cut 896 would be inside): no snapshot, load() returns the
    input itself, and the forward / save get that very object"""
    prompt, host, store, exports, events, pre = _setup(tmp_path, monkeypatch, n=900)
    runner = _runner(pre, _connector(host, events), events)
    TTModelRunner._execute_model_impl(runner, SimpleNamespace())
    fwd = [e[1] for e in events if isinstance(e, tuple) and e[0] == "forward"]
    saved = [e[1] for e in events if isinstance(e, tuple) and e[0] == "save"]
    assert fwd == [pre] and saved == [pre] and pre.kda_snapshot_at is None
    K.TTKVConnectorHost._export_kda_snapshots(host, saved[0])
    assert exports == []
