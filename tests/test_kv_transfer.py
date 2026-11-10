"""Host-only tests for KV connector support (vllm_tt_plugin.kv_transfer and scheduler
helpers)."""

from types import SimpleNamespace

import torch

from vllm_tt_plugin import kv_transfer
from vllm_tt_plugin.scheduler import _merge_connector_metadata


class Meta:
    def __init__(self, **kw):
        self.reqs_to_recv = {}
        self.reqs_to_save = {}
        self.reqs_in_batch = set()
        self.scheduler_clock = 0.0
        self.__dict__.update(kw)


def test_merge_keeps_first_pass_receives():
    first = Meta(reqs_to_recv={"a": 1}, reqs_in_batch={"a"}, scheduler_clock=1.0)
    second = Meta(reqs_to_save={"b": 2}, reqs_in_batch={"b"}, scheduler_clock=2.0)
    out = _merge_connector_metadata(first, second)
    assert out is second
    assert out.reqs_to_recv == {"a": 1} and out.reqs_to_save == {"b": 2}
    assert out.reqs_in_batch == {"a", "b"}
    assert out.scheduler_clock == 2.0  # scalars from the second pass


def test_merge_multi_connector_children():
    """MultiConnector metadata: each connector's first-pass work survives the second
    pass."""
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorMetadata,
    )

    first = MultiKVConnectorMetadata(
        metadata=(Meta(reqs_to_recv={"a": 1}), Meta(reqs_to_save={"x": 1})),
        extra_async_saves={"a": 1},
    )
    second = MultiKVConnectorMetadata(
        metadata=(Meta(reqs_to_save={"b": 2}), Meta(reqs_to_save={"y": 2})),
    )
    out = _merge_connector_metadata(first, second)
    nixl, lmc = out.metadata
    assert nixl.reqs_to_recv == {"a": 1} and nixl.reqs_to_save == {"b": 2}
    assert lmc.reqs_to_save == {"x": 1, "y": 2}
    assert out.extra_async_saves == {"a": 1}


def test_merge_none():
    m = Meta()
    assert _merge_connector_metadata(None, m) is m
    assert _merge_connector_metadata(m, None) is m


def params(**kw):
    base = dict(
        prompt_logprobs=None,
        seed=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_decode_join_sampling_rules():
    assert kv_transfer.decode_join_sampling_ok(None)
    assert kv_transfer.decode_join_sampling_ok(params())
    assert not kv_transfer.decode_join_sampling_ok(params(seed=3))
    assert not kv_transfer.decode_join_sampling_ok(params(presence_penalty=0.5))
    assert not kv_transfer.decode_join_sampling_ok(params(repetition_penalty=1.1))
    assert not kv_transfer.decode_join_sampling_ok(params(prompt_logprobs=1))


class FakeOp:
    def __init__(self):
        self.ticket = None
        self.done = set()

    def take_load_ticket(self):
        t, self.ticket = self.ticket, None
        return t

    def load_done(self, t):
        return t in self.done


def test_gate_holds_until_load_done(monkeypatch):
    op = FakeOp()
    monkeypatch.setattr(kv_transfer, "_copy_op", op)
    monkeypatch.setattr(kv_transfer, "_held_recvs", [])
    out = SimpleNamespace(finished_recving={"r1"})
    op.ticket = 7
    kv_transfer.gate_received(out)
    assert out.finished_recving is None  # load still running: held back
    out2 = SimpleNamespace(finished_recving=None)
    op.done.add(7)
    kv_transfer.gate_received(out2)
    assert out2.finished_recving == {"r1"}  # released in a later step


def test_shadow_caches_are_storage_free():
    spec = SimpleNamespace(
        num_kv_heads=8, block_size=64, head_size=128, dtype=torch.bfloat16
    )
    cfg = SimpleNamespace(
        num_blocks=1000,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["l0", "l1"])],
    )
    caches = kv_transfer.shadow_kv_caches(cfg)
    assert set(caches) == {"l0", "l1"}
    t = caches["l0"]
    assert t.shape == (1000, 8, 64, 256) and t.dtype == torch.bfloat16
    assert t.untyped_storage().nbytes() == 2  # one element, expanded


def test_release_ready_at_sample_time(monkeypatch):
    op = FakeOp()
    monkeypatch.setattr(kv_transfer, "_copy_op", op)
    monkeypatch.setattr(kv_transfer, "_held_recvs", [])
    out = SimpleNamespace(finished_recving={"r1"})
    op.ticket = 3
    kv_transfer.gate_received(out)  # after the lifecycle: load still running
    assert out.finished_recving is None
    op.done.add(3)
    kv_transfer.release_ready(out)  # at sample time, same step
    assert out.finished_recving == {"r1"}


def test_fast_top_k_top_p_matches_upstream():
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    from vllm_tt_plugin.host_sampler import fast_apply_top_k_top_p

    g = torch.Generator().manual_seed(0)
    vocab = 32000
    for scale, temp in (
        (3.0, 0.6),
        (1.0, 1.0),
        (0.2, 1.0),
        (8.0, 0.3),
    ):  # sharp ... very flat
        logits = torch.randn(16, vocab, generator=g) * scale / temp
        p = torch.rand(16, generator=g) * 0.95 + 0.04
        for k in (None, torch.randint(1, 300, (16,), generator=g)):
            ref = apply_top_k_top_p_pytorch(
                logits.clone(), k, p.clone(), allow_cpu_sync=True
            )
            got = fast_apply_top_k_top_p(logits.clone(), k, p.clone())
            assert torch.equal(torch.isfinite(ref), torch.isfinite(got)), (
                scale,
                temp,
                k is None,
            )
            kept = torch.isfinite(ref)
            assert torch.equal(ref[kept], got[kept])


def test_compat_factors_from_copy_op(monkeypatch):
    monkeypatch.setattr(kv_transfer, "_copy_op", None)
    assert kv_transfer.kv_transfer_compat_factors() == {}
    monkeypatch.setattr(
        kv_transfer, "_copy_op", SimpleNamespace(layout_hash=lambda: "abc")
    )
    assert kv_transfer.kv_transfer_compat_factors() == {"tt_kv_layout": "abc"}


def test_copy_op_built_before_caches_registered(monkeypatch):
    order = []

    class Conn:
        def register_kv_caches(self, caches):
            order.append(("register", kv_transfer._copy_op))

        def set_host_xfer_buffer_ops(self, op):
            order.append(("ops", op))

    import vllm.distributed.kv_transfer as kvt

    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: Conn())
    monkeypatch.setattr(kv_transfer, "_copy_op_provider", lambda m, c, cfg: "OP")
    monkeypatch.setattr(kv_transfer, "_copy_op", None)
    spec = SimpleNamespace(
        num_kv_heads=1, block_size=4, head_size=8, dtype=torch.bfloat16
    )
    cfg = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["l0"])],
    )
    kv_transfer.register_device_caches(None, {}, cfg)
    assert order == [
        ("register", "OP"),
        ("ops", "OP"),
    ]  # the op existed when caches were registered


def test_block_float_spec_and_shadow_bytes():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    base = FullAttentionSpec(
        block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    assert kv_transfer.device_kv_spec(base, "bfloat16") is base
    spec = kv_transfer.device_kv_spec(base, "bfloat8_b")
    # 2 (K, V) x 4 tiles across the head x 1088 bytes per tile / 32 rows
    assert spec.state_content_bytes == 272 and spec.dtype == torch.uint8
    tiles_per_block = 2 * 8 * (64 // 32) * (128 // 32)
    assert spec.page_size_bytes == tiles_per_block * 1088
    cfg = SimpleNamespace(
        num_blocks=10,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["l0"])],
    )
    sh = kv_transfer.shadow_kv_caches(cfg)["l0"]
    assert sh.dtype == torch.uint8 and sh.shape == (10, 8, 64, 272)
    assert sh.numel() == 10 * spec.page_size_bytes


def test_wire_layout_spec_is_the_wire_dtype():
    """With a wire layout host buffers hold bf16 rows whatever the device stores, so
    bf16 and bfloat8_b caches declare the same spec: the same host buffers, NIXL block
    lengths and pages."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    base = FullAttentionSpec(
        block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )
    for dt in ("bfloat16", "bfloat8_b"):
        spec = kv_transfer.device_kv_spec(base, dt, "LBHNC")
        assert spec is base and spec.page_size_bytes == 2 * 8 * 64 * 128 * 2


# Streamed saves: a chunk's save overlaps the next chunk; only the step that completes a
# prompt waits.


def sched(new=(), cached=(), finished=(), preempted=None):
    """Scheduler output: `new` / `cached` are (req_id, computed_before, scheduled)
    tuples."""
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, num_computed_tokens=c) for r, c, _ in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _, _ in cached],
            num_computed_tokens=[c for _, c, _ in cached],
        ),
        num_scheduled_tokens={r: n for r, _, n in (*new, *cached)},
        finished_req_ids=set(finished),
        preempted_req_ids=preempted,
    )


PROMPTS = {"a": 3000, "b": 1000}


def test_partial_prefill_steps_leave_saves_running():
    lens = PROMPTS.get
    assert not kv_transfer.output_needs_saves(sched(new=[("a", 0, 1024)]), lens)
    assert not kv_transfer.output_needs_saves(sched(cached=[("a", 1024, 1024)]), lens)
    assert kv_transfer.output_needs_saves(
        sched(cached=[("a", 2048, 952)]), lens
    )  # last chunk
    # any request that can finish makes the step wait (saves complete in order)
    assert kv_transfer.output_needs_saves(
        sched(new=[("b", 0, 1000)], cached=[("a", 1024, 1024)]), lens
    )
    assert kv_transfer.output_needs_saves(
        sched(new=[("x", 0, 10)]), lens
    )  # unknown length
    assert kv_transfer.output_needs_saves(sched(), lens)
    # a request resumed after preemption: only reaching the end of the prompt can sample
    assert not kv_transfer.output_needs_saves(sched(cached=[("b", 0, 512)]), lens)


class FakeSaveOp:
    """Asynchronous saves: tickets in order, completed only when waited."""

    def __init__(self):
        self.issued, self.waits = 0, []

    def save(self):
        self.issued += 1

    def wait_saves(self):
        self.waits.append(self.issued)


def test_saves_of_freed_requests_are_waited(monkeypatch):
    op = FakeSaveOp()
    monkeypatch.setattr(kv_transfer, "_copy_op", op)
    monkeypatch.setattr(kv_transfer, "_unwaited_saves", set())
    kv_transfer.defer_saves(sched(new=[("a", 0, 1024)]))
    kv_transfer.wait_saves_of_freed(sched(finished={"b"}))
    assert op.waits == []  # nothing of "a" freed
    kv_transfer.wait_saves_of_freed(sched(finished={"a"}))  # aborted mid-prefill
    assert op.waits == [0] and not kv_transfer._unwaited_saves
    kv_transfer.defer_saves(sched(new=[("a", 0, 1024)]))
    kv_transfer.wait_saves_of_freed(sched(preempted={"a"}))
    assert op.waits == [0, 0]


def test_runner_waits_only_for_the_step_that_completes_a_prompt(monkeypatch):
    """Three chunks of one prompt through the runner's connector lifecycle: each step
    issues its chunk's save after the forward; only the last step's output waits, and
    the next chunk's forward is enqueued while earlier saves run."""
    from collections import deque
    from contextlib import contextmanager, nullcontext

    import vllm.forward_context
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorModelRunnerMixin,
    )

    from vllm_tt_plugin.model_runner import TTModelRunner

    op = FakeSaveOp()
    monkeypatch.setattr(kv_transfer, "_copy_op", op)
    monkeypatch.setattr(kv_transfer, "_unwaited_saves", set())
    monkeypatch.setattr(kv_transfer, "_held_recvs", [])
    events = []

    @contextmanager
    def lifecycle(scheduler_output):
        yield SimpleNamespace(finished_recving=None)
        # wait_for_save: the connector issues this step's save after the forward.
        op.save()
        events.append(("save", op.issued))

    def body(self, scheduler_output):
        events.append(("forward", op.issued))
        self._pending_samples.append(
            lambda grammar: SimpleNamespace(get_output=lambda: None)
        )
        return None

    monkeypatch.setattr(
        kv_transfer,
        "handle_preemptions",
        lambda out: events.append(("preempt", op.issued)),
    )
    monkeypatch.setattr(
        KVConnectorModelRunnerMixin, "_get_kv_connector_output", staticmethod(lifecycle)
    )
    monkeypatch.setattr(
        vllm.forward_context, "set_forward_context", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(TTModelRunner, "_execute_model_body", body)
    runner = SimpleNamespace(
        vllm_config=None,
        _pending_samples=deque(),
        requests={"a": SimpleNamespace(num_prompt_tokens=3000)},
    )
    for computed, n in ((0, 1024), (1024, 1024), (2048, 952)):
        out = (
            sched(new=[("a", 0, n)])
            if computed == 0
            else sched(cached=[("a", computed, n)])
        )
        assert TTModelRunner._execute_model_with_kv_connector(runner, out) is None
        runner._pending_samples.popleft()(None)
    assert events == [
        ("preempt", 0),
        ("forward", 0),
        ("save", 1),
        ("preempt", 1),
        ("forward", 1),
        ("save", 2),
        ("preempt", 2),
        ("forward", 2),
        ("save", 3),
    ]
    assert op.waits == [3]  # once, after the last chunk's save
    assert not kv_transfer._unwaited_saves


def test_copy_op_supplies_connector_caches(monkeypatch):
    registered = []

    class Conn:
        def register_kv_caches(self, caches):
            registered.append(caches)

        def set_host_xfer_buffer_ops(self, op):
            pass

    class OwnShadows:
        def shadow_kv_caches(self):
            return {"l0": "provider shadow"}

    import vllm.distributed.kv_transfer as kvt

    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: Conn())
    monkeypatch.setattr(kv_transfer, "_copy_op", None)
    spec = SimpleNamespace(
        num_kv_heads=1, block_size=4, head_size=8, dtype=torch.bfloat16
    )
    cfg = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["l0"])],
    )
    monkeypatch.setattr(
        kv_transfer, "_copy_op_provider", lambda m, c, cfg: OwnShadows()
    )
    kv_transfer.register_device_caches(None, {}, cfg)
    monkeypatch.setattr(kv_transfer, "_copy_op_provider", lambda m, c, cfg: object())
    kv_transfer.register_device_caches(None, {}, cfg)
    assert registered[0] == {"l0": "provider shadow"}
    assert registered[1]["l0"].shape == (
        2,
        1,
        4,
        16,
    )  # the plugin's own, without the method


def test_handle_preemptions_reaches_the_connector(monkeypatch):
    import vllm.distributed.kv_transfer as kvt

    seen = []
    conn = SimpleNamespace(handle_preemptions=seen.append)
    monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: conn)
    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: False)
    kv_transfer.handle_preemptions(SimpleNamespace(kv_connector_metadata="m"))
    assert seen == []
    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: True)
    kv_transfer.handle_preemptions(SimpleNamespace(kv_connector_metadata=None))
    kv_transfer.handle_preemptions(SimpleNamespace(kv_connector_metadata="m"))
    assert seen == ["m"]


def test_host_buffer_layout_follows_vllm_kv_cache_layout(monkeypatch):
    """A row-major host-buffer layout is selected with vLLM's own VLLM_KV_CACHE_LAYOUT,
    only when a connector is configured and the copy-op provider converts to it;
    otherwise device bytes."""
    import vllm.envs as envs
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    on = SimpleNamespace(kv_transfer_config=object())
    off = SimpleNamespace(kv_transfer_config=None)
    monkeypatch.setattr(kv_transfer, "_wire_layouts", ("LBHNC", "LBNHC"))
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    assert envs.VLLM_KV_CACHE_LAYOUT is None
    assert kv_transfer.host_buffer_wire_layout(on) is None
    assert kv_transfer.supported_kv_cache_layouts(on) == ["LBNHC"]
    for name, want in (("LBHNC", "LBHNC"), ("HND", "LBHNC"), ("LBNHC", "LBNHC")):
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", name)
        assert kv_transfer.host_buffer_wire_layout(on) == want
        assert kv_transfer.host_buffer_wire_layout(off) is None
        supported = kv_transfer.supported_kv_cache_layouts(on)
        assert supported[0] == "LBNHC" and want in supported
        # What vLLM's engine core then resolves from the worker's report.
        cfg = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_layout=None), kv_transfer_config=None
        )
        assert resolve_kv_cache_layout(cfg, [supported]).name == want
    # A provider without conversion keeps device bytes; vLLM itself refuses the
    # unsupported layout.
    monkeypatch.setattr(kv_transfer, "_wire_layouts", ())
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "LBHNC")
    assert kv_transfer.host_buffer_wire_layout(on) is None
    assert kv_transfer.supported_kv_cache_layouts(on) == ["LBNHC"]


def test_wire_layout_reaches_the_provider(monkeypatch):
    calls = []

    class Conn:
        def register_kv_caches(self, caches):
            pass

        def set_host_xfer_buffer_ops(self, op):
            pass

    import vllm.distributed.kv_transfer as kvt

    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: Conn())
    monkeypatch.setattr(kv_transfer, "_copy_op", None)
    spec = SimpleNamespace(
        num_kv_heads=1, block_size=4, head_size=8, dtype=torch.bfloat16
    )
    cfg = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["l0"])],
    )
    monkeypatch.setattr(
        kv_transfer,
        "_copy_op_provider",
        lambda m, c, cfg, **kw: calls.append(kw) or object(),
    )
    kv_transfer.register_device_caches(None, {}, cfg)
    kv_transfer.register_device_caches(None, {}, cfg, wire_layout="LBHNC")
    assert calls == [{}, {"wire_layout": "LBHNC"}]


def _device_buffer_config(monkeypatch, kv_buffer_device):
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_buffer_device=kv_buffer_device),
        parallel_config=SimpleNamespace(data_parallel_rank_local=None),
    )
    monkeypatch.setattr(kv_transfer, "_worker_config", cfg)
    return cfg


def test_device_buffers_registered_without_host_ops(monkeypatch):
    """kv_buffer_device="tt": the connector registers the provider's device tensors, gets
    no host-buffer ops, and wait_saves() reaches the handle (the forward fence)."""
    calls = []

    class Conn:
        def register_kv_caches(self, caches):
            calls.append(("register", caches))

        def set_host_xfer_buffer_ops(self, op):
            calls.append(("ops", op))

    class Handle:
        caches = {"l0": "device tensor"}

        def wait_saves(self):
            calls.append(("fence",))

    import vllm.distributed.kv_transfer as kvt

    vcfg = _device_buffer_config(monkeypatch, "tt")
    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: Conn())
    monkeypatch.setattr(kv_transfer, "_copy_op", None)
    seen = []

    def provider(mesh, caches, kv_cfg, ktc, index):
        seen.append((ktc, index))
        return Handle()

    monkeypatch.setattr(kv_transfer, "_device_buffer_provider", provider)
    kv_transfer.register_device_caches(None, {}, "KVCFG")
    assert calls == [("register", {"l0": "device tensor"})]
    assert seen == [(vcfg.kv_transfer_config, 0)]
    kv_transfer.wait_saves()
    assert calls[-1] == ("fence",)


def test_device_buffers_without_provider_refused(monkeypatch):
    import pytest
    import vllm.distributed.kv_transfer as kvt

    _device_buffer_config(monkeypatch, "tt")
    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: object())
    monkeypatch.setattr(kv_transfer, "_device_buffer_provider", None)
    with pytest.raises(RuntimeError, match="tt-kv"):
        kv_transfer.register_device_caches(None, {}, "KVCFG")


def test_nixl_platform_hooks_follow_kv_buffer_device(monkeypatch):
    from vllm_tt_plugin.platform import TTPlatform

    _device_buffer_config(monkeypatch, "cpu")
    monkeypatch.setattr(kv_transfer, "_prepare_device_buffers", None)
    assert TTPlatform.get_nixl_supported_devices() == {"tt": ("cpu",)}
    assert TTPlatform.get_nixl_memory_type() is None
    # vLLM evaluates the device table at import, possibly before any config exists.
    monkeypatch.setattr(kv_transfer, "_worker_config", None)
    prepared = []
    monkeypatch.setattr(
        kv_transfer, "_prepare_device_buffers", lambda: prepared.append(1) or True
    )
    assert TTPlatform.get_nixl_supported_devices() == {"tt": ("cpu", "tt")}
    assert prepared == [1]
    _device_buffer_config(monkeypatch, "tt")
    assert TTPlatform.get_nixl_memory_type() == "VRAM"
    # Backend not findable: "tt" is not offered, so vLLM refuses the config up front.
    monkeypatch.setattr(kv_transfer, "_prepare_device_buffers", lambda: False)
    assert TTPlatform.get_nixl_supported_devices() == {"tt": ("cpu",)}
