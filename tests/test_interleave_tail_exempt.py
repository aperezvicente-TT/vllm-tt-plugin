# SPDX-License-Identifier: Apache-2.0
"""GLM53_INTERLEAVE_TAIL_EXEMPT=1 (program 2026-10-28T06:51:35Z (2)): a request's last prefill chunk of at most
GLM53_INTERLEAVE_TAIL_EXEMPT_TOKENS (default 128) runs in the same prefill run as its predecessor, without the decode
interleave's D decode-only steps before it. Under k4d5p's P1/D24 the storing tail split's 128-token last chunk of a 32K
arrival waited one whole D window (3.7 s with 8 decoders)."""

from types import SimpleNamespace

from vllm.v1.core.sched.output import SchedulerOutput

from .test_decode_interleave import _scheduler


def _record_phases(scheduler, monkeypatch, steps):
    """test_decode_interleave's driver, with a prefill step that runs the partial prefill's next chunk (2,048 tokens or
    its remainder): after its last chunk it is a decoder like the others"""
    phases = []

    def step(kind):
        def stub():
            phases.append(kind)
            if kind == "P":
                for r in scheduler.running:
                    n, done = getattr(r, "num_prompt_tokens", None), getattr(r, "num_computed_tokens", None)
                    if getattr(r, "is_prefill_chunk", False) and n is not None:
                        r.num_computed_tokens = min(n, done + 2048)
                        r.is_prefill_chunk = r.num_computed_tokens < n
            out = SchedulerOutput.make_empty()
            out.num_scheduled_tokens = {"r": 1}
            out.total_num_scheduled_tokens = 1
            return out

        return stub

    monkeypatch.setattr(scheduler, "_schedule_prefill_only", step("P"))
    monkeypatch.setattr(scheduler, "_schedule_decode_only", step("D"))
    for _ in range(steps):
        scheduler.schedule()
    return "".join(phases)


def _part(left, n=32768):
    return SimpleNamespace(is_prefill_chunk=True, num_prompt_tokens=n, num_computed_tokens=n - left)


def _after_one_chunk(scheduler):
    """the policy right after the predecessor chunk ran: its P1 run is at the bound"""
    scheduler._decode_interleave.record_step(is_decode=False, prefill_pending=True)
    return scheduler


def test_the_short_last_chunk_runs_at_once_then_the_d_window(monkeypatch):
    monkeypatch.setenv("GLM53_INTERLEAVE_TAIL_EXEMPT", "1")
    s = _after_one_chunk(_scheduler(running=[_part(128), SimpleNamespace(is_prefill_chunk=False)],
                                    decode_interleave_prefill_steps=1, decode_interleave_decode_steps=3))
    assert _record_phases(s, monkeypatch, 1) == "P"  # the tail now, in its predecessor's prefill run
    assert s.interleave_tail_exempts == 1 and not s.running[0].is_prefill_chunk
    # the D window still follows: the next pending prefill (another prompt) waits for it
    assert s._decode_interleave.wants_decode_step(has_pending_prefill=True, has_running_decode=True)


def test_off_by_default_the_tail_waits_for_the_d_window(monkeypatch):
    monkeypatch.delenv("GLM53_INTERLEAVE_TAIL_EXEMPT", raising=False)
    s = _after_one_chunk(_scheduler(running=[_part(128), SimpleNamespace(is_prefill_chunk=False)],
                                    decode_interleave_prefill_steps=1, decode_interleave_decode_steps=3))
    assert _record_phases(s, monkeypatch, 4) == "DDDP"
    assert not getattr(s, "interleave_tail_exempts", 0)


def test_a_longer_remainder_or_a_waiting_prompt_is_not_a_tail(monkeypatch):
    monkeypatch.setenv("GLM53_INTERLEAVE_TAIL_EXEMPT", "1")
    s = _after_one_chunk(_scheduler(running=[_part(1920), SimpleNamespace(is_prefill_chunk=False)],
                                    decode_interleave_prefill_steps=1, decode_interleave_decode_steps=2))
    assert _record_phases(s, monkeypatch, 3) == "DDP"
    s = _after_one_chunk(_scheduler(running=[_part(128), SimpleNamespace(is_prefill_chunk=False)], waiting=1,
                                    decode_interleave_prefill_steps=1, decode_interleave_decode_steps=2))
    assert _record_phases(s, monkeypatch, 3) == "DDP"


def test_the_bound_is_a_knob(monkeypatch):
    monkeypatch.setenv("GLM53_INTERLEAVE_TAIL_EXEMPT", "1")
    monkeypatch.setenv("GLM53_INTERLEAVE_TAIL_EXEMPT_TOKENS", "512")
    s = _after_one_chunk(_scheduler(running=[_part(400), SimpleNamespace(is_prefill_chunk=False)],
                                    decode_interleave_prefill_steps=1, decode_interleave_decode_steps=2))
    assert _record_phases(s, monkeypatch, 1) == "P" and s.interleave_tail_exempts == 1


def test_a_request_without_token_counts_is_never_exempt(monkeypatch):
    monkeypatch.setenv("GLM53_INTERLEAVE_TAIL_EXEMPT", "1")
    s = _after_one_chunk(_scheduler(running=[SimpleNamespace(is_prefill_chunk=True), SimpleNamespace(is_prefill_chunk=False)],
                                    decode_interleave_prefill_steps=1, decode_interleave_decode_steps=2))
    assert _record_phases(s, monkeypatch, 3) == "DDP"
