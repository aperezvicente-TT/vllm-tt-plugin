# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GLM53_PREFILL_ONE_REQUEST=1: at most one request's prompt chunk per step.

A step that carries two requests' prompt chunks (the tail of one prompt and the head of the next, at a user boundary)
runs both slots through one prefill program. On the 1M construction their state writes share a 32-user tile inside that
program, and the concurrent cell varied run to run (2026-10-24). With the knob, the first request given prompt tokens
owns the step's prefill, and the other waits for a later step. Decode rows and the default (off) are unchanged."""

from types import SimpleNamespace

import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.scheduler import TTScheduler


def _scheduler(one_request: bool):
    s = TTScheduler.__new__(TTScheduler)
    s._prefill_chunk_alignment = 128
    s.need_mamba_block_aligned_split = False
    s._prefill_one_request_flag = one_request
    s._step_prefill_owner = None
    return s


def _request(rid, prompt_tokens, computed=0, generated=0):
    return SimpleNamespace(request_id=rid, num_prompt_tokens=prompt_tokens, num_tokens=prompt_tokens + generated,
                           num_computed_tokens=computed, mm_features=[], has_encoder_inputs=False)  # fmt: skip


def test_the_second_request_prompt_chunk_waits_for_a_later_step():
    s = _scheduler(True)
    a, b = _request("a", 131072, computed=129024), _request("b", 131072)
    assert s._mamba_block_aligned_split(a, 1920) == 1920  # a's tail owns this step
    assert s._mamba_block_aligned_split(b, 128) == 0  # b's head would share the program: it waits
    assert s._mamba_block_aligned_split(a, 1920) == 1920  # the owner may be sized again in the same step
    assert s.prefill_one_request_deferred == 1
    s._step_prefill_owner = None  # the next step (schedule() resets it)
    assert s._mamba_block_aligned_split(b, 2048) == 2048


def test_decode_rows_are_untouched():
    s = _scheduler(True)
    assert s._mamba_block_aligned_split(_request("a", 131072), 2048) == 2048
    decoding = _request("c", 128, computed=132, generated=5)  # num_computed = num_tokens - 1
    assert s._mamba_block_aligned_split(decoding, 1) == 1


def test_off_by_default_both_requests_share_the_step(monkeypatch):
    monkeypatch.delenv("GLM53_PREFILL_ONE_REQUEST", raising=False)
    s = _scheduler(True)
    del s._prefill_one_request_flag  # read from the env (unset: off)
    a, b = _request("a", 131072, computed=129024), _request("b", 131072)
    assert s._mamba_block_aligned_split(a, 1920) == 1920
    assert s._mamba_block_aligned_split(b, 128) == 128
    assert getattr(s, "prefill_one_request_deferred", 0) == 0


def test_the_env_switches_it_on(monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_ONE_REQUEST", "1")
    s = _scheduler(True)
    del s._prefill_one_request_flag
    assert s._prefill_one_request() is True


def test_schedule_starts_every_step_without_an_owner():
    s = _scheduler(True)
    s._step_prefill_owner = "a"  # left over from the previous step
    seen = []
    s._take_preempted_requests_with_pending_outputs = lambda: []
    s._schedule_without_pending_output_resumes = lambda throttle=False: seen.append(s._step_prefill_owner) or "out"
    assert s.schedule() == "out" and seen == [None]
