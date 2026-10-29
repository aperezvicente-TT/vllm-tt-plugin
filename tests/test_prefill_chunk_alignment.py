# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Partial prefill chunks end on the configured alignment grid.

A request admitted on a step's leftover budget would otherwise start every
later chunk on an odd offset, which the model realigns and pads up to the next
power of two. The scheduler rounds the chunk end instead, through vLLM's own
block-aligned-split hook, before any block is allocated.
"""

from types import SimpleNamespace

import pytest
import vllm  # noqa: F401  (resolve the platform plugin before importing ours)

from vllm_tt_plugin.config import get_tt_prefill_chunk_alignment
from vllm_tt_plugin.scheduler import (
    TTScheduler,
    _effective_prefill_step_limit,
    _validate_prefill_chunk_alignment,
)


def _scheduler(alignment=128, *, base_mamba=False):
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler._prefill_chunk_alignment = alignment
    scheduler.need_mamba_block_aligned_split = base_mamba
    return scheduler


def _request(prompt_tokens, computed=0, mm=False, generated=0):
    return SimpleNamespace(
        num_prompt_tokens=prompt_tokens,
        num_tokens=prompt_tokens + generated,
        num_computed_tokens=computed,
        mm_features=[object()] if mm else [],
        has_encoder_inputs=mm,
    )


def test_leftover_budget_first_chunk_is_rounded_down_to_the_grid():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(32612), 157) == 128
    assert s._mamba_block_aligned_split(_request(32612), 4096) == 4096


def test_a_new_request_that_cannot_reach_the_grid_waits_a_step():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(32612), 100) == 0


def test_a_running_request_waits_for_the_grid_too():
    s = _scheduler()
    # A continuation after a 640-token hit, given this step's 9-token leftover
    # (2026-10-23 gx2c5, 32 live rows): a 9-token chunk would end at 649 and
    # start the next chunk off the model's 64-token latent page -> it waits.
    assert s._mamba_block_aligned_split(_request(668, computed=640), 9) == 0
    # A chunk that reaches the next boundary is rounded down to it ...
    assert s._mamba_block_aligned_split(_request(1200, computed=640), 200) == 128
    # ... and the chunk that finishes the prompt is left alone.
    assert s._mamba_block_aligned_split(_request(668, computed=640), 28) == 28
    # Started unaligned (a legacy cached prefix): the first aligned end is 4224,
    # so a 4096 chunk from 157 shrinks to 4067; a chunk too short to reach it waits.
    assert (
        s._mamba_block_aligned_split(_request(32612, computed=157), 4096) == 4096 - 29
    )
    assert s._mamba_block_aligned_split(_request(32612, computed=157), 50) == 0


def test_a_running_request_under_a_permanent_cap_still_makes_progress():
    s = _scheduler()
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4096, long_prefill_token_threshold=64
    )
    assert s._mamba_block_aligned_split(_request(1024, computed=640), 64) == 64


def test_a_preempted_request_recompute_stays_on_the_grid():
    s = _scheduler()
    # Preempted after generating 320 tokens: the re-prefill covers prompt + output
    # (621 tokens). A 300-token chunk from 256 crosses the prompt end (301) but not
    # the known tokens, so it is rounded to 512, not left to end at 556 ...
    assert s._mamba_block_aligned_split(_request(301, computed=256, generated=320), 300) == 256
    # ... and the chunk that reaches every known token is the last one: left alone.
    assert s._mamba_block_aligned_split(_request(301, computed=512, generated=320), 109) == 109


def test_the_chunk_that_finishes_the_prompt_is_left_alone():
    s = _scheduler()
    assert s._mamba_block_aligned_split(_request(32612, computed=28672), 3940) == 3940
    assert s._mamba_block_aligned_split(_request(300), 300) == 300


def test_decode_and_multimodal_requests_are_untouched():
    s = _scheduler()
    decode = SimpleNamespace(
        num_prompt_tokens=300, num_tokens=310, num_computed_tokens=309, mm_features=[]
    )
    assert s._mamba_block_aligned_split(decode, 1) == 1
    assert s._mamba_block_aligned_split(_request(32612, mm=True), 157) == 157


def test_zero_disables_and_the_hook_is_only_armed_when_needed():
    off = _scheduler(alignment=0)
    assert off.need_mamba_block_aligned_split is False
    assert off._mamba_block_aligned_split(_request(32612), 157) == 157
    on = _scheduler()
    assert on.need_mamba_block_aligned_split is True
    # The base scheduler's own flag still shows through.
    mamba = _scheduler(alignment=0, base_mamba=True)
    assert mamba.need_mamba_block_aligned_split is True


def test_config_accessor_defaults_and_validates():
    cfg = SimpleNamespace(additional_config={"tt": {}})
    assert get_tt_prefill_chunk_alignment(cfg) == 128
    cfg = SimpleNamespace(additional_config={"tt": {"prefill_chunk_alignment": 0}})
    assert get_tt_prefill_chunk_alignment(cfg) == 0
    cfg = SimpleNamespace(additional_config={"tt": {"prefill_chunk_alignment": -8}})
    with pytest.raises(ValueError):
        get_tt_prefill_chunk_alignment(cfg)


def test_a_permanent_cap_below_the_grid_still_makes_progress():
    # long_prefill_token_threshold=64 caps every step at 64 tokens: deferring would
    # never converge, so the new request keeps its 64-token chunk ...
    s = _scheduler()
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4096, long_prefill_token_threshold=64
    )
    assert s._mamba_block_aligned_split(_request(1024), 64) == 64
    # ... while a chunk cut short by this step's leftover budget still waits.
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4096, long_prefill_token_threshold=0
    )
    assert s._mamba_block_aligned_split(_request(1024), 100) == 0


def test_init_rejects_caps_below_the_alignment():
    ok = SimpleNamespace(max_num_batched_tokens=4096, long_prefill_token_threshold=4096)
    _validate_prefill_chunk_alignment(128, ok)
    _validate_prefill_chunk_alignment(
        128,
        SimpleNamespace(max_num_batched_tokens=4096, long_prefill_token_threshold=0),
    )
    with pytest.raises(ValueError, match="long_prefill_token_threshold=64"):
        _validate_prefill_chunk_alignment(
            128,
            SimpleNamespace(
                max_num_batched_tokens=4096, long_prefill_token_threshold=64
            ),
        )
    with pytest.raises(ValueError, match="max_num_batched_tokens=64"):
        _validate_prefill_chunk_alignment(
            128,
            SimpleNamespace(max_num_batched_tokens=64, long_prefill_token_threshold=0),
        )


def test_max_num_scheduled_tokens_is_the_step_limit_when_set():
    # vLLM schedules with max_num_scheduled_tokens when it is configured; the
    # alignment cap and its validation must see the same limit (review on #169).
    cfg = SimpleNamespace(
        max_num_batched_tokens=4096,
        max_num_scheduled_tokens=256,
        long_prefill_token_threshold=0,
    )
    assert _effective_prefill_step_limit(cfg) == (256, "max_num_scheduled_tokens")
    s = _scheduler()
    s.scheduler_config = cfg
    assert s._full_step_prefill_cap() == 256
    # unset -> max_num_batched_tokens, as before
    cfg.max_num_scheduled_tokens = None
    assert _effective_prefill_step_limit(cfg) == (4096, "max_num_batched_tokens")
    # a positive long_prefill_token_threshold caps the resolved limit
    cfg.max_num_scheduled_tokens = 256
    cfg.long_prefill_token_threshold = 100
    assert _effective_prefill_step_limit(cfg) == (100, "long_prefill_token_threshold")
    assert s._full_step_prefill_cap() == 100


def test_init_rejects_a_scheduled_token_limit_below_the_alignment():
    # The reviewer's trigger: batched 4096, scheduled 64, threshold 0, alignment
    # 128, an uncached 1,024-token prompt. Every step starts with a 64-token
    # budget, the aligned split returns 0 forever and the waiting queue stalls.
    with pytest.raises(ValueError, match="max_num_scheduled_tokens=64"):
        _validate_prefill_chunk_alignment(
            128,
            SimpleNamespace(
                max_num_batched_tokens=4096,
                max_num_scheduled_tokens=64,
                long_prefill_token_threshold=0,
            ),
        )
    # and the scheduler's own cap agrees, so a 64-token proposal is a full step
    s = _scheduler()
    s.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4096,
        max_num_scheduled_tokens=64,
        long_prefill_token_threshold=0,
    )
    assert s._full_step_prefill_cap() == 64
    assert s._mamba_block_aligned_split(_request(1024), 64) == 64


# ---------------------------------------------------------- tt.plan_cut (dev-only diagnostic, program 12:18:35Z)
def _hinted(prompt_tokens, computed=0, cut=None, cold=False):
    r = _request(prompt_tokens, computed=computed)
    kv = {}
    if cut is not None:
        kv["tt.plan_cut"] = cut
    if cold:
        kv["tt.cold"] = True
    r.kv_transfer_params = kv or None
    return r


def _run_budget_scheduler(base=2048, run=4):
    s = _scheduler()
    s.max_num_scheduled_tokens = base
    s._prefill_chunk_run = run
    s.waiting = []
    s.skipped_waiting = []
    return s


def test_plan_cut_reproduces_the_hit_plan_cold():
    """{"tt.cold": true, "tt.plan_cut": 7936} on a cold golden 32K: (0,2048) (2048,7936) (7936,9984) (9984,18176)
    (18176,26368) (26368,31872) (31872,32000) -- the 7,936-token hit's continuation plan, without the hit"""
    s = _run_budget_scheduler()
    s._kda_checkpoint_store = object()  # the KDA checkpoint policy (the last-boundary cut) as served
    s._checkpoint_grid = 128
    plan, start = [], 0
    while start < 32000:
        r = _hinted(32000, computed=start, cut=7936, cold=True)
        budget = s.max_num_scheduled_tokens if start == 0 else s._prefill_chunk_run_budget([r], [])
        n = s._mamba_block_aligned_split(r, min(budget, 32000 - start))
        assert n > 0, (start, budget)
        plan.append((start, start + n))
        start += n
    assert plan == [(0, 2048), (2048, 7936), (7936, 9984), (9984, 18176), (18176, 26368), (26368, 31872),
                    (31872, 32000)]  # fmt: skip
    # without the hint: the cold plan
    plan, start = [], 0
    while start < 32000:
        r = _request(32000, computed=start)
        budget = s.max_num_scheduled_tokens if start == 0 else s._prefill_chunk_run_budget([r], [])
        n = s._mamba_block_aligned_split(r, min(budget, 32000 - start))
        plan.append((start, start + n))
        start += n
    assert plan[:3] == [(0, 2048), (2048, 10240), (10240, 18432)]


def test_plan_cut_is_inert_when_absent_or_out_of_range():
    s = _run_budget_scheduler()
    for cut in (None, 0, 32000, 40000, "x"):
        r = _hinted(32000, computed=2048, cut=cut)
        assert s._mamba_block_aligned_split(r, 8192) == 8192
        assert s._prefill_chunk_run_budget([r], []) == 8192
    r = _hinted(32000, computed=2048, cut=7936)
    r.kv_transfer_params = None
    r.sampling_params = SimpleNamespace(extra_args={"kv_transfer_params": {"tt.plan_cut": 7936}})
    assert s._mamba_block_aligned_split(r, 8192) == 5888  # the hint through sampling_params.extra_args too


# ---------------------------------------------------------- tt.chunk_cap (dev-only diagnostic: buckets <= N)
def test_chunk_cap_keeps_every_prefill_chunk_at_or_below_the_cap():
    """{"tt.cold": true, "tt.chunk_cap": 512} on G[:16100]: every chunk <= 512 tokens (no 2,048 bucket), each ending on
    the 128 grid except the prompt's tail; without the hint the run budget schedules 2,048+ token steps"""
    s = _run_budget_scheduler()
    s._kda_checkpoint_store = object()
    s._checkpoint_grid = 128
    plan, start = [], 0
    while start < 16100:
        r = _request(16100, computed=start)
        r.kv_transfer_params = {"tt.cold": True, "tt.chunk_cap": 512}
        budget = s.max_num_scheduled_tokens if start == 0 else s._prefill_chunk_run_budget([r], [])
        n = s._mamba_block_aligned_split(r, min(budget, 16100 - start))
        assert n > 0, (start, budget)
        plan.append((start, start + n))
        start += n
    assert max(e - b for b, e in plan) == 512
    assert all(e % 128 == 0 for _, e in plan[:-1]) and plan[-1][1] == 16100
    assert plan[-2][1] == 16000  # the KDA checkpoint boundary still cuts the last aligned chunk
    r = _request(16100, computed=2048)
    assert s._mamba_block_aligned_split(r, s._prefill_chunk_run_budget([r], [])) > 512


@pytest.mark.parametrize("knob,value", [("tt.chunk_cap", v) for v in (1, 64, 100, 127, 200, 513)]
                         + [("tt.plan_cut", v) for v in (1, 100, 200, 2049, 7937)])  # fmt: skip
@pytest.mark.parametrize("computed", [0, 2048])  # a waiting (new) request and a running one (a continuation)
def test_an_off_grid_dev_hint_is_ignored_and_the_prefill_always_progresses(knob, value, computed):
    """an off-grid tt.chunk_cap / tt.plan_cut (any API client can send one) cannot park a request: the split ignores it,
    every step hands the request a non-empty aligned chunk, and the prefill reaches the prompt end exactly as without
    the hint"""
    s = _run_budget_scheduler()

    def plan(kv):
        out, start = [], computed
        while start < 32000:
            r = _request(32000, computed=start)
            r.kv_transfer_params = kv
            budget = s.max_num_scheduled_tokens if start == 0 else s._prefill_chunk_run_budget([r], [])
            n = s._mamba_block_aligned_split(r, min(budget, 32000 - start))
            assert n > 0, (knob, value, start, budget)
            assert start + n == 32000 or (start + n) % 128 == 0
            out.append((start, start + n))
            start += n
        return out

    assert plan({"tt.cold": True, knob: value}) == plan({"tt.cold": True})


def test_on_grid_dev_hints_still_apply():
    s = _run_budget_scheduler()
    r = _request(32000, computed=2048)
    r.kv_transfer_params = {"tt.chunk_cap": 256}
    assert s._mamba_block_aligned_split(r, 8192) == 256
    r = _hinted(32000, computed=2048, cut=2304)
    assert s._mamba_block_aligned_split(r, 8192) == 256


def test_chunk_cap_is_inert_when_absent_or_invalid():
    s = _run_budget_scheduler()
    for cap in (None, 0, -512, "x"):
        r = _request(32000, computed=2048)
        r.kv_transfer_params = None if cap is None else {"tt.chunk_cap": cap}
        assert s._mamba_block_aligned_split(r, 8192) == 8192
    r = _request(32000, computed=2048)
    r.sampling_params = SimpleNamespace(extra_args={"kv_transfer_params": {"tt.chunk_cap": 512}})
    r.kv_transfer_params = None
    assert s._mamba_block_aligned_split(r, 8192) == 512  # through sampling_params.extra_args too


# ---------------------------------------------------------- plan-aligned resume cuts (GLM53_PF_RESUME_GRID)
def _sub_chunks(plan, grid=2048):
    """the model's sub-chunks of a scheduler plan: each chunk is split into `grid`-token pieces from its own start"""
    return [(s + o, min(s + o + grid, e)) for s, e in plan for o in range(0, e - s, grid)]


def _plan(s, n, start, *, kv=None):
    out = []
    while start < n:
        r = _request(n, computed=start)
        r.kv_transfer_params = kv
        budget = s.max_num_scheduled_tokens if (start == 0 or not out) else s._prefill_chunk_run_budget([r], [])
        k = s._mamba_block_aligned_split(r, min(budget, n - start))
        assert k > 0, (start, budget)
        out.append((start, start + k))
        start += k
    return out


def _resume_scheduler(grid):
    s = _run_budget_scheduler()
    s._kda_checkpoint_store = object()  # the KDA checkpoint policy (the last-boundary cut) as served
    s._checkpoint_grid = 128
    s._resume_grid = grid
    return s


def test_a_resumed_continuation_runs_the_cold_sub_chunks_after_its_first_unit():
    """G[:32000] resumed at 16,000 (a hit): the first chunk ends at the next 2,048 boundary (16,384); from there on its
    sub-chunks are exactly the cold plan's. Without the grid every sub-chunk was offset (16,000, 18,048, ...)"""
    on, off = _resume_scheduler(2048), _resume_scheduler(0)
    cold = _sub_chunks(_plan(on, 32000, 0))
    assert cold == _sub_chunks(_plan(off, 32000, 0))  # a cold prompt never starts off the grid: unchanged
    hit = _plan(on, 32000, 16000)
    assert hit[0] == (16000, 16384)
    assert [c for c in _sub_chunks(hit) if c[0] >= 16384] == [c for c in cold if c[0] >= 16384]
    old = _sub_chunks(_plan(off, 32000, 16000))
    body = {c for c in cold if 16000 <= c[0] and c[1] <= 31872}  # before the checkpoint cut's tail chunk
    assert old[0] == (16000, 18048) and not set(old) & body


def test_resume_cuts_leave_aligned_and_short_continuations_alone():
    s = _resume_scheduler(2048)
    assert _plan(s, 32000, 16384)[0][0] == 16384 and _plan(s, 32000, 16384)[0][1] % 2048 in (0, 32000 % 2048)
    assert _plan(s, 16100, 16000) == [(16000, 16100)]  # the tail chunk after a checkpoint cut: already ends the prompt
    assert _plan(s, 8600, 7936)[0] == (7936, 8192)
    assert _plan(s, 8600, 7936)[-1][1] == 8600



def test_a_resume_grid_off_the_alignment_grid_never_starves_a_continuation():
    """GLM53_PF_RESUME_GRID=auto takes the step budget as the grid. A budget off the 128-token alignment grid (150) cut a
    chunk starting at 128 to end at 150, which the alignment rounded back to 128: 0 tokens on every step, a livelock (the
    host engine tests' preemption recompute, review round 29). The cut now applies only when it ends on the alignment
    grid; the served 2,048 budget keeps it."""
    s = _resume_scheduler(150)
    k = s._mamba_block_aligned_split(_request(400, computed=128), 150)
    assert k > 0 and (128 + k) % 128 == 0, k
    assert _plan(_resume_scheduler(2048), 32000, 16000)[0] == (16000, 16384)

# ---------------------------------------------------------- no KDA-checkpoint tail split for a request that stores nothing
def _tail_scheduler(skip):
    s = _run_budget_scheduler()
    s._kda_checkpoint_store = object()
    s._checkpoint_grid = 128
    s._tail_split_skip = skip
    return s


def test_a_request_that_stores_nothing_runs_one_chunk_without_the_tail_split(monkeypatch):
    """GLM53_PF_TAIL_SPLIT_SKIP=1: a cold (lmcache.skip_save) 2,048-token prompt runs ONE (0, 2048) chunk instead of
    1,920 + 128; the same prompt that stores keeps the checkpoint cut; "0" restores the cut for all"""
    monkeypatch.delenv("GLM53_LMCACHE_MIN_STORE_TOKENS", raising=False)
    on, off = _tail_scheduler("1"), _tail_scheduler("0")
    cold = {"tt.cold": True, "lmcache.skip_save": True}
    assert _plan(on, 2048, 0, kv=cold) == [(0, 2048)]
    assert _plan(on, 2048, 0, kv=None) == [(0, 1920), (1920, 2048)]
    assert _plan(off, 2048, 0, kv=cold) == [(0, 1920), (1920, 2048)]
    r = _request(2048, computed=0)  # the mark where the connector writes it: sampling_params.extra_args
    r.kv_transfer_params = None
    r.sampling_params = SimpleNamespace(extra_args={"kv_transfer_params": {"lmcache.skip_save": True}})
    assert on._mamba_block_aligned_split(r, 2048) == 2048


def test_floor_mode_skips_the_tail_split_only_under_the_store_floor(monkeypatch):
    """GLM53_PF_TAIL_SPLIT_SKIP=floor: a 1,000-token prompt under the 1,024 floor runs one chunk (nobody stores it); a
    2,048-token tt.cold probe keeps the 1,920 + 128 plan a first-time request of that length runs"""
    monkeypatch.setenv("GLM53_LMCACHE_MIN_STORE_TOKENS", "1024")
    fl, off = _tail_scheduler("floor"), _tail_scheduler("0")
    cold = {"tt.cold": True, "lmcache.skip_save": True}
    assert _plan(fl, 1000, 0, kv=None) == [(0, 1000)]
    assert _plan(off, 1000, 0, kv=None) == [(0, 896), (896, 1000)]
    assert _plan(fl, 2048, 0, kv=cold) == [(0, 1920), (1920, 2048)]
    assert _plan(_tail_scheduler("1"), 2048, 0, kv=cold) == [(0, 2048)]


def test_the_prefix_oracle_keeps_its_boundary_even_when_nothing_is_stored(monkeypatch):
    monkeypatch.delenv("GLM53_LMCACHE_MIN_STORE_TOKENS", raising=False)
    s = _tail_scheduler("1")
    s._prefix_oracle = SimpleNamespace(boundary=128)
    assert _plan(s, 2048, 0, kv={"lmcache.skip_save": True}) == [(0, 1920), (1920, 2048)]
