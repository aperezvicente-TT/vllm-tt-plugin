# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GLM53_MIXED_DECODE_FOLLOW / GLM53_SCHED_CTL_FILE (Stage 12, program 2026-10-26T21:47:35Z arm C): with mixed
steps on, K decode-only steps follow every mixed step while requests decode, so a decoder gets K + 1 tokens per prefill
chunk instead of one; the prefill still finishes (the follow-up counter always falls); the control file switches K on a
running scheduler. Host-only: the real TTScheduler over the real vLLM base scheduler (test_mixed_steps' helpers)."""

from __future__ import annotations

import os

import pytest

from tests.test_mixed_steps import BUDGET, _commit, _decoding, _request, _scheduler
from vllm_tt_plugin.config import MIXED_DECODE_FOLLOW_ENV, SCHED_CTL_FILE_ENV, get_tt_mixed_decode_follow
from vllm_tt_plugin.scheduler import is_tt_mixed_step


@pytest.fixture(autouse=True)
def _no_follow_knobs(monkeypatch):
    monkeypatch.delenv(MIXED_DECODE_FOLLOW_ENV, raising=False)
    monkeypatch.delenv(SCHED_CTL_FILE_ENV, raising=False)


def _settle(scheduler):
    """one ordinary decode step with no prefill pending: the interleave counters and the follow-up budget start clean"""
    out = scheduler.schedule()
    assert not is_tt_mixed_step(out) and all(n == 1 for n in out.num_scheduled_tokens.values())
    _commit(scheduler, out)


def _kinds(scheduler, prompt_id, max_steps=40):
    """run until the prompt finishes its prefill: per step 'M' (mixed), 'D' (decode-only) or 'P' (prefill-only)"""
    kinds = []
    for _ in range(max_steps):
        req = scheduler.requests.get(prompt_id)
        if kinds and (req is None or (not req.is_prefill_chunk and req.num_output_tokens > 0)):
            break  # its prefill is done (a max_tokens=1 prompt has finished and left the scheduler)
        out = scheduler.schedule()
        n = out.num_scheduled_tokens
        kinds.append("M" if is_tt_mixed_step(out) else ("P" if n.get(prompt_id) else "D"))
        _commit(scheduler, out)
    return "".join(kinds)


def test_default_is_unchanged_every_step_mixed():
    scheduler = _scheduler()
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 3 * BUDGET))
    assert _kinds(scheduler, "p") == "MMM"


def test_k_decode_only_steps_follow_each_mixed_step_and_the_prefill_finishes(monkeypatch):
    monkeypatch.setenv(MIXED_DECODE_FOLLOW_ENV, "2")
    scheduler = _scheduler()
    _decoding(scheduler, "d1")
    _decoding(scheduler, "d2")
    _settle(scheduler)  # (d2's own admission was a mixed step: a settled decode step clears its follow-ups)
    before = scheduler.requests["d1"].num_output_tokens
    scheduler.add_request(_request("p", 3 * BUDGET))
    kinds = _kinds(scheduler, "p")
    # each of p's chunks is a mixed step followed by K decode-only steps (the last chunk's follow-ups run after it)
    assert kinds == "MDDMDDM", kinds
    assert scheduler.requests["d1"].num_output_tokens - before == len(kinds)  # the decoders advanced on every step
    assert scheduler.mixed_follow_steps == 4


def test_the_control_file_switches_k_on_a_running_scheduler(monkeypatch, tmp_path):
    ctl = tmp_path / "sched.ctl"
    monkeypatch.setenv(SCHED_CTL_FILE_ENV, str(ctl))
    monkeypatch.setenv(MIXED_DECODE_FOLLOW_ENV, "0")
    scheduler = _scheduler()
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p1", 2 * BUDGET, max_tokens=1))
    # (the prompts emit one token and finish, so the 4 seats never fill)
    assert _kinds(scheduler, "p1") == "MM"  # no file: the env's 0
    ctl.write_text("# A/B\nmixed_decode_follow=1\n")
    _settle(scheduler)
    scheduler.add_request(_request("p2", 2 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p2") == "MDM"
    ctl.write_text("mixed_decode_follow=0\n")
    os.utime(ctl, ns=(os.stat(ctl).st_mtime_ns + 10**9,) * 2)  # a new mtime even within the clock's granularity
    _settle(scheduler)
    scheduler.add_request(_request("p3", 2 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p3") == "MM"
    ctl.unlink()  # removed: back to the env value
    _settle(scheduler)
    scheduler.add_request(_request("p4", 2 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p4") == "MM"


def test_no_decoders_no_follow_steps(monkeypatch):
    monkeypatch.setenv(MIXED_DECODE_FOLLOW_ENV, "3")
    scheduler = _scheduler()
    scheduler.add_request(_request("p", 3 * BUDGET))
    kinds = _kinds(scheduler, "p")
    assert set(kinds) == {"P"} and len(kinds) == 3, kinds
    assert scheduler.mixed_follow_steps == 0


@pytest.mark.parametrize("bad", ["-1", "x"])
def test_a_bad_env_value_is_refused(monkeypatch, bad):
    monkeypatch.setenv(MIXED_DECODE_FOLLOW_ENV, bad)
    with pytest.raises(ValueError, match="GLM53_MIXED_DECODE_FOLLOW"):
        get_tt_mixed_decode_follow()


# --- GLM53_DECODE_INTERLEAVE_FILE (program 2026-10-26T21:58:59Z (a)): the decode interleave + mixed steps at run
# time from the control file, so one resident runs A (P2/D1) / B (P1/D1) / C (P1/D3) / D (P1/D4) and the mixed arms


def _ctl(path, text):
    path.write_text(text)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9 * (1 + int(st.st_mtime_ns % 7))))  # a fresh mtime


def test_the_file_sets_the_decode_interleave_at_run_time(monkeypatch, tmp_path):
    ctl = tmp_path / "interleave.ctl"
    monkeypatch.setenv(SCHED_CTL_FILE_ENV, str(ctl))
    scheduler = _scheduler(mixed=False)  # interleave mode (start-up default P2/D1)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p1", 4 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p1") == "PPDPP", "the start-up P2/D1"
    _ctl(ctl, "decode_interleave_prefill_steps=1\ndecode_interleave_decode_steps=3\n")
    _settle(scheduler)
    scheduler.add_request(_request("p2", 4 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p2") == "PDDDPDDDPDDDP", "P1/D3 from the file"
    _ctl(ctl, "decode_interleave_enabled=0\n")
    _settle(scheduler)
    scheduler.add_request(_request("p3", 3 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p3") == "PPP", "interleave off"
    ctl.unlink()
    _settle(scheduler)
    scheduler.add_request(_request("p4", 4 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p4") == "PPDPP", "file gone: back to the start-up values"


def test_the_file_turns_mixed_steps_off_and_back_on(monkeypatch, tmp_path):
    ctl = tmp_path / "interleave.ctl"
    monkeypatch.setenv(SCHED_CTL_FILE_ENV, str(ctl))
    scheduler = _scheduler()  # mixed on at start-up
    _decoding(scheduler, "d1")
    _ctl(ctl, "mixed_steps=0\ndecode_interleave_prefill_steps=1\ndecode_interleave_decode_steps=3\n")
    _settle(scheduler)
    scheduler.add_request(_request("p1", 2 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p1") == "PDDDP", "mixed off: P1/D3"
    _ctl(ctl, "mixed_steps=1\nmixed_decode_follow=3\n")
    _settle(scheduler)
    scheduler.add_request(_request("p2", 2 * BUDGET, max_tokens=1))
    assert _kinds(scheduler, "p2") == "MDDDM", "mixed back on with K=3 follow-ups"


def test_mixed_steps_cannot_be_turned_on_when_they_were_off_at_start_up(monkeypatch, tmp_path):
    ctl = tmp_path / "interleave.ctl"
    monkeypatch.setenv(SCHED_CTL_FILE_ENV, str(ctl))
    _ctl(ctl, "mixed_steps=1\n")
    scheduler = _scheduler(mixed=False)
    _decoding(scheduler, "d1")
    _settle(scheduler)
    scheduler.add_request(_request("p", 2 * BUDGET, max_tokens=1))
    assert "M" not in _kinds(scheduler, "p")


# --- tt.prefill_segments (segmented traces, doc/ttnn/14b-persistent-stack/segtrace/runner_design.md item 11): S of the
# segmented form from the tt config, overridable at run time by the control file's prefill_segments key


def test_prefill_segments_default_config_and_bounds():
    from vllm_tt_plugin.config import PREFILL_SEGMENTS_MAX

    assert _scheduler()._prefill_segments() == 1  # off by default: one replay per chunk
    assert _scheduler(prefill_segments=4)._prefill_segments() == 4
    for bad in (0, PREFILL_SEGMENTS_MAX + 1, True, "3"):
        with pytest.raises(ValueError, match="prefill_segments"):
            _scheduler(prefill_segments=bad)


def test_the_control_file_overrides_prefill_segments_at_run_time(monkeypatch, tmp_path):
    ctl = tmp_path / "sched.ctl"
    monkeypatch.setenv(SCHED_CTL_FILE_ENV, str(ctl))
    scheduler = _scheduler(prefill_segments=4)
    assert scheduler._prefill_segments() == 4  # no file: the config's S
    stamp = [10**18]

    def _ctl(path, text):  # strictly increasing mtimes (several rewrites within one clock tick)
        path.write_text(text)
        stamp[0] += 10**9
        os.utime(path, ns=(stamp[0], stamp[0]))

    _ctl(ctl, "prefill_segments=3\n")
    assert scheduler._prefill_segments() == 3
    _ctl(ctl, "prefill_segments=99\n")  # out of range: ignored, the config's S
    assert scheduler._prefill_segments() == 4
    _ctl(ctl, "prefill_segments=x\n")
    assert scheduler._prefill_segments() == 4
    _ctl(ctl, "prefill_segments=1\n")  # the A/B arm with segmentation off
    assert scheduler._prefill_segments() == 1
    ctl.unlink()
    assert scheduler._prefill_segments() == 4


# --- GLM53_ADMIT_KV_BUDGET (program 2026-10-27T00:05:27Z (3)): admission by KV budget -- a waiting request is admitted
# only while the running requests' reserves (blocks of min(prompt + max_tokens, max_model_len)) plus its own fit the pool


def _pool_blocks(scheduler):
    return int(scheduler.kv_cache_manager.block_pool.num_gpu_blocks) - 1


def test_kv_budget_holds_a_request_until_the_pool_has_room(monkeypatch):
    from tests.test_mixed_steps import BLOCK, MAX_MODEL_LEN

    monkeypatch.setenv("GLM53_ADMIT_KV_BUDGET", "1")
    scheduler = _scheduler(mixed=False)
    per = MAX_MODEL_LEN // BLOCK  # a max-length reserve (prompt + max_tokens capped at max_model_len)
    fit = _pool_blocks(scheduler) // per  # how many max-length requests the pool holds (3 of 255 blocks)
    for i in range(fit):
        scheduler.add_request(_request(f"r{i}", 64, max_tokens=MAX_MODEL_LEN - 64))
    for _ in range(fit + 1):
        _commit(scheduler, scheduler.schedule())
    assert len(scheduler.running) == fit
    scheduler.add_request(_request("late", 64, max_tokens=MAX_MODEL_LEN - 64))
    scheduler.add_request(_request("small", 64, max_tokens=8))  # behind it: must not overtake
    for _ in range(3):
        out = scheduler.schedule()
        assert "late" not in out.num_scheduled_tokens and "small" not in out.num_scheduled_tokens, out.num_scheduled_tokens
        _commit(scheduler, out)
    assert scheduler.admissions_deferred_kv >= 3
    assert not any(getattr(r, "num_preemptions", 0) for r in scheduler.requests.values())


def test_kv_budget_admits_when_it_fits_and_when_nothing_runs(monkeypatch):
    monkeypatch.setenv("GLM53_ADMIT_KV_BUDGET", "1")
    scheduler = _scheduler(mixed=False)
    scheduler.add_request(_request("a", BUDGET, max_tokens=64))
    scheduler.add_request(_request("b", BUDGET, max_tokens=64))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens.get("a")  # small reserves: both fit (b may take the next step's budget)
    _commit(scheduler, out)
    out = scheduler.schedule()
    assert getattr(scheduler, "admissions_deferred_kv", 0) == 0
    # a request larger than the whole pool is still admitted when nothing runs (it cannot wait forever)
    s2 = _scheduler(mixed=False)
    s2.add_request(_request("huge", BUDGET, max_tokens=10**7))
    assert s2.schedule().num_scheduled_tokens.get("huge")


def test_kv_budget_off_by_default(monkeypatch):
    monkeypatch.delenv("GLM53_ADMIT_KV_BUDGET", raising=False)
    scheduler = _scheduler(mixed=False)
    total = _pool_blocks(scheduler)
    big = total * 64 * 6 // 10
    scheduler.add_request(_request("a", BUDGET, max_tokens=big))
    _commit(scheduler, scheduler.schedule())
    scheduler.add_request(_request("b", BUDGET, max_tokens=big))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens.get("b")  # today's admission: on the prompt's blocks alone


def _no_max_tokens(request_id, prompt_len=64):
    """an OpenAI request without max_tokens: vLLM hands the scheduler max_tokens = max_model_len - prompt"""
    from tests.test_mixed_steps import MAX_MODEL_LEN

    return _request(request_id, prompt_len, max_tokens=MAX_MODEL_LEN - prompt_len)


def _admitted(scheduler, ids, steps):
    for _ in range(steps):
        _commit(scheduler, scheduler.schedule())
    return [i for i in ids if scheduler.requests[i].status.name == "RUNNING"]


def test_kv_budget_reserve_cap_admits_clients_without_max_tokens(monkeypatch):
    """14's 00:21:37Z [M]: uncapped, a client without max_tokens reserves max_model_len (3 of the test pool's 255
    blocks fit); GLM53_ADMIT_RESERVE_TOKENS=1024 reserves blocks(64 + 1024) = 17, so 15 run and the 16th waits"""
    monkeypatch.setenv("GLM53_ADMIT_KV_BUDGET", "1")
    monkeypatch.setenv("GLM53_ADMIT_RESERVE_TOKENS", "0")  # no cap: the old reserve
    s0 = _scheduler(mixed=False)
    ids = [f"u{i}" for i in range(4)]
    for i in ids:
        s0.add_request(_no_max_tokens(i))
    assert len(_admitted(s0, ids, 6)) == 3
    monkeypatch.setenv("GLM53_ADMIT_RESERVE_TOKENS", "1024")
    s1 = _scheduler(mixed=False)
    s1.scheduler_config.max_num_seqs = 32
    s1.max_num_running_reqs = 32
    ids = [f"u{i}" for i in range(16)]
    for i in ids:
        s1.add_request(_no_max_tokens(i))
    run = _admitted(s1, ids, 20)
    assert run == ids[:15], run
    assert s1.admissions_deferred_kv >= 1 and s1.admit_reserve_capped == 16


def test_kv_budget_reserve_counts_the_lookahead(monkeypatch):
    """14's [S1]: the reserve adds the scheduler's lookahead (spec K / K + 1): prompt 64 + max_tokens 3,200 = 51
    blocks, 5 fit the 255; one lookahead token makes it 52 blocks and the 5th waits"""
    monkeypatch.setenv("GLM53_ADMIT_KV_BUDGET", "1")
    ids = [f"r{i}" for i in range(5)]
    for look, want in ((0, 5), (1, 4)):
        s = _scheduler(mixed=False)
        s.scheduler_config.max_num_seqs = 8
        s.max_num_running_reqs = 8
        s.num_lookahead_tokens = look
        for i in ids:
            s.add_request(_request(i, 64, max_tokens=3200))
        assert len(_admitted(s, ids, 8)) == want, look


def test_kv_budget_counts_requests_waiting_for_remote_kvs(monkeypatch):
    """9's 00:37:17Z (b): a WAITING_FOR_REMOTE_KVS request holds blocks: it is resident for the budget"""
    from vllm.v1.request import RequestStatus

    monkeypatch.setenv("GLM53_ADMIT_KV_BUDGET", "1")
    s = _scheduler(mixed=False)
    for i in range(3):  # 3 x 64 blocks of 255
        r = _no_max_tokens(f"remote{i}")
        r.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        s.requests[r.request_id] = r
    late = _no_max_tokens("late")
    late.status = RequestStatus.WAITING
    assert s._kv_budget_defers(late)  # 4 x 64 > 255
    for r in list(s.requests.values()):
        r.status = RequestStatus.WAITING  # none resident: admit
    assert not s._kv_budget_defers(late)
