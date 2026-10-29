# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The GLM-5.3-Flash serving self-test: scorer, plan, HTTP one-shot, startup
hook and its modes, logprobs and top-1-only scoring -- pure python (synthetic
golden-consistent observations and recorded mock responses; no device, no
server)."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_tt_plugin.selftest import __main__ as cli
from vllm_tt_plugin.selftest import endpoint as E
from vllm_tt_plugin.selftest import score as S
from vllm_tt_plugin.selftest.runner import (
    Config,
    effective_logprobs,
    load_observations,
    parse_completion,
    run_http_selftest,
)

PACK = S.load_pack()
DATA = Path(__file__).with_name("data")
MODEL = "zai-org/GLM-5.3-Flash"


def _top_list(
    golden: int, *, miss: str | None = None, k: int = 100
) -> list[tuple[int, float]]:
    """A top-K list with the golden token first; miss="top5" puts it second,
    miss="absent" leaves it out. Fillers are distinct ids > 10**6."""
    fill = [(10**6 + i, -3.0 - i) for i in range(k)]
    if miss is None:
        return [(golden, -0.01), (golden + 1, -2.0)] + fill[: k - 2]
    if miss == "top5":
        return [(golden + 1, -0.05), (golden, -0.4)] + fill[: k - 2]
    return [(golden + 1, -0.05)] + fill[: k - 1]


def golden_observation(
    req: S.Request, pack=PACK, *, miss: dict[int, str] | None = None
) -> S.Observation:
    """What a server that reproduces the fp32 goldens returns; ``miss`` maps a
    golden position to "top5" / "absent"."""
    miss = miss or {}
    gold = pack["prompts"][req.prompt_key]["golden_ids"]
    toks, top = [], []
    for k in range(req.max_tokens):
        pos = req.t + k
        g = gold[pos] if pos < len(gold) else 7
        tl = _top_list(g, miss=miss.get(pos))
        order = sorted(tl, key=lambda x: (-x[1], x[0]))
        toks.append(order[0][0])
        top.append(tl)
    return S.Observation(token_ids=toks, top=top)


def observe(plan, miss_by_kind: dict[str, dict[int, str]] | None = None, pack=PACK):
    miss_by_kind = miss_by_kind or {}
    return {
        r.name: golden_observation(r, pack, miss=miss_by_kind.get(r.kind)) for r in plan
    }


def near_ties(pack=PACK, key=S.AIME_KEY, n=32, thr=0.5):
    return [t for t, m in enumerate(pack["prompts"][key]["margins"][:n]) if m < thr]


def cleared(pack=PACK, key=S.AIME_KEY, n=32, thr=0.5):
    return [t for t, m in enumerate(pack["prompts"][key]["margins"][:n]) if m >= thr]


def without_logprobs(obs: dict[str, S.Observation]) -> dict[str, S.Observation]:
    """What a launch serving no logprobs returns: the greedy tokens only."""
    return {
        n: S.Observation(token_ids=list(o.token_ids), latency_s=o.latency_s)
        for n, o in obs.items()
    }


def _admit_mtp(monkeypatch, *, narrow_decode: bool):
    """A TT launch with an admitted MTP plan (--speculative-config method mtp,
    1 token), seeded as the platform's admission handle."""
    from vllm_tt_plugin.config import store_tt_spec_plan
    from vllm_tt_plugin.platform import TTPlatform
    from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, SpecPlan

    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        additional_config={},
    )
    store_tt_spec_plan(
        vllm_config,
        SpecPlan(
            effective_k=1,
            lanes_per_request=2,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state="internal",
            supports_narrow_decode=narrow_decode,
        ),
    )
    monkeypatch.setattr(TTPlatform, "_tt_vllm_config", vllm_config)
    return vllm_config


@pytest.fixture
def speculating_launch(monkeypatch):
    """The n29h launch: an MTP plan whose model declares narrow decode (GLM's
    spec_plan does): a logprobs request is served without speculation (the
    plain path, 2ff5fab), a greedy request without logprobs speculates."""
    return _admit_mtp(monkeypatch, narrow_decode=True)


@pytest.fixture
def refusing_spec_launch(monkeypatch):
    """An MTP plan without narrow decode: the platform refuses every request
    that asks for logprobs, as the n29g server (3b6d243) did at 22:24Z."""
    return _admit_mtp(monkeypatch, narrow_decode=False)


def speculates(params) -> bool:
    """Whether the TT runner speculates this request on the admitted launch:
    a plan is live, greedy, no logprobs (TTModelRunner._request_is_speculable)."""
    from vllm_tt_plugin.config import get_tt_spec_plan
    from vllm_tt_plugin.platform import TTPlatform

    cfg = TTPlatform._resolve_tt_admission_handle()
    return (
        cfg is not None
        and get_tt_spec_plan(cfg) is not None
        and params.logprobs is None
        and not params.temperature
    )


def platform_refusal(params) -> str | None:
    """The TT platform's per-request speculative check, as validate_request runs
    it: the refusal message, or None when the request is served."""
    from vllm_tt_plugin.platform import TTPlatform

    try:
        TTPlatform._reject_unsupported_speculative_request(params)
    except ValueError as exc:
        return str(exc)
    return None


# ------------------------------------------------------------ the pack and the plan


def test_pack_is_the_aime24_reference():
    aime = PACK["prompts"][S.AIME_KEY]
    assert PACK["model"] == MODEL
    assert len(aime["prompt_ids"]) == 161
    assert len(aime["golden_ids"]) >= 40 and len(aime["margins"]) == len(
        aime["golden_ids"]
    )
    # the 0.09-logit near-tie at the AIME first token (DECISIONS 2026-10-22 07:19Z)
    assert aime["margins"][0] < 0.5
    assert len(PACK["prompts"]["g128"]["prompt_ids"]) == 128
    assert len(PACK["prompts"]["g512"]["prompt_ids"]) == 512
    assert S.pack_matches_model(PACK, "/x/models--zai-org--GLM-5.3-Flash/snapshots/abc")
    assert not S.pack_matches_model(PACK, "/x/Qwen3", "qwen")


def test_plan_is_bounded():
    plan = S.build_plan(PACK)
    assert len(plan) == 39
    tf = [r for r in plan if r.kind == "tf"]
    assert [r.length for r in tf] == list(range(161, 193)) and all(
        r.max_tokens == 2 for r in tf
    )
    assert [
        (r.prompt_key, r.length, r.max_tokens) for r in plan if r.kind == "smoke"
    ] == [(S.AIME_KEY, 161, 8), ("g128", 128, 8), ("g512", 512, 8)]
    assert [(r.length, r.t, r.max_tokens) for r in plan if r.kind == "decode"] == [
        (177, 16, 8),
        (192, 31, 8),
    ]
    aime = PACK["prompts"][S.AIME_KEY]
    assert list(tf[5].prompt_ids) == aime["prompt_ids"] + aime["golden_ids"][:5]
    assert len({r.name for r in plan}) == 39
    with pytest.raises(S.SelftestError):
        S.build_plan(PACK, tf_steps=10_000)


# ------------------------------------------------------------------ scoring


def test_golden_server_passes():
    plan = S.build_plan(PACK)
    res = S.score(PACK, plan, observe(plan), duration_s=12.5, budget_s=60)
    assert res["pass"] and res["status"] == "pass" and res["reasons"] == []
    p = res["prefill_path"]
    assert (
        p["n"] == 32
        and p["n_cleared"] == len(cleared())
        and p["top1_cleared"] == 1.0
        and p["top5_cleared"] == 1.0
    )
    assert res["near_ties"] == near_ties()
    assert res["requests"] == 39 and res["responses"] == 39 and res["within_budget"]
    # decode rows: a t+1 row for every TF hit (t = 0..31) + 7 per decode request
    assert res["decode_path"]["n"] == 32 + 7 + 7
    assert all(r["ok"] and r["matched"] == 8 for r in res["smoke"])
    line = S.summary_line(res)
    assert line.startswith(
        f"SELFTEST PASS top-1 1.000 / top-5 1.000 (prefill, n={len(cleared())} of 32"
    )
    assert "39/39 requests in 12.5 s (budget 60 s)" in line
    table = S.render_table(res)
    assert len(table) == 1 + 32 + 3 + 2 + 2
    assert "near-tie" in table[1]  # t = 0 is the AIME near-tie


def test_near_tie_misses_are_excluded_from_the_gate():
    plan = S.build_plan(PACK)
    miss = {t: "absent" for t in near_ties()}
    res = S.score(PACK, plan, observe(plan, {"tf": miss}))
    assert res["pass"], res["reasons"]
    assert (
        res["prefill_path"]["top1"] < 1.0 and res["prefill_path"]["top1_cleared"] == 1.0
    )
    assert all(
        r["prefill_rank"] is None for r in res["tf_rows"] if r["t"] in near_ties()
    )


def test_clear_margin_misses_fail_the_gate():
    plan = S.build_plan(PACK)
    cl = cleared()
    n = len(cl)
    # one top-5 miss: top-1 (n-1)/n >= 0.85 and top-5 1.0 -> pass
    res = S.score(PACK, plan, observe(plan, {"tf": {cl[3]: "top5"}}))
    assert res["pass"] and res["prefill_path"]["misses_cleared"] == [cl[3]]
    # one absent: top-5 (n-1)/n < 0.98 -> fail on top-5
    res = S.score(PACK, plan, observe(plan, {"tf": {cl[3]: "absent"}}))
    assert not res["pass"] and "prefill top-1" in res["reasons"][0]
    # enough top-5 misses to drop top-1 below 0.85
    k = int(n * 0.15) + 1
    res = S.score(PACK, plan, observe(plan, {"tf": {t: "top5" for t in cl[:k]}}))
    assert not res["pass"] and res["prefill_path"]["top1_cleared"] < 0.85
    assert S.summary_line(res).startswith("SELFTEST FAIL")
    assert sum("MISS" in ln or "miss (top-5)" in ln for ln in S.render_table(res)) == k


def test_greedy_smoke_forks():
    plan = S.build_plan(PACK)
    g128 = PACK["prompts"]["g128"]["margins"]
    big = next(i for i, m in enumerate(g128[:8]) if m >= 0.5)
    res = S.score(PACK, plan, observe(plan, {"smoke": {big: "absent"}}))
    assert not res["pass"] and any("greedy smoke" in r for r in res["reasons"])
    row = next(r for r in res["smoke"] if r["prompt"] == "g128")
    assert row["fork_at"] == big and not row["ok"]
    # a fork at an fp32 near-tie position: by construction under top-1 scoring (no
    # served logprobs); with logprobs the fp32 margin is informational (ruling
    # 2026-10-25T12:26:33Z) and this fork's served margin is clear: still a fork
    pack = copy.deepcopy(PACK)
    pack["prompts"]["g128"]["margins"][big] = 0.1
    obs = observe(plan, pack=pack)
    g128_req = next(r for r in plan if r.name == "smoke/g128")
    obs[g128_req.name] = golden_observation(g128_req, pack, miss={big: "absent"})
    res = S.score(pack, plan, without_logprobs(obs), scoring="top1")
    assert res["pass"], res["reasons"]
    row = next(r for r in res["smoke"] if r["prompt"] == "g128")
    assert row["near_tie"] and row["matched"] == big and row["ok"]
    assert row["fork_basis"] == "fp32"
    assert (
        "near-tie fork"
        in [ln for ln in S.render_table(res) if ln.startswith("smoke g128")][0]
    )
    res = S.score(pack, plan, obs)
    row = next(r for r in res["smoke"] if r["prompt"] == "g128")
    assert not res["pass"] and not row["ok"] and row["fork_basis"] == "served"
    assert row["served_margin_bound"] and row["served_margin"] >= 0.5


def _fork_obs(req: S.Request, at: int, *, served_gap: float | None, pack=PACK) -> S.Observation:
    """Golden tokens up to ``at``, then the alternative ``golden + 1`` sampled: the
    served gap to the golden token is ``served_gap`` nats (None = golden absent
    from the top-K, whose K-th logprob is 10 nats below)."""
    gold = pack["prompts"][req.prompt_key]["golden_ids"]
    obs = golden_observation(req, pack)
    alt = gold[at] + 1
    fill = [(10**6 + i, -10.0) for i in range(20)]
    lst = [(alt, -0.4)] + ([] if served_gap is None else [(gold[at], -0.4 - served_gap)]) + fill
    obs.top[at] = lst
    obs.token_ids = list(obs.token_ids[:at]) + [alt] + list(obs.token_ids[at + 1:])
    return obs


def test_smoke_fork_served_margin():
    """cf3h-h' (2026-10-25 12:19Z): g512 forked at step 5 where fp32's gap is 1.01
    nats, the served model's 0.00-0.50 (ruling 12:20:49Z (1)): a served near-tie
    passes; a clear served fork at a clear fp32 position still fails; the fork's
    served logprobs are kept; top-1 scoring (no logprobs) keeps the fp32 rule."""
    plan = S.build_plan(PACK)
    g512 = PACK["prompts"]["g512"]["margins"]
    at = next(i for i, m in enumerate(g512[:8]) if m >= 0.5)
    req = next(r for r in plan if r.name == "smoke/g512")
    for gap, ok in ((0.25, True), (0.0, True), (0.75, False), (None, False)):
        obs = observe(plan)
        obs[req.name] = _fork_obs(req, at, served_gap=gap)
        res = S.score(PACK, plan, obs)
        row = next(r for r in res["smoke"] if r["prompt"] == "g512")
        assert row["fork_at"] == at and row["fork_margin"] == g512[at]
        assert row["ok"] is ok and row["near_tie"] is ok and res["pass"] is ok, (gap, res["reasons"])
        assert row["served_margin"] == (9.6 if gap is None else gap)
        assert row["served_margin_bound"] is (gap is None)
        assert row["served_logp_sampled"] == -0.4
        assert [PACK["prompts"]["g512"]["golden_ids"][at] + 1, -0.4] in row["served_top5_at_fork"]
        line = [ln for ln in S.render_table(res) if ln.startswith("smoke g512")][0]
        assert ("near-tie fork" in line) is ok and "served" in line
    # golden below the top-K: the margin is a lower bound, never a served near-tie (even a flat top-K)
    obs = observe(plan)
    obs[req.name] = _fork_obs(req, at, served_gap=None)
    obs[req.name].top[at] = [(t, -0.4) for t, _ in obs[req.name].top[at]]
    res = S.score(PACK, plan, obs)
    row = next(r for r in res["smoke"] if r["prompt"] == "g512")
    assert row["served_margin_bound"] and row["served_margin"] == 0.0 and not row["ok"]
    # a clear served decode margin, but the teacher-forced prefill at the fork is an
    # exact tie (cf3h-g0 g512: decode 0.50, TF 0.00): a near-tie; a TF gap is not
    gold = PACK["prompts"]["g512"]["golden_ids"]
    for tf_gap, ok in ((0.0, True), (0.125, False)):
        obs = observe(plan)
        obs[req.name] = _fork_obs(req, at, served_gap=0.75)
        assert [r.name for r in S.fork_tf_requests(PACK, plan, obs)] == ["forktf/g512"]
        ftf = S.fork_tf_requests(PACK, plan, obs)[0]
        assert ftf.t == at and ftf.max_tokens == 1 and ftf.prompt_ids == req.prompt_ids + tuple(gold[:at])
        obs[ftf.name] = S.Observation(token_ids=[gold[at]], top=[[(gold[at], -0.7), (gold[at] + 1, -0.7 - tf_gap)]])
        res = S.score(PACK, plan, obs)
        row = next(r for r in res["smoke"] if r["prompt"] == "g512")
        assert row["ok"] is ok and row["tf_exact_tie"] is ok and row["tf_margin"] == tf_gap, row
        assert res["requests"] == len(plan) + 1 and res["pass"] is ok
    # top-1 scoring: no served logprobs, the fp32 margin decides (a clear one: fail)
    obs = observe(plan)
    obs[req.name] = _fork_obs(req, at, served_gap=0.0)
    res = S.score(PACK, plan, without_logprobs(obs), scoring="top1")
    row = next(r for r in res["smoke"] if r["prompt"] == "g512")
    assert not row["ok"] and "served_margin" not in row and row["fork_basis"] == "fp32"
    assert S.fork_tf_requests(PACK, plan, observe(plan)) == []


def test_decode_after_prefill_rows_stop_at_the_first_divergence():
    plan = S.build_plan(PACK)
    res = S.score(PACK, plan, observe(plan))
    rows = [r for r in res["decode_rows"] if r["src"] == "decode/len=177"]
    assert [r["pos"] for r in rows] == list(range(17, 24))
    res = S.score(PACK, plan, observe(plan, {"decode": {18: "top5"}}))
    rows = [r for r in res["decode_rows"] if r["src"] == "decode/len=177"]
    assert [r["pos"] for r in rows] == [17, 18] and rows[1]["decode_rank"] == 1
    assert (
        "decode/len=177: decode ranks at positions [17, 18] = [0, 1]"
        in S.render_table(res)
    )


def test_decode_path_gate():
    plan = S.build_plan(PACK)
    # every decode row absent -> decode top-5 0 -> the decode path fails, prefill intact
    miss = {t: "absent" for t in range(1, 48)}
    obs = observe(plan, {"decode": miss})
    for r in plan:
        if r.kind == "tf":
            o = golden_observation(r)
            o.top[1] = _top_list(
                PACK["prompts"][S.AIME_KEY]["golden_ids"][r.t + 1]
                if r.t + 1 < 48
                else 7,
                miss="absent",
            )
            obs[r.name] = o
    res = S.score(PACK, plan, obs)
    assert res["prefill_path"]["top1_cleared"] == 1.0
    assert not res["pass"] and any(x.startswith("decode top-1") for x in res["reasons"])


def test_missing_and_errored_requests_fail():
    plan = S.build_plan(PACK)
    obs = observe(plan)
    del obs["tf/t=3"]
    obs["smoke/g512"] = S.Observation(error="HTTP 500: boom")
    res = S.score(PACK, plan, obs, duration_s=75, budget_s=60)
    assert not res["pass"] and res["errors"] == [
        {"request": "tf/t=3", "error": "no response"},
        {"request": "smoke/g512", "error": "HTTP 500: boom"},
    ]
    assert res["responses"] == 37 and not res["within_budget"]
    assert (
        "OVER BUDGET" in S.summary_line(res)
        and "2 request error(s)" in res["reasons"][0]
    )
    assert any(ln.startswith("ERROR smoke/g512") for ln in S.render_table(res))


def test_too_few_clear_rows_is_not_a_pass():
    plan = S.build_plan(PACK, tf_steps=4)
    res = S.score(PACK, plan, observe(plan))
    assert not res["pass"] and "clear-margin prefill rows" in res["reasons"][0]


def test_effective_logprobs():
    assert effective_logprobs(100, -1) == 100
    assert effective_logprobs(100, None) == 100
    assert effective_logprobs(100, 20) == 20
    with pytest.raises(S.SelftestError):
        effective_logprobs(100, 3)


def test_config_from_env():
    cfg = Config.from_env({})
    assert (
        cfg.mode == "strict"
        and cfg.tf_steps == 32
        and cfg.budget_s == 60
        and cfg.concurrency == 1
    )
    cfg = Config.from_env(
        {
            "GLM53_SELFTEST": "warn",
            "GLM53_SELFTEST_TOP1": "0.9",
            "GLM53_SELFTEST_OUT": "/tmp/v.json",
            "GLM53_SELFTEST_CONCURRENCY": "4",
        }
    )
    assert (
        cfg.mode == "warn"
        and cfg.thresholds.top1 == 0.9
        and str(cfg.out_path) == "/tmp/v.json"
        and cfg.concurrency == 4
    )
    with pytest.raises(S.SelftestError):
        Config.from_env({"GLM53_SELFTEST": "maybe"})
    assert cfg.scoring == "auto" and Config.from_env({}).scoring == "auto"
    assert Config.from_env({"GLM53_SELFTEST_SCORING": " TOP1 "}).scoring == "top1"
    assert Config.from_env({"GLM53_SELFTEST_SCORING": "logprobs"}).scoring == "logprobs"
    with pytest.raises(S.SelftestError):
        Config.from_env({"GLM53_SELFTEST_SCORING": "top5"})


# ------------------------------------------------------------------ recorded responses


def test_parse_recorded_completion():
    """A /v1/completions response recorded from the live int25t server
    (2026-10-22 09:58Z, host-sampler logprobs, token_id keys): the shape the
    CLI parses."""
    rec = DATA / "selftest_completion_live.json"
    payload = json.loads(rec.read_text())
    o = parse_completion(payload)
    assert len(o.token_ids) == payload["usage"]["completion_tokens"] == len(o.top)
    assert all(len(pos) >= 5 for pos in o.top)
    assert all(
        isinstance(t, int) and isinstance(lp, float) for pos in o.top for t, lp in pos
    )
    assert S.rank_of(o.top[0], o.token_ids[0]) == 0


def test_recorded_live_observations_score_as_recorded():
    """The logprobs the one-shot CLI recorded against the live int25t server
    (2026-10-22 09:58Z, SELFTEST PASS 0.931 / 1.000, top-16 per position kept):
    the scorer reproduces the recorded verdict."""
    rec = DATA / "selftest_observations_live.json"
    obs = load_observations(rec)
    expect = json.loads((DATA / "selftest_verdict_live.json").read_text())
    # the recording predates the short-prompt rows (2026-10-25): its plan, without them
    plan = S.build_plan(PACK, tf_steps=expect["prefill_path"]["n"], short_prompts=())
    res = S.score(PACK, plan, obs, S.Thresholds(**expect["thresholds"]))
    assert res["status"] == expect["status"]
    assert (
        res["prefill_path"] == expect["prefill_path"]
        and res["decode_path"] == expect["decode_path"]
    )
    assert [r["ok"] for r in res["smoke"]] == [r["ok"] for r in expect["smoke"]]
    assert res["pass"] and S.summary_line(res).startswith(
        "SELFTEST PASS top-1 0.931 / top-5 1.000 (prefill, n=29 of 32"
    )


def test_logprobs_scoring_reproduces_the_recorded_verdict_key_for_key():
    """The logprobs scoring is unchanged by the top-1-only mode: every scored
    key of the verdict recorded before it existed comes back byte-identical
    (JSON); the verdict only gains the scoring keys."""
    obs = load_observations(DATA / "selftest_observations_live.json")
    expect = json.loads((DATA / "selftest_verdict_live.json").read_text())
    # the recording predates the short-prompt rows (2026-10-25): its plan, without them
    plan = S.build_plan(PACK, tf_steps=expect["prefill_path"]["n"], short_prompts=())
    res = S.score(
        PACK,
        plan,
        obs,
        S.Thresholds(**expect["thresholds"]),
        duration_s=expect["duration_s"],
        budget_s=expect["budget_s"],
        mode=expect["mode"],
    )
    added = {
        "scoring",
        "scoring_requested",
        "scoring_reason",
        "logprobs_attempt",
        "spec_identity",
        "short",
        "short_ok",
    }
    assert set(res) == set(expect) | added
    for key in sorted(set(expect) - {"info", "utc", "goldens"}):
        assert json.dumps(res[key], sort_keys=True) == json.dumps(
            expect[key], sort_keys=True
        ), key
    # the pack gained the short-prompt rows (2026-10-25), so its sha moved; the recorded plan's three prompts are
    # unchanged entry for entry (content hashes of the pack the verdict was recorded with)
    assert res["goldens"]["sha256"] != expect["goldens"]["sha256"]
    for key, h in (("aime24_161", "3592b2a1727990d6"), ("g128", "8bbe14b8bcafc8af"), ("g512", "331dcc61204903ae")):
        assert hashlib.sha256(json.dumps(PACK["prompts"][key], sort_keys=True).encode()).hexdigest()[:16] == h, key
    assert res["scoring"] == "logprobs" and res["logprobs_attempt"] is None
    assert res["spec_identity"] is None
    assert "top-1-only" not in S.summary_line(res) and "MTP" not in S.summary_line(res)


def test_top1_scoring_agrees_with_the_logprobs_top1_on_live_observations():
    """The live int25t observations with their logprobs dropped: the served
    greedy token is the logprobs' rank 0, so the top-1-only scoring finds the
    same clear-margin rows and the same top-1 misses."""
    obs = load_observations(DATA / "selftest_observations_live.json")
    expect = json.loads((DATA / "selftest_verdict_live.json").read_text())
    # the recording predates the short-prompt rows (2026-10-25): its plan, without them
    plan = S.build_plan(PACK, tf_steps=expect["prefill_path"]["n"], short_prompts=())
    thr = S.Thresholds(**expect["thresholds"])
    lp = S.score(PACK, plan, obs, thr)
    t1 = S.score(PACK, plan, without_logprobs(obs), thr, scoring="top1")
    for path in ("prefill_path", "decode_path"):
        for key in ("n", "n_cleared", "top1", "top1_cleared", "misses_cleared"):
            assert t1[path][key] == lp[path][key], (path, key)
        assert t1[path]["top5"] is None and t1[path]["top5_cleared"] is None
    assert t1["near_ties"] == lp["near_ties"] and t1["smoke"] == lp["smoke"]
    assert t1["pass"] and S.summary_line(t1).startswith(
        "SELFTEST PASS top-1 0.931 / top-5 n/a (prefill, n=29 of 32"
    )


# ------------------------------------------------------------------ top-1-only scoring


def test_top1_scoring_passes_a_golden_server():
    plan = S.build_plan(PACK)
    res = S.score(
        PACK,
        plan,
        without_logprobs(observe(plan)),
        duration_s=12.5,
        budget_s=60,
        scoring="top1",
        scoring_info={"scoring_requested": "auto", "scoring_reason": "auto: x"},
    )
    assert res["pass"] and res["reasons"] == [] and res["errors"] == []
    assert (res["scoring"], res["scoring_requested"], res["scoring_reason"]) == (
        "top1",
        "auto",
        "auto: x",
    )
    p, d = res["prefill_path"], res["decode_path"]
    assert (
        p["n"] == 32 and p["n_cleared"] == len(cleared()) and p["top1_cleared"] == 1.0
    )
    assert p["top5"] is p["top5_cleared"] is d["top5"] is d["top5_cleared"] is None
    # the same rows as the logprobs scoring: a t+1 row per TF hit + 7 per decode
    assert d["n"] == 32 + 7 + 7 and res["near_ties"] == near_ties()
    assert all(r["prefill_rank"] == 0 for r in res["tf_rows"])
    line = S.summary_line(res)
    assert line.startswith(
        f"SELFTEST PASS top-1 1.000 / top-5 n/a (prefill, n={len(cleared())} of 32"
    )
    assert "decode top-1 1.000 / top-5 n/a" in line and "top-1-only scoring" in line
    table = S.render_table(res)
    assert len(table) == 1 + 32 + 3 + 2 + 2 + 1 and table[-1] == "scoring top1: auto: x"
    assert "decode/len=177: decode top-1 at positions" in "\n".join(table)
    with pytest.raises(S.SelftestError):
        S.score(PACK, plan, observe(plan), scoring="top5")


def test_top1_scoring_gate():
    plan = S.build_plan(PACK)
    cl = cleared()
    n = len(cl)
    # the golden token absent from the top-K at one clear position: the logprobs
    # scoring fails on top-5; top-1-only: one wrong served token, top-1
    # (n-1)/n >= 0.85, top-5 is not measured and not gated -> pass
    obs = observe(plan, {"tf": {cl[3]: "absent"}})
    assert not S.score(PACK, plan, obs)["pass"]
    res = S.score(PACK, plan, without_logprobs(obs), scoring="top1")
    assert res["pass"] and res["prefill_path"]["misses_cleared"] == [cl[3]]
    assert "MISS" in S.render_table(res)[1 + cl[3]]
    # near-tie misses stay excluded
    obs = observe(plan, {"tf": {t: "absent" for t in near_ties()}})
    res = S.score(PACK, plan, without_logprobs(obs), scoring="top1")
    assert res["pass"] and res["prefill_path"]["top1"] < 1.0
    # top-1 below 0.85 on the clear-margin rows: fail, threshold not lowered
    k = int(n * 0.15) + 1
    obs = observe(plan, {"tf": {t: "top5" for t in cl[:k]}})
    res = S.score(PACK, plan, without_logprobs(obs), scoring="top1")
    top1 = res["prefill_path"]["top1_cleared"]
    assert not res["pass"] and top1 < 0.85 == res["thresholds"]["top1"]
    assert res["reasons"] == [f"prefill top-1 {top1:.3f} < 0.85 (top-1-only scoring)"]
    assert S.summary_line(res).startswith(f"SELFTEST FAIL top-1 {top1:.3f} / top-5 n/a")
    # the served token is scored, not returned logprobs: rank 0 in the logprobs
    # but a different sampled token is a top-1 miss
    obs = observe(plan)
    for t in cl[:k]:
        obs[f"tf/t={t}"].token_ids[0] += 1
    assert S.score(PACK, plan, obs)["pass"]
    assert not S.score(PACK, plan, obs, scoring="top1")["pass"]
    # the decode path is gated on top-1 too: every TF request's second token wrong
    obs = without_logprobs(observe(plan))
    for r in plan:
        if r.kind == "tf":
            obs[r.name].token_ids[1] += 1
    res = S.score(PACK, plan, obs, scoring="top1")
    assert res["prefill_path"]["top1_cleared"] == 1.0
    assert not res["pass"] and res["reasons"][0].startswith("decode top-1")
    # fewer clear-margin rows than min_rows is not a pass
    plan4 = S.build_plan(PACK, tf_steps=4)
    res = S.score(PACK, plan4, without_logprobs(observe(plan4)), scoring="top1")
    assert not res["pass"] and "clear-margin prefill rows" in res["reasons"][0]
    # no tokens = a request error
    obs = without_logprobs(observe(plan))
    obs["smoke/g128"] = S.Observation(token_ids=[])
    res = S.score(PACK, plan, obs, scoring="top1")
    assert not res["pass"] and res["errors"][0]["request"] == "smoke/g128"


def test_mtp_identity_row():
    """A speculating launch: the AIME smoke prompt without logprobs (the
    speculative path) must equal the plain-path smoke row token for token."""
    plan = S.build_plan(PACK)
    obs = observe(plan)
    sreq = S.spec_identity_request(plan)
    assert (sreq.name, sreq.kind, sreq.prompt_ids, sreq.max_tokens) == (
        f"spec/{S.AIME_KEY}",
        "spec",
        next(r for r in plan if r.name == S.SPEC_REFERENCE).prompt_ids,
        8,
    )
    same = S.Observation(token_ids=list(obs[S.SPEC_REFERENCE].token_ids))
    spec = "method=mtp, num_speculative_tokens=1"
    res = S.score(PACK, plan, obs, speculative=spec, spec_observation=same)
    si = res["spec_identity"]
    assert res["pass"] and si["checked"] and si["ok"] and si["identical"]
    assert si["spec_token_ids"] == si["plain_token_ids"] and si["diverge_at"] is None
    assert si["speculative"] == spec and res["requests"] == res["responses"] == 40
    assert S.summary_line(res).endswith("; MTP==plain 8/8")
    assert S.render_table(res)[-1].endswith("len 161: 8/8 identical")
    # one token off at k: FAIL, even at a near-tie (speculation must be lossless)
    for k in (3, 0):
        bad = S.Observation(token_ids=list(same.token_ids))
        bad.token_ids[k] += 1
        res = S.score(PACK, plan, obs, speculative=spec, spec_observation=bad)
        si = res["spec_identity"]
        assert not res["pass"] and not si["ok"] and si["diverge_at"] == k
        assert res["reasons"] == [si["reason"]]
        assert si["reason"].startswith(f"MTP path differs from plain path at token {k}")
        assert (
            si["golden_margin_at_divergence"]
            == PACK["prompts"][S.AIME_KEY]["margins"][k]
        )
        assert f"MTP!=plain at token {k}" in S.summary_line(res)
    assert PACK["prompts"][S.AIME_KEY]["margins"][0] < 0.5  # k = 0 is a near-tie
    # short output = a divergence where it ends
    short = S.Observation(token_ids=list(same.token_ids[:5]))
    res = S.score(PACK, plan, obs, speculative=spec, spec_observation=short)
    assert res["spec_identity"]["diverge_at"] == 5 and not res["pass"]
    # the speculative request failed / never answered: FAIL, a request error
    for o in (S.Observation(error="RuntimeError: boom"), None):
        res = S.score(PACK, plan, obs, speculative=spec, spec_observation=o)
        assert not res["pass"] and res["reasons"][0] == "1 request error(s)"
        assert res["reasons"][-1].startswith("MTP identity request spec/aime24_161:")
        assert res["errors"][-1]["request"] == "spec/aime24_161"
        assert (res["requests"], res["responses"]) == (40, 39)
    # no plain-path reference: FAIL (and the request error is counted)
    obs2 = dict(obs)
    obs2[S.SPEC_REFERENCE] = S.Observation(error="HTTP 500")
    res = S.score(PACK, plan, obs2, speculative=spec, spec_observation=same)
    assert "MTP identity: no plain-path reference" in res["reasons"][-1]
    # top-1-only scoring: every request ran speculatively, nothing to compare
    res = S.score(PACK, plan, without_logprobs(obs), scoring="top1", speculative=spec)
    si = res["spec_identity"]
    assert res["pass"] and not si["checked"] and si["ok"] and res["requests"] == 39
    assert si["reason"].startswith("not measurable")
    assert "MTP identity not measurable" in S.summary_line(res)
    # no speculation: no row
    assert S.score(PACK, plan, obs)["spec_identity"] is None


# ------------------------------------------------------- the one-shot CLI over HTTP


class _GoldenHandler(BaseHTTPRequestHandler):
    """A fake OpenAI endpoint answering from the goldens pack (``server.miss``
    flips golden positions of the TF path)."""

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": MODEL}]})
        return self._send(404, {"error": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        self.server.requests.append(body)
        if self.path != "/v1/completions" or body.get("model") != MODEL:
            return self._send(400, {"error": "bad request"})
        ids = body["prompt"]
        for key, p in PACK["prompts"].items():
            base = p["prompt_ids"]
            if ids[: len(base)] == base and (key != S.AIME_KEY or len(ids) >= 161):
                t = len(ids) - len(base)
                if ids[len(base) :] == p["golden_ids"][:t]:
                    break
        else:
            return self._send(400, {"error": "unknown prompt"})
        kind = "tf" if body["max_tokens"] == 2 else "x"
        req = S.Request("r", kind, key, t, tuple(ids), body["max_tokens"])
        o = golden_observation(req, miss=self.server.miss.get(kind))
        k = body.get("logprobs")
        from vllm.sampling_params import SamplingParams

        params = SamplingParams(
            max_tokens=body["max_tokens"], temperature=body["temperature"], logprobs=k
        )
        refusal = platform_refusal(params)
        div = self.server.spec_diverge_at
        if speculates(params) and div is not None:  # a lossy speculative path
            o.token_ids[div] += 1
        if refusal:  # vLLM's 400 for a ValueError from validate_request
            return self._send(
                400,
                {
                    "error": {
                        "message": refusal,
                        "type": "BadRequestError",
                        "param": None,
                        "code": 400,
                    }
                },
            )
        return self._send(
            200,
            {
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "length",
                        "token_ids": o.token_ids,
                        "logprobs": None
                        if k is None
                        else {
                            "tokens": [f"token_id:{t}" for t in o.token_ids],
                            "top_logprobs": [
                                {f"token_id:{t}": lp for t, lp in pos[:k]}
                                for pos in o.top
                            ],
                        },
                    }
                ],
                "usage": {"completion_tokens": len(o.token_ids)},
            },
        )


@pytest.fixture
def golden_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _GoldenHandler)
    srv.requests, srv.miss, srv.spec_diverge_at = [], {}, None
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


def test_cli_passes_against_a_golden_endpoint(golden_server, tmp_path, capsys):
    srv, url = golden_server
    out = tmp_path / "verdict.json"
    rc = cli.main(
        [
            "--url",
            url,
            "--out",
            str(out),
            "--save-observations",
            str(tmp_path / "obs.json"),
            "--concurrency",
            "4",
        ]
    )
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["pass"] and res["requests"] == 39 == len(srv.requests)
    assert {r["logprobs"] for r in srv.requests} == {100} and all(
        r["temperature"] == 0.0 for r in srv.requests
    )
    assert sorted(len(r["prompt"]) for r in srv.requests) == sorted(
        [128, 512, 177, 192, 129, 257] + list(range(161, 193)) + [161]
    )
    assert res["info"]["driver"] == "http" and res["info"]["model"] == MODEL
    printed = capsys.readouterr().out
    assert "SELFTEST PASS top-1 1.000 / top-5 1.000" in printed
    # the recorded observations reproduce the verdict offline
    obs = load_observations(tmp_path / "obs.json")
    again = S.score(PACK, S.build_plan(PACK), obs)
    assert again["prefill_path"] == res["prefill_path"]


def test_cli_fails_and_exits_1_on_clear_margin_misses(golden_server, tmp_path, capsys):
    srv, url = golden_server
    srv.miss = {"tf": {t: "absent" for t in cleared()[:3]}}
    rc = cli.main(["--url", url, "--out", str(tmp_path / "v.json"), "--quiet"])
    assert rc == 1
    assert "SELFTEST FAIL" in capsys.readouterr().out
    assert (
        json.loads((tmp_path / "v.json").read_text())["prefill_path"]["misses_cleared"]
        == cleared()[:3]
    )


def test_cli_exits_2_without_a_server(tmp_path):
    assert cli.main(["--url", "http://127.0.0.1:9", "--timeout-s", "2"]) == 2


def test_http_runner_counts_errors(golden_server):
    srv, url = golden_server
    cfg = Config(mode="cli", request_timeout_s=10)
    res = run_http_selftest(url, cfg, model="not-the-served-model")
    assert not res["pass"] and res["responses"] == 0 and len(res["errors"]) == 39
    assert res["errors"][0]["error"].startswith("HTTP 400")
    # a 400 that is not the logprobs refusal never switches the scoring
    assert res["scoring"] == "logprobs" and res["logprobs_attempt"] is None


def test_cli_auto_falls_back_to_top1_on_a_server_refusing_logprobs(
    golden_server, refusing_spec_launch, tmp_path, capsys
):
    srv, url = golden_server
    out = tmp_path / "v.json"
    rc = cli.main(["--url", url, "--out", str(out), "--quiet"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["pass"] and res["scoring"] == "top1" and res["errors"] == []
    att = res["logprobs_attempt"]
    assert att["request"] == "tf/t=0" and att["logprobs"] == 100
    assert att["error"].startswith("HTTP 400") and "logprobs=100" in att["error"]
    assert res["scoring_reason"].startswith(
        "auto: the first logprobs request was refused"
    )
    # one refused logprobs request, then the whole plan without logprobs
    assert len(srv.requests) == 1 + 39 and srv.requests[0]["logprobs"] == 100
    assert not any("logprobs" in r for r in srv.requests[1:])
    assert res["info"]["logprobs"] is None
    assert "SELFTEST PASS top-1 1.000 / top-5 n/a" in capsys.readouterr().out
    # explicit logprobs scoring never falls back: 39 refusals, exit 1
    srv.requests.clear()
    rc = cli.main(["--url", url, "--scoring", "logprobs", "--quiet"])
    assert rc == 1 and len(srv.requests) == 39
    assert "39 request error(s)" in capsys.readouterr().out


def test_cli_spec_identity_on_a_speculating_server(
    golden_server, speculating_launch, tmp_path, capsys
):
    """The n29h server: logprobs served on the plain path, so the logprobs
    scoring stands; --spec-identity adds the AIME smoke prompt without
    logprobs (speculative) and requires it to equal the plain path."""
    srv, url = golden_server
    out = tmp_path / "v.json"
    rc = cli.main(["--url", url, "--out", str(out), "--spec-identity", "--quiet"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["pass"] and res["scoring"] == "logprobs" and res["errors"] == []
    assert res["prefill_path"]["top5_cleared"] == 1.0
    assert res["spec_identity"]["ok"] and res["spec_identity"]["identical"]
    assert len(srv.requests) == 39 + 1
    assert all(r["logprobs"] == 100 for r in srv.requests[:39])
    last = srv.requests[-1]
    assert "logprobs" not in last and last["max_tokens"] == 8
    assert last["prompt"] == PACK["prompts"][S.AIME_KEY]["prompt_ids"]
    assert "MTP==plain 8/8" in capsys.readouterr().out
    # a lossy speculative path: FAIL, exit 1, the token named
    srv.requests.clear()
    srv.spec_diverge_at = 2
    rc = cli.main(["--url", url, "--out", str(out), "--spec-identity", "--quiet"])
    assert rc == 1
    res = json.loads(out.read_text())
    assert res["prefill_path"]["top1_cleared"] == 1.0  # the plain path is fine
    assert res["reasons"] == [res["spec_identity"]["reason"]]
    assert "MTP path differs from plain path at token 2" in capsys.readouterr().out
    # without --spec-identity no extra request
    srv.requests.clear()
    assert cli.main(["--url", url, "--quiet"]) == 0 and len(srv.requests) == 39


# ------------------------------------------------------------------ the startup hook


class FakeEngine:
    """engine_client.generate as the hook uses it, answering from the pack."""

    def __init__(
        self,
        miss=None,
        fail_after=None,
        vllm_config=None,
        spec_diverge_at=None,
        spec_fail=False,
    ):
        self.miss, self.fail_after, self.calls = miss or {}, fail_after, []
        # requests on the speculative path: token spec_diverge_at flipped
        # (a lossy MTP path), or an exception
        self.spec_diverge_at, self.spec_fail, self.paths = (
            spec_diverge_at,
            spec_fail,
            [],
        )
        self.model_config = SimpleNamespace(
            model="/m/models--zai-org--GLM-5.3-Flash/snapshots/x", max_logprobs=-1
        )
        if vllm_config is not None:  # what the hook may read the launch from
            self.vllm_config = vllm_config

    async def generate(self, prompt, params, request_id):
        self.calls.append(
            (
                list(prompt["prompt_token_ids"]),
                params.max_tokens,
                params.logprobs,
                params.temperature,
                request_id,
            )
        )
        # AsyncLLM.add_request -> TTPlatform.validate_request
        refusal = platform_refusal(params)
        if refusal:
            raise ValueError(refusal)
        spec_path = speculates(params)
        self.paths.append("spec" if spec_path else "plain")
        if spec_path and self.spec_fail:
            raise RuntimeError("MTP verify timeout")
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("device timeout")
        ids = prompt["prompt_token_ids"]
        for key, p in PACK["prompts"].items():
            base = p["prompt_ids"]
            if (
                ids[: len(base)] == base
                and ids[len(base) :] == p["golden_ids"][: len(ids) - len(base)]
            ):
                break
        kind = "tf" if params.max_tokens == 2 else "x"
        o = golden_observation(
            S.Request(
                "r", kind, key, len(ids) - len(base), tuple(ids), params.max_tokens
            ),
            miss=self.miss.get(kind),
        )
        if spec_path and self.spec_diverge_at is not None:
            o.token_ids[self.spec_diverge_at] += 1
        logprobs = [
            {
                t: SimpleNamespace(logprob=lp, rank=i + 1)
                for i, (t, lp) in enumerate(pos)
            }
            for pos in o.top
        ]
        if params.logprobs is None:  # vLLM returns none when none were asked
            logprobs = None
        await asyncio.sleep(0)
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=o.token_ids[:1],
                    logprobs=None if logprobs is None else logprobs[:1],
                )
            ]
        )
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=o.token_ids, logprobs=logprobs)]
        )


@pytest.fixture
def tt_log():
    """Messages of the plugin's logger tree (it does not propagate to the root
    logger caplog listens on)."""
    msgs: list[str] = []

    class _Grab(logging.Handler):
        def emit(self, record):
            msgs.append(record.getMessage())

    h = _Grab()
    lg = logging.getLogger("vllm.tt")
    level = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        yield msgs
    finally:
        lg.removeHandler(h)
        lg.setLevel(level)


def _init(engine, env, tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith("GLM53_SELFTEST"):
            monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GLM53_SELFTEST_OUT", str(tmp_path / "verdict.json"))
    state = SimpleNamespace()
    args = Namespace(model="zai-org/GLM-5.3-Flash", served_model_name=None)
    plugin = E.create_plugin()
    asyncio.run(plugin.init_state(engine, state, args))
    return state


def test_startup_hook_passes_and_records(tmp_path, monkeypatch, tt_log):
    eng = FakeEngine()
    state = _init(eng, {"GLM53_SELFTEST": "strict"}, tmp_path, monkeypatch)
    res = state.glm53_selftest
    assert res["pass"] and res["requests"] == 39 == len(eng.calls)
    assert all(c[2] == 100 and c[3] == 0.0 for c in eng.calls)
    assert json.loads((tmp_path / "verdict.json").read_text())["status"] == "pass"
    assert (
        res["info"]["driver"] == "engine"
        and res["info"]["model"] == "zai-org/GLM-5.3-Flash"
    )
    assert any("SELFTEST PASS top-1 1.000 / top-5 1.000" in m for m in tt_log)
    assert sum(E.LOG_TAG in m for m in tt_log) >= 1 + 32 + 3 + 2 + 1
    assert res["scoring"] == "logprobs" and res["scoring_requested"] == "auto"
    assert res["spec_identity"] is None and eng.paths == ["plain"] * 39
    assert not any("top-1-only" in m or "MTP" in m for m in tt_log)


def _timeless(rows):
    return [{k: v for k, v in r.items() if k != "latency_s"} for r in rows]


def _scored(res):
    """The verdict minus wall-clock fields."""
    out = {k: v for k, v in res.items() if k not in ("utc", "duration_s", "info")}
    return {
        **out,
        "tf_rows": _timeless(res["tf_rows"]),
        "smoke": _timeless(res["smoke"]),
        "short": _timeless(res.get("short", [])),
    }


def test_startup_hook_auto_keeps_logprobs_scoring_on_a_logprobs_launch(
    tmp_path, monkeypatch
):
    """auto on a launch that serves logprobs (no speculation in the engine's
    config) = the logprobs scoring as before: every request asks for 100,
    distinct request ids, a verdict equal to the explicit logprobs one and to
    the scorer's on the same observations."""
    eng = FakeEngine(vllm_config=SimpleNamespace(speculative_config=None))
    auto = _init(eng, {}, tmp_path, monkeypatch).glm53_selftest
    assert auto["pass"] and auto["scoring"] == "logprobs"
    assert (
        auto["scoring_reason"]
        == "auto: logprobs served (the first logprobs request was not refused)"
    )
    assert auto["spec_identity"] is None
    assert auto["logprobs_attempt"] is None and auto["info"]["logprobs"] == 100
    assert len(eng.calls) == 39 and {c[2] for c in eng.calls} == {100}
    assert len({c[4] for c in eng.calls}) == 39
    eng2 = FakeEngine()
    forced = _init(
        eng2, {"GLM53_SELFTEST_SCORING": "logprobs"}, tmp_path, monkeypatch
    ).glm53_selftest
    assert [c[:4] for c in eng2.calls] == [c[:4] for c in eng.calls]
    assert forced["scoring_reason"] == "GLM53_SELFTEST_SCORING=logprobs"
    drop = ("scoring_requested", "scoring_reason")
    assert {k: v for k, v in _scored(auto).items() if k not in drop} == {
        k: v for k, v in _scored(forced).items() if k not in drop
    }
    plan = S.build_plan(PACK)
    ref = S.score(PACK, plan, observe(plan))
    assert auto["prefill_path"] == ref["prefill_path"]
    assert auto["decode_path"] == ref["decode_path"]
    assert auto["decode_rows"] == ref["decode_rows"]
    assert _timeless(auto["tf_rows"]) == _timeless(ref["tf_rows"])
    assert _timeless(auto["smoke"]) == _timeless(ref["smoke"])


def test_startup_hook_speculating_launch_keeps_logprobs_and_checks_mtp(
    tmp_path, monkeypatch, tt_log, speculating_launch
):
    """The n29h launch (MTP K=1, narrow decode): auto keeps the strict logprobs
    scoring (served on the plain path) and adds the MTP identity request --
    the AIME smoke prompt without logprobs, alone, last -- on the speculative
    path; identical tokens = pass."""
    eng = FakeEngine(vllm_config=speculating_launch)
    res = _init(eng, {"GLM53_SELFTEST": "strict"}, tmp_path, monkeypatch).glm53_selftest
    assert res["pass"] and res["scoring"] == "logprobs" and res["errors"] == []
    assert (
        res["scoring_reason"]
        == "auto: logprobs served (the first logprobs request was not refused)"
    )
    assert res["prefill_path"]["top5_cleared"] == 1.0 and res["info"]["logprobs"] == 100
    assert len(eng.calls) == 39 + 1 and [c[2] for c in eng.calls] == [100] * 39 + [None]
    assert eng.paths == ["plain"] * 39 + ["spec"]
    prompt, max_tokens, _, temperature, _ = eng.calls[-1]
    assert prompt == PACK["prompts"][S.AIME_KEY]["prompt_ids"]
    assert (max_tokens, temperature) == (8, 0.0)
    assert len({c[4] for c in eng.calls}) == 40  # distinct request ids
    assert res["requests"] == res["responses"] == 40
    si = res["spec_identity"]
    assert si["checked"] and si["ok"] and si["identical"]
    assert si["speculative"] == "method=mtp, num_speculative_tokens=1"
    assert (
        si["spec_token_ids"]
        == si["plain_token_ids"]
        == (PACK["prompts"][S.AIME_KEY]["golden_ids"][:8])
    )
    assert json.loads((tmp_path / "verdict.json").read_text())["spec_identity"] == si
    assert any(
        "SELFTEST PASS top-1 1.000 / top-5 1.000" in m and "MTP==plain 8/8" in m
        for m in tt_log
    )
    assert any("8/8 identical" in m for m in tt_log)


def test_startup_hook_mtp_path_differing_refuses_to_serve(
    tmp_path, monkeypatch, tt_log, speculating_launch
):
    eng = FakeEngine(vllm_config=speculating_launch, spec_diverge_at=3)
    with pytest.raises(S.SelftestFailed) as exc:
        _init(eng, {}, tmp_path, monkeypatch)  # strict, auto
    assert "MTP path differs from plain path at token 3" in str(exc.value)
    v = json.loads((tmp_path / "verdict.json").read_text())
    si = v["spec_identity"]
    assert not si["ok"] and si["diverge_at"] == 3
    assert si["spec_token_ids"][3] == si["plain_token_ids"][3] + 1
    # the plain path itself is on golden: only the MTP identity fails
    assert v["prefill_path"]["top1_cleared"] == 1.0 and v["reasons"] == [si["reason"]]
    assert any("NOT READY, refusing to serve" in m for m in tt_log)
    # the speculative request errors: FAIL too
    eng = FakeEngine(vllm_config=speculating_launch, spec_fail=True)
    res = _init(eng, {"GLM53_SELFTEST": "warn"}, tmp_path, monkeypatch).glm53_selftest
    assert res["status"] == "fail" and res["responses"] == 39 == res["requests"] - 1
    assert res["errors"] == [
        {
            "request": "spec/aime24_161",
            "error": "RuntimeError: MTP verify timeout",
        }
    ]
    assert res["reasons"] == [
        "1 request error(s)",
        "MTP identity request spec/aime24_161: RuntimeError: MTP verify timeout",
    ]


def test_startup_hook_top1_strict_refuses_below_the_threshold(
    tmp_path, monkeypatch, tt_log, refusing_spec_launch
):
    k = int(len(cleared()) * 0.15) + 1
    eng = FakeEngine(
        miss={"tf": {t: "top5" for t in cleared()[:k]}},
        vllm_config=refusing_spec_launch,
    )
    with pytest.raises(S.SelftestFailed) as exc:
        _init(eng, {}, tmp_path, monkeypatch)  # strict, auto
    assert str(exc.value).startswith("SELFTEST FAIL top-1")
    assert "(top-1-only scoring)" in str(exc.value)
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["scoring"] == "top1" and v["prefill_path"]["top1_cleared"] < 0.85
    assert any("NOT READY, refusing to serve" in m for m in tt_log)


@pytest.mark.parametrize("config_visible", [True, False])
def test_startup_hook_auto_falls_back_after_a_logprobs_refusal(
    tmp_path, monkeypatch, refusing_spec_launch, config_visible
):
    """A speculating launch that refuses logprobs (no narrow decode): the first
    request is refused, the plan re-runs without logprobs (all speculative:
    no MTP identity to measure), the verdict records why -- whether or not
    the hook sees the speculative config."""
    eng = FakeEngine(vllm_config=refusing_spec_launch if config_visible else None)
    res = _init(eng, {"GLM53_SELFTEST": "strict"}, tmp_path, monkeypatch).glm53_selftest
    assert res["pass"] and res["scoring"] == "top1" and res["errors"] == []
    assert res["requests"] == res["responses"] == 39
    att = res["logprobs_attempt"]
    assert att["request"] == "tf/t=0" and att["logprobs"] == 100
    assert att["error"].startswith("ValueError: Speculative decoding on tt cannot")
    assert res["scoring_reason"].startswith(
        "auto: the first logprobs request was refused, re-ran the plan without "
        "logprobs (tf/t=0: ValueError"
    )
    assert [c[2] for c in eng.calls] == [100] + [None] * 39
    assert len({c[4] for c in eng.calls}) == 40  # distinct request ids
    assert json.loads((tmp_path / "verdict.json").read_text())["scoring"] == "top1"
    si = res["spec_identity"]
    if config_visible:
        assert not si["checked"] and si["reason"].startswith("not measurable")
    else:
        assert si is None


def test_startup_hook_forced_logprobs_never_falls_back(
    tmp_path, monkeypatch, refusing_spec_launch
):
    """GLM53_SELFTEST_SCORING=logprobs on a launch refusing logprobs: the
    logprobs scoring as before -- every plan request refused, a failing
    verdict (the MTP identity request has no plain-path reference)."""
    eng = FakeEngine(vllm_config=refusing_spec_launch)
    res = _init(
        eng,
        {"GLM53_SELFTEST": "warn", "GLM53_SELFTEST_SCORING": "logprobs"},
        tmp_path,
        monkeypatch,
    ).glm53_selftest
    assert res["status"] == "fail" and res["scoring"] == "logprobs"
    assert len(res["errors"]) == 39 and len(eng.calls) == 39 + 1
    assert "39 request error(s)" in res["reasons"][0]
    assert res["logprobs_attempt"] is None
    assert "no plain-path reference" in res["spec_identity"]["reason"]


def test_startup_hook_forced_top1_on_a_logprobs_launch(tmp_path, monkeypatch):
    eng = FakeEngine()
    res = _init(
        eng, {"GLM53_SELFTEST_SCORING": "top1"}, tmp_path, monkeypatch
    ).glm53_selftest
    assert res["pass"] and res["scoring"] == "top1"
    assert res["scoring_reason"] == "GLM53_SELFTEST_SCORING=top1"
    assert {c[2] for c in eng.calls} == {None} and len(eng.calls) == 39


def test_startup_hook_strict_refuses_to_serve(tmp_path, monkeypatch, tt_log):
    eng = FakeEngine(miss={"tf": {t: "absent" for t in cleared()[:3]}})
    with pytest.raises(S.SelftestFailed) as exc:
        _init(eng, {}, tmp_path, monkeypatch)  # default mode is strict
    assert "SELFTEST FAIL" in str(exc.value)
    assert json.loads((tmp_path / "verdict.json").read_text())["status"] == "fail"
    assert any("NOT READY, refusing to serve" in m for m in tt_log)


def test_startup_hook_warn_serves_anyway(tmp_path, monkeypatch, tt_log):
    eng = FakeEngine(miss={"tf": {t: "absent" for t in cleared()[:3]}})
    state = _init(eng, {"GLM53_SELFTEST": "warn"}, tmp_path, monkeypatch)
    assert state.glm53_selftest["status"] == "fail"
    assert any("serving anyway" in m for m in tt_log)


def test_startup_hook_off_skips(tmp_path, monkeypatch):
    eng = FakeEngine()
    state = _init(eng, {"GLM53_SELFTEST": "off"}, tmp_path, monkeypatch)
    assert state.glm53_selftest["status"] == "skipped" and eng.calls == []


def test_startup_hook_request_errors_refuse_in_strict(tmp_path, monkeypatch):
    eng = FakeEngine(fail_after=5)
    with pytest.raises(S.SelftestFailed):
        _init(eng, {"GLM53_SELFTEST": "strict"}, tmp_path, monkeypatch)
    res = json.loads((tmp_path / "verdict.json").read_text())
    assert (
        res["status"] == "fail"
        and len(res["errors"]) == 34
        and "RuntimeError: device timeout" in res["errors"][0]["error"]
    )


def test_startup_hook_skips_another_model(tmp_path, monkeypatch):
    eng = FakeEngine()
    eng.model_config.model = "/m/Qwen3-8B"
    monkeypatch.setenv("GLM53_SELFTEST", "strict")
    state = SimpleNamespace()
    asyncio.run(
        E.create_plugin().init_state(
            eng, state, Namespace(model="Qwen3-8B", served_model_name=None)
        )
    )
    assert state.glm53_selftest["status"] == "skipped" and eng.calls == []


def test_startup_hook_logprob_cap_is_an_error(tmp_path, monkeypatch):
    eng = FakeEngine()
    eng.model_config.max_logprobs = 3
    with pytest.raises(S.SelftestFailed, match="could not run"):
        _init(eng, {"GLM53_SELFTEST": "strict"}, tmp_path, monkeypatch)
    state = _init(eng, {"GLM53_SELFTEST": "warn"}, tmp_path, monkeypatch)
    assert state.glm53_selftest["status"] == "error"


def test_selftest_route_and_endpoint_patch():
    from fastapi import FastAPI

    app = FastAPI()
    E.create_plugin().attach_router(app)
    assert any(getattr(r, "path", None) == "/selftest" for r in app.routes)

    import vllm.plugins as P

    saved = P.load_endpoint_plugins
    try:
        P.load_endpoint_plugins = lambda supported_tasks=None: []
        E.install_endpoint_patch()
        patched = P.load_endpoint_plugins
        E.install_endpoint_patch()
        assert P.load_endpoint_plugins is patched  # idempotent
        got = P.load_endpoint_plugins(("generate",))
        assert [p.name for p in got] == [E.PLUGIN_NAME]
        assert P.load_endpoint_plugins(("render",)) == []
        # an installed, allowlisted entry point already loaded it: no duplicate
        P.load_endpoint_plugins = lambda supported_tasks=None: [E.create_plugin()]
        E.install_endpoint_patch()
        assert len(P.load_endpoint_plugins(("generate",))) == 1
    finally:
        P.load_endpoint_plugins = saved


# ------------------------------------------------------- the short-prompt rows (2026-10-25)


def test_short_prompt_rows_gate_finite_logprobs_and_the_first_token():
    """Program 2026-10-25T09:02:00Z (3): cf3h-g's 129-token prompt decoded [2, 2, 2, 2] with non-finite logprobs
    while every first-token grade passed. The short rows (g129 / g257, 8 greedy tokens with logprobs) fail on a
    non-finite logprob anywhere, on a first token outside the fp32 top-5 of the prefix, and on a short answer."""
    plan = S.build_plan(PACK)
    short = [r for r in plan if r.kind == "short"]
    assert [(r.prompt_key, r.length, r.max_tokens) for r in short] == [("g129", 129, 8), ("g257", 257, 8)]
    assert PACK["prompts"]["g129"]["first_top5"][0] == PACK["prompts"]["g129"]["golden_ids"][0]
    obs = observe(plan)
    res = S.score(PACK, plan, obs)
    assert res["pass"] and res["short_ok"] and all(r["logprobs_finite"] and r["ok"] for r in res["short"])
    assert ", short 2/2;" in S.summary_line(res)
    # the cf3h-g defect: a NaN logprob at decode step 3 of g129
    bad = copy.deepcopy(obs)
    bad["short/g129"].top[3][0] = (bad["short/g129"].top[3][0][0], float("nan"))
    res = S.score(PACK, plan, bad)
    row = next(r for r in res["short"] if r["prompt"] == "g129")
    assert not res["pass"] and not row["ok"] and row["logprobs_finite"] is False
    assert any("short-prompt row failed" in r and "g129" in r for r in res["reasons"])
    assert any(ln.startswith("short g129 (len 129): FAIL") and "NON-FINITE" in ln for ln in S.render_table(res))
    # a first token outside the fp32 top-5
    off = copy.deepcopy(obs)
    off["short/g257"].token_ids[0] = 10**6 + 99
    res = S.score(PACK, plan, off)
    assert not res["pass"] and not next(r for r in res["short"] if r["prompt"] == "g257")["first_in_fp32_top5"]
    # a short answer (fewer than 8 tokens)
    cut = copy.deepcopy(obs)
    cut["short/g129"].token_ids = cut["short/g129"].token_ids[:3]
    assert not S.score(PACK, plan, cut)["pass"]


def test_short_prompt_rows_under_top1_scoring_and_an_old_pack():
    """top-1 scoring carries no logprobs: finiteness is not measurable (None, not a fail); a pack without the short
    entries refuses to plan them (rebuild it with build_pack.py)"""
    plan = S.build_plan(PACK)
    res = S.score(PACK, plan, without_logprobs(observe(plan)), scoring="top1")
    assert res["pass"] and all(r["logprobs_finite"] is None and r["ok"] for r in res["short"])
    old = copy.deepcopy(PACK)
    del old["prompts"]["g129"]
    with pytest.raises(S.SelftestError, match="short prompt g129"):
        S.build_plan(old)
    assert len(S.build_plan(old, short_prompts=())) == 37
