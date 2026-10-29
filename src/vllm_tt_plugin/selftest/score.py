# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pure-python scoring of the GLM-5.3-Flash serving self-test.

No vLLM, torch or network imports: the same scorer serves the startup hook
(observations from the in-process engine client), the one-shot CLI (observations
from HTTP) and the unit tests (recorded or synthetic observations).

The check is the strict teacher-forcing metric Stage 9 serves with
(doc/ttnn/09-vllm/scripts/serve_checks.py ``tf`` and the coordinator ruling of
2026-10-22 07:19Z): for t = 0..T-1 a FRESH prefill of ``prompt + golden[:t]``
with two greedy tokens and top-K logprobs; generated position 0 is the prefill
path's prediction of ``golden[t]`` (its rank in the served logprobs), generated
position 1 the decode path's prediction of ``golden[t+1]`` when position 0 hit.
Positions whose fp32 top1-top2 logit margin is below ``Thresholds.margin`` are
near-ties a bf16 device may order either way: reported, excluded from the gate.
A greedy smoke fork is a near-tie (passes) when the fp32 margin OR the SERVED
margin at the fork (logp(sampled) - logp(golden) in the served top-K; coordinator
ruling 2026-10-25T12:20:49Z (1)) is below ``Thresholds.margin``; both are kept in
the smoke row.

Two scorings (program ruling 2026-10-22T22:30:25Z): ``logprobs`` (the
above; the default whenever the launch serves logprobs) and ``top1`` for a
launch that refuses them (a speculating TT launch whose model declares no
narrow decode): the same plan requested without logprobs, top-1 = the served
greedy token equals the golden top-1 on the same clear-margin rows, top-5 not
measurable (null, not part of the gate), same top-1 threshold and min-rows rule.

MTP identity (Stage 17's identity rule): on a
speculating launch the logprobs requests decode WITHOUT speculation (the plain
path), so one extra greedy request without logprobs on the AIME smoke prompt
takes the speculative path; its tokens must equal the plain-path smoke row's,
token for token (no near-tie exemption: speculation must be lossless).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PACK_FORMAT = "glm53-selftest-goldens/v1"
DEFAULT_PACK = Path(__file__).with_name("goldens") / "glm53_aime24.json"
AIME_KEY = "aime24_161"
MODES = ("strict", "warn", "off")
# how a verdict is scored: top-K logprob ranks, or the served greedy token only
SCORINGS = ("logprobs", "top1")
SCORING_MODES = ("auto", *SCORINGS)

# pack prompt keys of the greedy smoke, in order
DEFAULT_SMOKE_PROMPTS = (AIME_KEY, "g128", "g512")
# the short-prompt multi-token rows (program 2026-10-25T09:02:00Z (3): cf3h-g's 129-token prompt decoded
# [2, 2, 2, 2] with non-finite logprobs while the first-token-only grades passed): greedy, logprobs, every position
# finite, the first token within the fp32 top-5 of its prefix (the golden prompt's own TF top-k)
DEFAULT_SHORT_PROMPTS = ("g129", "g257")
# golden offsets into the AIME continuation for the decode-after-prefill rows
DEFAULT_DECODE_OFFSETS = (16, 31)
# the plain-path smoke row the MTP identity request repeats without logprobs
SPEC_REFERENCE = f"smoke/{AIME_KEY}"


class SelftestError(RuntimeError):
    """The self-test could not run (bad pack, request errors, ...)."""


class SelftestFailed(RuntimeError):
    """The served stack missed the accuracy gate (strict mode refuses to serve)."""


@dataclass(frozen=True)
class Thresholds:
    top1: float = 0.85
    top5: float = 0.98
    # fp32 top1 - top2 logit margin below which a position is a near-tie (a smoke
    # fork also when its served margin is below it)
    margin: float = 0.5
    # fewer scorable (clear-margin) prefill rows than this = the check did not run
    min_rows: int = 16


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Request:
    name: str
    kind: str  # "tf" | "smoke" | "decode" | "short"
    prompt_key: str
    t: int  # golden tokens appended to the pack prompt
    prompt_ids: tuple[int, ...]
    max_tokens: int

    @property
    def length(self) -> int:
        return len(self.prompt_ids)


@dataclass
class Observation:
    """What the server returned for one request.

    ``top[k]`` is the top-K list of generated position k as (token_id, logprob)
    pairs in any order; ``token_ids`` the greedy tokens.
    """

    token_ids: list[int] = field(default_factory=list)
    top: list[list[tuple[int, float]]] = field(default_factory=list)
    error: str | None = None
    latency_s: float | None = None


# ------------------------------------------------------------------ goldens pack


def load_pack(path: str | Path = DEFAULT_PACK) -> dict[str, Any]:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise SelftestError(f"goldens pack {p}: {exc}") from exc
    pack = json.loads(data)
    if pack.get("format") != PACK_FORMAT:
        raise SelftestError(
            f"goldens pack {p}: format {pack.get('format')!r}, want {PACK_FORMAT!r}"
        )
    for key, entry in pack.get("prompts", {}).items():
        for f in ("prompt_ids", "golden_ids", "margins"):
            if not isinstance(entry.get(f), list) or not entry[f]:
                raise SelftestError(f"goldens pack {p}: prompt {key} lacks {f}")
        if len(entry["margins"]) != len(entry["golden_ids"]):
            raise SelftestError(
                f"goldens pack {p}: prompt {key} has {len(entry['golden_ids'])} "
                f"golden ids but {len(entry['margins'])} margins"
            )
    if AIME_KEY not in pack.get("prompts", {}):
        raise SelftestError(f"goldens pack {p}: no {AIME_KEY} prompt")
    pack["_path"] = str(p)
    pack["_sha256"] = hashlib.sha256(data).hexdigest()
    return pack


def pack_matches_model(pack: dict[str, Any], *names: str | None) -> bool:
    """True when one of the served model names/paths names the pack's model."""
    want = str(pack.get("model", "")).split("/")[-1]
    if not want:
        return True
    return any(want in n for n in names if n)


# ------------------------------------------------------------------ the plan


def build_plan(
    pack: dict[str, Any],
    *,
    tf_steps: int = 32,
    smoke_tokens: int = 8,
    smoke_prompts: tuple[str, ...] = DEFAULT_SMOKE_PROMPTS,
    decode_offsets: tuple[int, ...] = DEFAULT_DECODE_OFFSETS,
    decode_tokens: int = 8,
    short_prompts: tuple[str, ...] = DEFAULT_SHORT_PROMPTS,
    short_tokens: int = 8,
) -> list[Request]:
    """The bounded request list: tf_steps fresh prefills (2 tokens), one greedy
    smoke per prompt (smoke_tokens), one decode-after-prefill request per
    offset (decode_tokens), one short-prompt multi-token row per short prompt
    (short_tokens). Default 32 + 3 + 2 + 2 = 39 requests."""
    prompts = pack["prompts"]
    aime = prompts[AIME_KEY]
    base = tuple(int(x) for x in aime["prompt_ids"])
    gold = [int(x) for x in aime["golden_ids"]]
    if tf_steps > len(gold):
        raise SelftestError(f"tf_steps {tf_steps} > {len(gold)} AIME golden tokens")
    plan: list[Request] = []
    for t in range(tf_steps):
        plan.append(Request(f"tf/t={t}", "tf", AIME_KEY, t, base + tuple(gold[:t]), 2))
    for key in smoke_prompts:
        if key not in prompts:
            raise SelftestError(f"smoke prompt {key} not in the goldens pack")
        if len(prompts[key]["golden_ids"]) < smoke_tokens:
            raise SelftestError(f"smoke prompt {key}: < {smoke_tokens} golden tokens")
        plan.append(
            Request(
                f"smoke/{key}",
                "smoke",
                key,
                0,
                tuple(int(x) for x in prompts[key]["prompt_ids"]),
                smoke_tokens,
            )
        )
    for off in decode_offsets:
        if off + decode_tokens > len(gold):
            raise SelftestError(
                f"decode offset {off} + {decode_tokens} > {len(gold)} golden tokens"
            )
        plan.append(
            Request(
                f"decode/len={len(base) + off}",
                "decode",
                AIME_KEY,
                off,
                base + tuple(gold[:off]),
                decode_tokens,
            )
        )
    for key in short_prompts:
        if key not in prompts or "first_top5" not in prompts[key]:
            raise SelftestError(
                f"short prompt {key} not in the goldens pack (rebuild it: build_pack.py)"
            )
        plan.append(
            Request(
                f"short/{key}",
                "short",
                key,
                0,
                tuple(int(x) for x in prompts[key]["prompt_ids"]),
                short_tokens,
            )
        )
    return plan


def spec_identity_request(plan: list[Request]) -> Request:
    """The MTP identity request: the AIME greedy smoke row again (same prompt,
    same token count), to be sent WITHOUT logprobs so a speculating launch
    decodes it speculatively."""
    smoke = [r for r in plan if r.kind == "smoke"]
    ref = next((r for r in smoke if r.name == SPEC_REFERENCE), None)
    if ref is None:
        if not smoke:
            raise SelftestError("the MTP identity check needs a greedy smoke row")
        ref = smoke[0]
    return Request(
        f"spec/{ref.prompt_key}",
        "spec",
        ref.prompt_key,
        ref.t,
        ref.prompt_ids,
        ref.max_tokens,
    )


# ------------------------------------------------------------------ scoring


def rank_of(top: list[tuple[int, float]], token: int) -> int | None:
    """0-based rank of ``token`` in ``top`` sorted by logprob (ties by id);
    None when absent from the returned top-K."""
    order = sorted(top, key=lambda x: (-float(x[1]), int(x[0])))
    for i, (tid, _) in enumerate(order):
        if int(tid) == token:
            return i
    return None


def served_fork(obs: Observation, k: int, gold: list[int]) -> dict[str, Any] | None:
    """The SERVED margin at a smoke fork (program ruling 2026-10-25T12:20:49Z
    (1): cf3h-h' forked g512 at step 5 where the fp32 gap is 1.01 nats but the
    served gap is 0.00-0.50 on the build that keeps the golden token).

    ``served_margin`` = logp(sampled) - logp(golden) at generated position ``k``;
    a golden token absent from the returned top-K sits below the K-th logprob,
    so the margin is that lower bound (``served_margin_bound``). None when the
    position carries no logprobs or the sampled token is not in its top-K."""
    if k >= len(obs.top) or k >= len(obs.token_ids) or not obs.top[k]:
        return None
    top = {int(t): float(v) for t, v in obs.top[k]}
    s_tok, g_tok = int(obs.token_ids[k]), int(gold[k])
    ls, lg = top.get(s_tok), top.get(g_tok)
    if ls is None:
        return None
    bound = lg is None
    margin = ls - (min(top.values()) if bound else lg)
    return {
        "served_margin": round(abs(margin), 4),
        "served_margin_bound": bound,
        "served_logp_sampled": round(ls, 4),
        "served_logp_golden": None if bound else round(lg, 4),
        "served_top5_at_fork": [
            [t, round(v, 4)] for t, v in sorted(top.items(), key=lambda x: (-x[1], x[0]))[:5]
        ],
    }


def fork_tf_name(prompt_key: str) -> str:
    return f"forktf/{prompt_key}"


def smoke_fork_at(req: Request, obs: Observation | None, gold: list[int]) -> int | None:
    """The first generated position where a smoke row leaves the golden tokens."""
    if obs is None or obs.error:
        return None
    sampled = [int(x) for x in obs.token_ids[: req.max_tokens]]
    k = 0
    while k < len(sampled) and sampled[k] == gold[k]:
        k += 1
    return k if k < len(sampled) else None


def fork_tf_requests(
    pack: dict[str, Any], plan: list[Request], observations: dict[str, Observation]
) -> list[Request]:
    """One teacher-forced prefill per forked smoke row: the pack prompt + the
    golden tokens before the fork, one token, scored by ``fork_tf_tie`` (the
    ruling of 2026-10-25T12:26:33Z: an exact TF tie at the fork is a near-tie)."""
    out = []
    for req in plan:
        if req.kind != "smoke":
            continue
        gold = [int(x) for x in pack["prompts"][req.prompt_key]["golden_ids"]]
        k = smoke_fork_at(req, observations.get(req.name), gold)
        if k is not None:
            out.append(
                Request(
                    fork_tf_name(req.prompt_key),
                    "forktf",
                    req.prompt_key,
                    k,
                    tuple(req.prompt_ids) + tuple(gold[:k]),
                    1,
                )
            )
    return out


def fork_tf_tie(obs: Observation | None, golden: int, sampled: int) -> dict[str, Any] | None:
    """The prefill path's view of a smoke fork: the served logprobs of the
    golden and the sampled token at the fork position, teacher-forced. Exact
    tie = equal logprobs (bf16 logits on one grid point)."""
    if obs is None or obs.error or not obs.top or not obs.top[0]:
        return None
    top = {int(t): float(v) for t, v in obs.top[0]}
    lg, ls = top.get(int(golden)), top.get(int(sampled))
    m = None if lg is None or ls is None else abs(lg - ls)
    return {
        "tf_logp_golden": None if lg is None else round(lg, 4),
        "tf_logp_sampled": None if ls is None else round(ls, 4),
        "tf_margin": None if m is None else round(m, 4),
        "tf_exact_tie": m is not None and m <= 1e-6,
    }


def _rate(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r < k) / max(len(ranks), 1)


def _path(
    rows: list[dict[str, Any]], key: str, thr: Thresholds, scoring: str = "logprobs"
) -> dict[str, Any]:
    """top-1 / top-5 over the rows' ``key`` ranks, all and clear-margin only.
    ``top1`` scoring: the ranks come from the served token alone (0 or None),
    so top-5 is not measured (None)."""
    ranks = [r[key] for r in rows]
    clear = [r[key] for r in rows if r["cleared"]]
    top5_measured = scoring == "logprobs"
    return {
        "n": len(rows),
        "n_cleared": len(clear),
        "top1": round(_rate(ranks, 1), 4),
        "top5": round(_rate(ranks, 5), 4) if top5_measured else None,
        "top1_cleared": round(_rate(clear, 1), 4),
        "top5_cleared": round(_rate(clear, 5), 4) if top5_measured else None,
        "misses_cleared": [r["pos"] for r in rows if r["cleared"] and r[key] != 0],
    }


def _gate(path: dict[str, Any], thr: Thresholds) -> bool:
    if path["top5_cleared"] is None:  # top1 scoring: top-5 is not measured
        return path["top1_cleared"] >= thr.top1
    return path["top1_cleared"] >= thr.top1 and path["top5_cleared"] >= thr.top5


def _gate_reason(name: str, path: dict[str, Any], thr: Thresholds) -> str:
    if path["top5_cleared"] is None:
        return (
            f"{name} top-1 {path['top1_cleared']:.3f} < {thr.top1} (top-1-only scoring)"
        )
    return (
        f"{name} top-1 {path['top1_cleared']:.3f} / "
        f"top-5 {path['top5_cleared']:.3f} "
        f"< {thr.top1} / {thr.top5}"
    )


def served_token_top(obs: Observation) -> Observation:
    """``top1`` scoring: each generated position's 'top-K' is the served greedy
    token alone (K = 1), so ``rank_of`` gives 0 when it is the golden token and
    None otherwise; any returned logprobs are ignored."""
    return Observation(
        token_ids=list(obs.token_ids),
        top=[[(int(t), 0.0)] for t in obs.token_ids],
        error=obs.error,
        latency_s=obs.latency_s,
    )


def _spec_identity(
    pack: dict[str, Any],
    plan: list[Request],
    observations: dict[str, Observation],
    speculative: str,
    spec_observation: Observation | None,
    scoring: str,
) -> dict[str, Any]:
    """The MTP identity row: the speculative (no-logprobs) greedy tokens vs the
    plain-path (logprobs) smoke row of the same prompt. ``ok`` False = a gate
    failure with ``reason``; ``checked`` False = not measurable (not gated)."""
    req = spec_identity_request(plan)
    ref_name = f"smoke/{req.prompt_key}"
    row: dict[str, Any] = {
        "speculative": speculative,
        "request": req.name,
        "reference": ref_name,
        "prompt": req.prompt_key,
        "len": req.length,
        "n": req.max_tokens,
        "checked": False,
        "ok": True,
        "identical": None,
        "diverge_at": None,
        "golden_margin_at_divergence": None,
        "spec_token_ids": None,
        "plain_token_ids": None,
        "latency_s": None,
        "reason": None,
    }
    if scoring != "logprobs":
        row["reason"] = (
            "not measurable: top-1-only scoring sent every request without "
            "logprobs (all on the speculative path), no plain-path reference"
        )
        return row
    row["checked"] = True
    ref = observations.get(ref_name)
    if spec_observation is None or spec_observation.error:
        err = spec_observation.error if spec_observation else "no response"
        row.update(ok=False, reason=f"MTP identity request {req.name}: {err}")
        return row
    row["latency_s"] = spec_observation.latency_s
    if ref is None or ref.error or not ref.token_ids:
        row.update(
            ok=False,
            reason=f"MTP identity: no plain-path reference ({ref_name} failed)",
        )
        return row
    n = req.max_tokens
    spec_ids = [int(x) for x in spec_observation.token_ids[:n]]
    plain_ids = [int(x) for x in ref.token_ids[:n]]
    k = 0
    while k < min(len(spec_ids), len(plain_ids)) and spec_ids[k] == plain_ids[k]:
        k += 1
    identical = k == n and len(spec_ids) == len(plain_ids) == n
    row.update(spec_token_ids=spec_ids, plain_token_ids=plain_ids, identical=identical)
    if not identical:
        margins = pack["prompts"][req.prompt_key]["margins"]
        m = float(margins[k]) if k < len(margins) else None

        def tok(ids: list[int]) -> int | None:
            return ids[k] if k < len(ids) else None

        row.update(
            ok=False,
            diverge_at=k,
            golden_margin_at_divergence=m,
            reason=(
                f"MTP path differs from plain path at token {k} "
                f"(speculative {tok(spec_ids)} vs plain {tok(plain_ids)}; "
                f"{req.prompt_key}, len {req.length})"
            ),
        )
    return row


def score(
    pack: dict[str, Any],
    plan: list[Request],
    observations: dict[str, Observation],
    thr: Thresholds = DEFAULT_THRESHOLDS,
    *,
    duration_s: float | None = None,
    budget_s: float = 60.0,
    mode: str = "strict",
    info: dict[str, Any] | None = None,
    scoring: str = "logprobs",
    scoring_info: dict[str, Any] | None = None,
    speculative: str | None = None,
    spec_observation: Observation | None = None,
) -> dict[str, Any]:
    """The verdict: pass = TF prefill gate AND decode-path gate AND every smoke
    prompt on golden (or forked only at a near-tie) AND no request errors AND
    enough clear-margin rows. Everything is reported, the gate is margin-aware.

    ``scoring="top1"``: the observations carry no logprobs; every rank is the
    served token's (0 = golden, None = not), top-5 is reported as None and is
    not part of the gate. ``scoring_info`` (requested scoring, reason, the
    refused logprobs attempt) is recorded in the verdict.

    ``speculative`` (the launch speculates; its description): the MTP identity
    row -- ``spec_observation`` (the AIME smoke prompt without logprobs) must
    equal the plain-path smoke row token for token, else FAIL. None = no
    speculation, no row (``spec_identity`` null)."""
    if scoring not in SCORINGS:
        raise SelftestError(f"scoring {scoring!r}: want one of {SCORINGS}")
    spec_identity = (
        None
        if speculative is None
        else _spec_identity(
            pack, plan, observations, speculative, spec_observation, scoring
        )
    )
    if scoring == "top1":
        observations = {n: served_token_top(o) for n, o in observations.items()}
    prompts = pack["prompts"]
    tf_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    short_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def margin_at(key: str, pos: int) -> float | None:
        ms = prompts[key]["margins"]
        return float(ms[pos]) if pos < len(ms) else None

    def cleared(m: float | None) -> bool:
        return m is None or m >= thr.margin

    for req in plan:
        obs = observations.get(req.name)
        gold = [int(x) for x in prompts[req.prompt_key]["golden_ids"]]
        if obs is None or obs.error:
            errors.append(
                {"request": req.name, "error": obs.error if obs else "no response"}
            )
            continue
        if len(obs.top) < min(req.max_tokens, 1) or not obs.token_ids:
            errors.append(
                {
                    "request": req.name,
                    "error": f"no logprobs/tokens: {len(obs.top)} positions",
                }
            )
            continue
        if req.kind == "tf":
            t = req.t
            r0 = rank_of(obs.top[0], gold[t])
            m0 = margin_at(req.prompt_key, t)
            row = {
                "t": t,
                "pos": t,
                "len": req.length,
                "golden": gold[t],
                "margin": m0,
                "cleared": cleared(m0),
                "prefill_rank": r0,
                "sampled": int(obs.token_ids[0]),
                "latency_s": obs.latency_s,
            }
            tf_rows.append(row)
            if (
                int(obs.token_ids[0]) == gold[t]
                and t + 1 < len(gold)
                and len(obs.top) > 1
            ):
                r1 = rank_of(obs.top[1], gold[t + 1])
                m1 = margin_at(req.prompt_key, t + 1)
                row.update(decode_rank=r1, decode_margin=m1)
                decode_rows.append(
                    {
                        "src": req.name,
                        "pos": t + 1,
                        "golden": gold[t + 1],
                        "margin": m1,
                        "cleared": cleared(m1),
                        "decode_rank": r1,
                        "sampled": int(obs.token_ids[1])
                        if len(obs.token_ids) > 1
                        else None,
                    }
                )
        elif req.kind == "decode":
            # position k predicts golden[t+k]; rows while the sampled prefix is on
            # golden
            ranks: list[int | None] = []
            for k in range(1, min(req.max_tokens, len(obs.top))):
                if [int(x) for x in obs.token_ids[:k]] != gold[req.t : req.t + k]:
                    break
                pos = req.t + k
                if pos >= len(gold):
                    break
                rk = rank_of(obs.top[k], gold[pos])
                m = margin_at(req.prompt_key, pos)
                ranks.append(rk)
                decode_rows.append(
                    {
                        "src": req.name,
                        "pos": pos,
                        "golden": gold[pos],
                        "margin": m,
                        "cleared": cleared(m),
                        "decode_rank": rk,
                        "sampled": int(obs.token_ids[k])
                        if len(obs.token_ids) > k
                        else None,
                    }
                )
        elif req.kind == "smoke":
            n = req.max_tokens
            sampled = [int(x) for x in obs.token_ids[:n]]
            matched = 0
            while matched < min(n, len(sampled)) and sampled[matched] == gold[matched]:
                matched += 1
            fork_margin = margin_at(req.prompt_key, matched) if matched < n else None
            forked = matched < min(n, len(sampled))
            served = (
                served_fork(obs, matched, gold)
                if forked and scoring == "logprobs"
                else None
            )
            if served is not None:
                tf = fork_tf_tie(
                    observations.get(fork_tf_name(req.prompt_key)),
                    gold[matched],
                    sampled[matched],
                )
                served.update(tf or {"tf_margin": None, "tf_exact_tie": False})
                served["fork_basis"] = "served"
                # a golden token below the served top-K has only a lower bound: never a served near-tie
                near_tie = (
                    not served["served_margin_bound"] and served["served_margin"] < thr.margin
                ) or served["tf_exact_tie"]
            else:
                # top-1 scoring (no logprobs) or an unmeasurable served margin: the fp32 rule
                near_tie = fork_margin is not None and fork_margin < thr.margin
                if forked:
                    served = {"fork_basis": "fp32"}
            row = {
                "prompt": req.prompt_key,
                "len": req.length,
                "n": n,
                "matched": matched,
                "fork_at": None if matched >= n else matched,
                "fork_margin": fork_margin,
                "near_tie": near_tie,
                "ok": matched >= n or near_tie,
                "sampled": sampled,
                "golden": gold[:n],
                "latency_s": obs.latency_s,
            }
            if served is not None:
                row.update(served)
            smoke_rows.append(row)

    for req in plan:  # the short-prompt rows (scored apart: a non-finite logprob is their point)
        if req.kind != "short":
            continue
        obs = observations.get(req.name)
        if obs is None or obs.error or not obs.token_ids:
            continue  # an error / empty response is already an error row (above)
        n = req.max_tokens
        sampled = [int(x) for x in obs.token_ids[:n]]
        lps = [float(lp) for pos in obs.top[:n] for (_t, lp) in pos]
        # top-1 scoring carries no logprobs (served_token_top): finiteness is not measurable there
        finite = None if scoring == "top1" else bool(lps) and all(math.isfinite(x) for x in lps)
        top5 = [int(x) for x in prompts[req.prompt_key]["first_top5"]]
        first_ok = bool(sampled) and sampled[0] in top5
        short_rows.append(
            {
                "prompt": req.prompt_key,
                "len": req.length,
                "n": n,
                "tokens": sampled,
                "first_in_fp32_top5": first_ok,
                "fp32_top5": top5,
                "logprobs_finite": finite,
                "ok": len(sampled) >= n and finite is not False and first_ok,
                "latency_s": obs.latency_s,
            }
        )

    n_requests = len(plan)
    for name in sorted(n for n in observations if n.startswith("forktf/")):
        # the fork TF follow-ups that were sent: counted, a failed one an error row
        n_requests += 1
        if observations[name].error:
            errors.append({"request": name, "error": observations[name].error})
    if spec_identity is not None and spec_identity["checked"]:
        # the MTP identity request was sent: counted, its failure an error row
        n_requests += 1
        if spec_observation is None or spec_observation.error:
            errors.append(
                {
                    "request": spec_identity["request"],
                    "error": spec_observation.error
                    if spec_observation
                    else "no response",
                }
            )
    tf_path = _path(tf_rows, "prefill_rank", thr, scoring)
    dec_path = _path(decode_rows, "decode_rank", thr, scoring)
    tf_gate = _gate(tf_path, thr)
    dec_gate = _gate(dec_path, thr)
    smoke_ok = bool(smoke_rows) and all(r["ok"] for r in smoke_rows)
    short_want = [r for r in plan if r.kind == "short"]
    short_ok = len(short_rows) == len(short_want) and all(r["ok"] for r in short_rows)
    enough = tf_path["n_cleared"] >= thr.min_rows
    reasons = []
    if errors:
        reasons.append(f"{len(errors)} request error(s)")
    if not enough:
        reasons.append(
            f"only {tf_path['n_cleared']} clear-margin prefill rows "
            f"(min {thr.min_rows})"
        )
    if not tf_gate:
        reasons.append(_gate_reason("prefill", tf_path, thr))
    if not dec_gate:
        reasons.append(_gate_reason("decode", dec_path, thr))
    if not smoke_ok:
        bad = [r["prompt"] for r in smoke_rows if not r["ok"]] or ["none ran"]
        reasons.append(f"greedy smoke forked at a clear margin: {bad}")
    if short_want and not short_ok:
        bad = [r["prompt"] for r in short_rows if not r["ok"]] or ["no response"]
        reasons.append(f"short-prompt row failed (non-finite logprobs / short / first token outside fp32 top-5): {bad}")
    if spec_identity is not None and not spec_identity["ok"]:
        reasons.append(spec_identity["reason"])
    passed = not reasons
    within = duration_s is None or duration_s <= budget_s
    return {
        "check": "glm53-selftest/v1",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "reasons": reasons,
        "mode": mode,
        "scoring": scoring,
        "scoring_requested": (scoring_info or {}).get("scoring_requested", scoring),
        "scoring_reason": (scoring_info or {}).get("scoring_reason"),
        "logprobs_attempt": (scoring_info or {}).get("logprobs_attempt"),
        "thresholds": asdict(thr),
        "requests": n_requests,
        "responses": n_requests - len(errors),
        "errors": errors,
        "duration_s": None if duration_s is None else round(duration_s, 2),
        "budget_s": budget_s,
        "within_budget": within,
        "prefill_path": tf_path,
        "decode_path": dec_path,
        "smoke": smoke_rows,
        "smoke_ok": smoke_ok,
        "short": short_rows,
        "short_ok": short_ok,
        "spec_identity": spec_identity,
        "near_ties": [r["t"] for r in tf_rows if not r["cleared"]],
        "tf_rows": tf_rows,
        "decode_rows": decode_rows,
        "goldens": {
            "path": pack.get("_path"),
            "sha256": pack.get("_sha256"),
            "model": pack.get("model"),
        },
        "info": info or {},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ------------------------------------------------------------------ rendering


def summary_line(res: dict[str, Any]) -> str:
    """The one line the acceptance cell greps: SELFTEST PASS|FAIL top-1 x /
    top-5 y ..."""
    p, d = res["prefill_path"], res["decode_path"]
    smoke = f"{sum(1 for r in res['smoke'] if r['ok'])}/{len(res['smoke'])}"
    dur = "?" if res["duration_s"] is None else f"{res['duration_s']:.1f}"

    def t5(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    line = (
        f"SELFTEST {'PASS' if res['pass'] else 'FAIL'} "
        f"top-1 {p['top1_cleared']:.3f} / top-5 "
        f"{t5(p['top5_cleared'])} (prefill, n={p['n_cleared']} of {p['n']}, "
        f"{len(res['near_ties'])} near-ties excluded) "
        f"decode top-1 {d['top1_cleared']:.3f} / top-5 {t5(d['top5_cleared'])} "
        f"(n={d['n_cleared']} of {d['n']}) smoke {smoke}"
        + (f", short {sum(1 for r in res.get('short', []) if r['ok'])}/{len(res.get('short', []))}" if res.get("short") else "")
        + "; "
        f"{res['responses']}/{res['requests']} requests in {dur} s "
        f"(budget {res['budget_s']:.0f} s"
        f"{'' if res['within_budget'] else ', OVER BUDGET'})"
    )
    if res.get("scoring") == "top1":
        line += "; top-1-only scoring (no logprobs on this launch)"
    si = res.get("spec_identity")
    if si and not si["checked"]:
        line += "; MTP identity not measurable"
    elif si and si["ok"]:
        line += f"; MTP==plain {si['n']}/{si['n']}"
    elif si and si["diverge_at"] is not None:
        line += f"; MTP!=plain at token {si['diverge_at']}"
    elif si:
        line += "; MTP identity FAILED"
    if res["reasons"]:
        line += " -- " + "; ".join(res["reasons"])
    return line


def render_table(res: dict[str, Any]) -> list[str]:
    """The per-position table (one line per TF row), the smoke and decode rows."""
    top1_only = res.get("scoring") == "top1"
    lines = [
        "   t  len  golden   margin  prefill  sampled  decode   note",
    ]

    def show(rank: Any) -> str:
        if top1_only and rank != "-":  # the served token: hit or miss, no rank
            return "hit" if rank == 0 else "miss"
        return "none" if rank is None else str(rank)

    for r in res["tf_rows"]:
        note = "" if r["cleared"] else "near-tie"
        if r["cleared"] and r["prefill_rank"] != 0:
            note = (
                "MISS"
                if (r["prefill_rank"] is None or r["prefill_rank"] >= 5)
                else "miss (top-5)"
            )
        dr = r.get("decode_rank", "-")
        margin = "-" if r["margin"] is None else f"{r['margin']:.2f}"
        prank = show(r["prefill_rank"])
        drank = show(dr)
        lines.append(
            f"{r['t']:4d} {r['len']:4d} {r['golden']:7d} {margin:>8} {prank:>8} "
            f"{r['sampled']:8d} {drank:>7}   {note}"
        )
    for r in res["smoke"]:
        m = "-" if r["fork_margin"] is None else f"{r['fork_margin']:.2f}"
        if "served_margin" in r:
            sm = f"{'>=' if r['served_margin_bound'] else ''}{r['served_margin']:.2f}"
            m = f"fp32 {m}, served {sm}"
        if r["ok"] and r["fork_at"] is None:
            desc = f"{r['matched']}/{r['n']} == golden"
        elif r["ok"]:
            desc = f"{r['matched']}/{r['n']} then a near-tie fork (margin {m})"
        else:
            desc = (
                f"FORK at {r['fork_at']} (margin {m}): sampled {r['sampled']} "
                f"golden {r['golden']}"
            )
        lines.append(f"smoke {r['prompt']} (len {r['len']}): {desc}")
    for src in sorted(
        {r["src"] for r in res["decode_rows"] if r["src"].startswith("decode/")}
    ):
        rows = [r for r in res["decode_rows"] if r["src"] == src]
        ranks = [show(r["decode_rank"]) for r in rows]
        lines.append(
            f"{src}: decode {'top-1' if top1_only else 'ranks'} at positions "
            f"{[r['pos'] for r in rows]} = [{', '.join(ranks)}]"
        )
    for r in res.get("short", []):
        fin = {True: "finite", False: "NON-FINITE", None: "n/a (top-1 scoring)"}[r["logprobs_finite"]]
        lines.append(
            f"short {r['prompt']} (len {r['len']}): {'ok' if r['ok'] else 'FAIL'}, tokens {r['tokens']}, "
            f"logprobs {fin}, first token {'in' if r['first_in_fp32_top5'] else 'NOT in'} fp32 top-5 {r['fp32_top5']}"
        )
    si = res.get("spec_identity")
    if si:
        head = (
            f"MTP identity {si['request']} (no logprobs, speculative path) vs "
            f"{si['reference']} (logprobs, plain path), len {si['len']}: "
        )
        if not si["checked"]:
            lines.append(head + str(si["reason"]))
        elif si["ok"]:
            lines.append(head + f"{si['n']}/{si['n']} identical")
        else:
            lines.append(
                head + f"FAIL {si['reason']}: speculative {si['spec_token_ids']} "
                f"plain {si['plain_token_ids']}"
            )
    for e in res["errors"]:
        lines.append(f"ERROR {e['request']}: {e['error']}")
    if top1_only:
        lines.append(f"scoring top1: {res.get('scoring_reason')}")
    return lines


def write_verdict(res: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1) + "\n")
