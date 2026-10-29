# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Drivers that collect the self-test's observations: in-process through the
engine client (the startup hook, before the HTTP server binds) and over HTTP
(the one-shot CLI against a running endpoint). Both feed ``score.score``.

Scoring (``GLM53_SELFTEST_SCORING`` = auto | logprobs | top1, default auto):
auto sends the first plan request with logprobs; answered -> the logprobs
scoring, refused for asking for logprobs (a speculating TT launch whose model
declares no narrow decode) -> the whole plan re-runs without logprobs, scored
top-1-only. A ``--max-logprobs`` below 5 stays an error (serve with -1).

MTP identity (speculative decoding configured, logprobs scoring): the logprobs
requests decode on the plain path, so one more greedy request WITHOUT logprobs
on the AIME smoke prompt -- sent alone, after the plan, so no logprobs row
shares its steps -- takes the speculative path; score.py requires it to equal
the plain-path smoke row."""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_tt_plugin.selftest.score import (
    AIME_KEY,
    DEFAULT_PACK,
    MODES,
    SCORING_MODES,
    Observation,
    Request,
    SelftestError,
    Thresholds,
    build_plan,
    fork_tf_requests,
    load_pack,
    score,
    spec_identity_request,
)

ENV_MODE = "GLM53_SELFTEST"
ENV_SCORING = "GLM53_SELFTEST_SCORING"


@dataclass(frozen=True)
class Config:
    mode: str = "strict"
    tf_steps: int = 32
    budget_s: float = 60.0
    concurrency: int = 1
    logprobs: int = 100
    pack_path: Path = DEFAULT_PACK
    out_path: Path | None = None
    thresholds: Thresholds = Thresholds()
    # per-request timeout (s): a stuck device must not hang the startup forever
    request_timeout_s: float = 600.0
    # auto | logprobs | top1 (score.SCORING_MODES)
    scoring: str = "auto"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        e = os.environ if env is None else env
        mode = e.get(ENV_MODE, "strict").strip().lower() or "strict"
        if mode not in MODES:
            raise SelftestError(f"{ENV_MODE}={mode!r}: want one of {MODES}")
        scoring = e.get(ENV_SCORING, "auto").strip().lower() or "auto"
        if scoring not in SCORING_MODES:
            raise SelftestError(
                f"{ENV_SCORING}={scoring!r}: want one of {SCORING_MODES}"
            )
        thr = Thresholds(
            top1=float(e.get("GLM53_SELFTEST_TOP1", Thresholds.top1)),
            top5=float(e.get("GLM53_SELFTEST_TOP5", Thresholds.top5)),
            margin=float(e.get("GLM53_SELFTEST_MARGIN", Thresholds.margin)),
            min_rows=int(e.get("GLM53_SELFTEST_MIN_ROWS", Thresholds.min_rows)),
        )
        out = e.get("GLM53_SELFTEST_OUT") or None
        return cls(
            mode=mode,
            tf_steps=int(e.get("GLM53_SELFTEST_TF_STEPS", cls.tf_steps)),
            budget_s=float(e.get("GLM53_SELFTEST_BUDGET_S", cls.budget_s)),
            concurrency=max(
                1, int(e.get("GLM53_SELFTEST_CONCURRENCY", cls.concurrency))
            ),
            logprobs=int(e.get("GLM53_SELFTEST_LOGPROBS", cls.logprobs)),
            pack_path=Path(e.get("GLM53_SELFTEST_GOLDENS") or DEFAULT_PACK),
            out_path=Path(out) if out else None,
            thresholds=thr,
            request_timeout_s=float(
                e.get("GLM53_SELFTEST_REQUEST_TIMEOUT_S", cls.request_timeout_s)
            ),
            scoring=scoring,
        )


def effective_logprobs(wanted: int, max_logprobs: int | None) -> int:
    """top-K to request: the server's max_logprobs caps it (-1 = unlimited);
    fewer than 5 cannot score top-5."""
    k = wanted
    if max_logprobs is not None and max_logprobs >= 0:
        k = min(wanted, max_logprobs)
    if k < 5:
        raise SelftestError(
            f"the server allows {max_logprobs} logprobs per position; "
            "the self-test needs >= 5 "
            "(serve with --max-logprobs -1 like Stage 9, or >= 100)"
        )
    return k


# ------------------------------------------------------------------ scoring choice


def speculative_config_desc(engine_client: Any = None) -> str | None:
    """The launch's speculative decoding (``engine_client.vllm_config``), or
    None when it does not speculate (or the client carries no config)."""
    vllm_config = getattr(engine_client, "vllm_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return None
    return (
        f"method={getattr(spec, 'method', None)}, "
        f"num_speculative_tokens={getattr(spec, 'num_speculative_tokens', None)}"
    )


def choose_scoring(requested: str) -> tuple[str, str]:
    """(scoring, why) before any request is sent: auto starts with logprobs
    (``collect`` switches on a refusal)."""
    if requested == "logprobs":
        return "logprobs", f"{ENV_SCORING}=logprobs"
    if requested == "top1":
        return "top1", f"{ENV_SCORING}=top1"
    if requested != "auto":
        raise SelftestError(f"scoring {requested!r}: want one of {SCORING_MODES}")
    return (
        "logprobs",
        "auto: logprobs served (the first logprobs request was not refused)",
    )


def is_logprobs_refusal(obs: Observation | None) -> bool:
    """The request was refused for asking for logprobs (the TT platform's
    speculative-decoding refusal: a ValueError in-process, HTTP 400 over the
    OpenAI API), not failed while being served."""
    err = (obs.error or "") if obs is not None else ""
    return "cannot serve this request" in err and "logprobs" in err


def _fallback_reason(name: str, obs: Observation) -> str:
    return (
        "auto: the first logprobs request was refused, re-ran the plan "
        f"without logprobs ({name}: {obs.error})"
    )


# ------------------------------------------------------------------ engine (in-process)


def _positions_from_engine(logprobs: Any) -> list[list[tuple[int, float]]]:
    """vLLM's SampleLogprobs (list[dict[int, Logprob]] or FlatLogprobs) -> top lists."""
    out: list[list[tuple[int, float]]] = []
    if logprobs is None:
        return out
    for i in range(len(logprobs)):
        pos = logprobs[i]
        if not pos:
            out.append([])
            continue
        out.append(
            [
                (int(tid), float(getattr(lp, "logprob", lp)))
                for tid, lp in dict(pos).items()
            ]
        )
    return out


async def run_engine(
    engine_client: Any,
    plan: list[Request],
    cfg: Config,
    *,
    logprobs: int | None,
    first_index: int = 0,
) -> tuple[dict[str, Observation], float]:
    """Every planned request through ``engine_client.generate`` (the AsyncLLM the
    OpenAI handlers use: scheduler, prefill admission, host logprobs sampler --
    the served stack), greedy, ``logprobs`` top-K (None: no logprobs, the
    top-1-only scoring). Returns observations and the wall time of the batch.
    ``first_index``: the request-id index of ``plan[0]`` (a plan run in parts
    keeps distinct ids)."""
    from vllm.sampling_params import SamplingParams

    sem = asyncio.Semaphore(cfg.concurrency)
    obs: dict[str, Observation] = {}
    tag = f"glm53-selftest-{int(time.time())}"
    if logprobs is None:
        tag = f"glm53-selftest-nolp-{int(time.time())}"

    async def one(i: int, req: Request) -> None:
        params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=0.0,
            logprobs=logprobs,
            ignore_eos=True,
        )
        t0 = time.perf_counter()
        async with sem:
            try:
                final = None
                gen = engine_client.generate(
                    {"prompt_token_ids": list(req.prompt_ids)}, params, f"{tag}-{i}"
                )

                async def drain():
                    nonlocal final
                    async for out in gen:
                        final = out

                await asyncio.wait_for(drain(), timeout=cfg.request_timeout_s)
                if final is None or not final.outputs:
                    obs[req.name] = Observation(
                        error="no output", latency_s=time.perf_counter() - t0
                    )
                    return
                co = final.outputs[0]
                obs[req.name] = Observation(
                    token_ids=[int(x) for x in co.token_ids],
                    top=_positions_from_engine(co.logprobs),
                    latency_s=time.perf_counter() - t0,
                )
            except Exception as exc:  # noqa: BLE001 - every failure is a verdict row
                obs[req.name] = Observation(
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    latency_s=time.perf_counter() - t0,
                )

    t_start = time.perf_counter()
    await asyncio.gather(*(one(i, r) for i, r in enumerate(plan, first_index)))
    return obs, time.perf_counter() - t_start


@dataclass
class Collected:
    observations: dict[str, Observation]
    duration_s: float
    logprobs: int | None  # top-K the scored requests asked for (None: none)
    scoring: dict[str, Any]  # score(scoring_info=...)
    speculative: str | None = None
    spec_observation: Observation | None = None

    def score_kwargs(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "scoring": self.scoring["scoring"],
            "scoring_info": self.scoring,
            "speculative": self.speculative,
            "spec_observation": self.spec_observation,
        }


async def collect(
    run: Any,
    plan: list[Request],
    cfg: Config,
    *,
    max_logprobs: int | None,
    speculative: str | None = None,
    pack: dict[str, Any] | None = None,
) -> Collected:
    """The plan's observations under the chosen scoring. ``run(plan, logprobs,
    first_index)`` -> (observations, seconds) is the driver (engine or HTTP).
    auto sends ``plan[0]`` first: refused for asking for logprobs -> the whole
    plan again without them. ``speculative`` (the launch speculates) with the
    logprobs scoring: the MTP identity request last, alone, without logprobs.
    ``pack`` with the logprobs scoring: one teacher-forced prefill per forked
    smoke row after the plan (``fork_tf_requests``; ruling 2026-10-25T12:26:33Z)."""
    scoring, why = choose_scoring(cfg.scoring)
    attempt: dict[str, Any] | None = None
    k: int | None = None
    dur = 0.0
    obs: dict[str, Observation] = {}
    if scoring == "logprobs":
        k = effective_logprobs(cfg.logprobs, max_logprobs)
        if cfg.scoring == "auto" and plan:
            obs, dur = await run(plan[:1], k, 0)
            first = obs.get(plan[0].name)
            if first is not None and is_logprobs_refusal(first):
                attempt = {
                    "logprobs": k,
                    "request": plan[0].name,
                    "error": first.error,
                    "duration_s": round(dur, 3),
                }
                scoring, why = "top1", _fallback_reason(plan[0].name, first)
            else:
                rest, dur_rest = await run(plan[1:], k, 1)
                obs.update(rest)
                dur += dur_rest
        else:
            obs, dur = await run(plan, k, 0)
    if scoring == "top1":
        k = None
        obs, dur_top1 = await run(plan, None, 0)
        dur += dur_top1
    if scoring == "logprobs" and pack is not None:
        fork_tf = fork_tf_requests(pack, plan, obs)
        if fork_tf:
            fobs, dur_fork = await run(fork_tf, k, len(plan) + 1)
            obs.update(fobs)
            dur += dur_fork
    spec_obs = None
    if speculative is not None and scoring == "logprobs":
        sreq = spec_identity_request(plan)
        sobs, dur_spec = await run([sreq], None, len(plan))
        spec_obs = sobs.get(sreq.name)
        dur += dur_spec
    info = {
        "scoring": scoring,
        "scoring_requested": cfg.scoring,
        "scoring_reason": why,
        "logprobs_attempt": attempt,
    }
    return Collected(obs, dur, k, info, speculative, spec_obs)


async def run_startup_selftest(
    engine_client: Any,
    cfg: Config,
    *,
    info: dict[str, Any] | None = None,
    max_logprobs: int | None = None,
) -> dict[str, Any]:
    pack = load_pack(cfg.pack_path)
    plan = build_plan(pack, tf_steps=cfg.tf_steps)

    async def run(sub: list[Request], k: int | None, first_index: int):
        return await run_engine(
            engine_client, sub, cfg, logprobs=k, first_index=first_index
        )

    got = await collect(
        run,
        plan,
        cfg,
        max_logprobs=max_logprobs,
        speculative=speculative_config_desc(engine_client),
        pack=pack,
    )
    res = score(
        pack,
        plan,
        got.observations,
        cfg.thresholds,
        budget_s=cfg.budget_s,
        mode=cfg.mode,
        info={
            **(info or {}),
            "logprobs": got.logprobs,
            "concurrency": cfg.concurrency,
            "driver": "engine",
        },
        **got.score_kwargs(),
    )
    return res


# ------------------------------------------------------------------ HTTP (one-shot CLI)


def _http_json(
    url: str, body: dict[str, Any] | None, timeout: float
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - local endpoint
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read().decode()[:500]
        except Exception:  # noqa: BLE001
            payload = ""
        return exc.code, payload


def served_model_name(url: str, timeout: float = 30.0) -> str:
    try:
        status, models = _http_json(url.rstrip("/") + "/v1/models", None, timeout)
    except Exception as exc:
        raise SelftestError(
            f"GET {url}/v1/models: {type(exc).__name__}: {exc}"
        ) from exc
    if status != 200 or not isinstance(models, dict) or not models.get("data"):
        raise SelftestError(f"GET /v1/models: {status} {str(models)[:200]}")
    return str(models["data"][0]["id"])


def parse_completion(payload: dict[str, Any]) -> Observation:
    """A /v1/completions response (``logprobs`` N, ``return_tokens_as_token_ids``)
    -> Observation. Keys of top_logprobs are ``token_id:<n>``."""
    choice = payload["choices"][0]
    lp = choice.get("logprobs") or {}
    tops = lp.get("top_logprobs") or []
    top: list[list[tuple[int, float]]] = []
    for d in tops:
        top.append(
            [(int(str(k).split(":")[-1]), float(v)) for k, v in (d or {}).items()]
        )
    ids = choice.get("token_ids")
    if not ids:
        ids = [int(str(t).split(":")[-1]) for t in (lp.get("tokens") or [])]
    return Observation(token_ids=[int(x) for x in ids], top=top)


def run_http(
    url: str,
    model: str,
    plan: list[Request],
    cfg: Config,
    *,
    logprobs: int | None,
    save_raw: str | Path | None = None,
) -> tuple[dict[str, Observation], float]:
    """``save_raw``: record the AIME smoke request's raw /v1/completions payload
    (a parse fixture) -- no extra request. ``logprobs`` None: the body asks for
    none (top-1-only scoring)."""
    base = url.rstrip("/")
    obs: dict[str, Observation] = {}

    def one(req: Request) -> None:
        body = {
            "model": model,
            "prompt": list(req.prompt_ids),
            "max_tokens": req.max_tokens,
            "temperature": 0.0,
            "logprobs": logprobs,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "ignore_eos": True,
        }
        if logprobs is None:
            del body["logprobs"]
        t0 = time.perf_counter()
        try:
            status, payload = _http_json(
                base + "/v1/completions", body, cfg.request_timeout_s
            )
            if status != 200:
                obs[req.name] = Observation(
                    error=f"HTTP {status}: {str(payload)[:200]}",
                    latency_s=time.perf_counter() - t0,
                )
                return
            o = parse_completion(payload)
            o.latency_s = time.perf_counter() - t0
            obs[req.name] = o
            if save_raw and req.name == f"smoke/{AIME_KEY}":
                p = Path(save_raw)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(payload) + "\n")
        except Exception as exc:  # noqa: BLE001
            obs[req.name] = Observation(
                error=f"{type(exc).__name__}: {exc}"[:300],
                latency_s=time.perf_counter() - t0,
            )

    t_start = time.perf_counter()
    if cfg.concurrency <= 1:
        for r in plan:
            one(r)
    else:
        with cf.ThreadPoolExecutor(cfg.concurrency) as ex:
            list(ex.map(one, plan))
    return obs, time.perf_counter() - t_start


def run_http_selftest(
    url: str,
    cfg: Config,
    *,
    model: str | None = None,
    max_logprobs: int | None = None,
    info: dict[str, Any] | None = None,
    save_observations: str | Path | None = None,
    save_raw: str | Path | None = None,
    speculative: str | None = None,
) -> dict[str, Any]:
    """The HTTP driver sees no engine config: ``speculative`` (the caller
    knows the server speculates, CLI ``--spec-identity``) adds the MTP
    identity request."""
    pack = load_pack(cfg.pack_path)
    plan = build_plan(pack, tf_steps=cfg.tf_steps)
    model = model or served_model_name(url)

    async def run(sub: list[Request], k: int | None, first_index: int):
        return run_http(url, model, sub, cfg, logprobs=k, save_raw=save_raw)

    got = asyncio.run(
        collect(
            run, plan, cfg, max_logprobs=max_logprobs, speculative=speculative, pack=pack
        )
    )
    obs = got.observations
    if save_observations:
        p = Path(save_observations)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "plan": [r.name for r in plan],
                    "scoring": got.scoring["scoring"],
                    "spec_observation": None
                    if got.spec_observation is None
                    else {
                        "token_ids": got.spec_observation.token_ids,
                        "error": got.spec_observation.error,
                        "latency_s": got.spec_observation.latency_s,
                    },
                    "observations": {
                        n: {
                            "token_ids": o.token_ids,
                            "top": o.top,
                            "error": o.error,
                            "latency_s": o.latency_s,
                        }
                        for n, o in obs.items()
                    },
                }
            )
            + "\n"
        )
    return score(
        pack,
        plan,
        obs,
        cfg.thresholds,
        budget_s=cfg.budget_s,
        mode=cfg.mode,
        info={
            **(info or {}),
            "url": url,
            "model": model,
            "logprobs": got.logprobs,
            "concurrency": cfg.concurrency,
            "driver": "http",
        },
        **got.score_kwargs(),
    )


def load_observations(path: str | Path) -> dict[str, Observation]:
    d = json.loads(Path(path).read_text())
    return {
        n: Observation(
            token_ids=o.get("token_ids") or [],
            top=[[(int(t), float(lp)) for t, lp in pos] for pos in o.get("top") or []],
            error=o.get("error"),
            latency_s=o.get("latency_s"),
        )
        for n, o in d["observations"].items()
    }
