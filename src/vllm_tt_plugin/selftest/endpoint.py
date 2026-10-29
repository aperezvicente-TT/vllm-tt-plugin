# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The startup self-test as a vLLM endpoint plugin.

``init_state`` runs inside ``init_app_state`` -- after the engine is up (model
loaded, warm-up and trace capture done) and BEFORE ``serve_http`` binds the
socket, so nothing can reach /health or /v1/* until the served stack has
proven its accuracy. In strict mode a miss raises: ``build_and_serve`` unwinds,
the engine shuts down cleanly and the server process exits non-zero (the
supervisor restarts or escalates). ``GET /selftest`` returns the verdict
(200 pass, 503 fail in warn mode, 200 with status skipped when off).
``GLM53_SELFTEST_SCORING`` (auto | logprobs | top1) picks logprob-rank or
top-1-only scoring; auto falls back to top-1-only only when the server refuses
logprobs -- the verdict records the scoring and why. A speculating launch
(the engine's speculative_config) adds the MTP identity request: the AIME
smoke prompt without logprobs (speculative path) == the plain-path smoke row.

Loaded two ways: the ``vllm.endpoint_plugins`` entry point (installed plugin,
needs ``VLLM_PLUGINS`` to name it: vLLM's opt-in rule for endpoint plugins)
and, for the PYTHONPATH-first serving trees that never reinstall, the
``_install_glm53_selftest_endpoint_patch`` hook that appends the plugin from
``load_endpoint_plugins`` (see platform.py; dedupes by name).
"""

from __future__ import annotations

from argparse import Namespace
from functools import wraps
from typing import Any

from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.selftest.runner import Config, run_startup_selftest
from vllm_tt_plugin.selftest.score import (
    SelftestError,
    SelftestFailed,
    load_pack,
    pack_matches_model,
    render_table,
    summary_line,
    write_verdict,
)

logger = init_tt_logger(__name__)

PLUGIN_NAME = "glm53_selftest"
STATE_ATTR = "glm53_selftest"
LOG_TAG = "[glm53-selftest]"


class Glm53SelftestEndpointPlugin:
    name = PLUGIN_NAME
    required_tasks = ("generate",)

    def attach_router(self, app: Any) -> None:
        from fastapi.responses import JSONResponse

        async def selftest_verdict():
            res = getattr(app.state, STATE_ATTR, None) or {"status": "not run"}
            code = 200 if res.get("status") in ("pass", "skipped") else 503
            return JSONResponse(content=res, status_code=code)

        app.add_api_route(
            "/selftest", selftest_verdict, methods=["GET"], name="glm53_selftest"
        )

    async def init_state(self, engine_client: Any, state: Any, args: Namespace) -> None:
        cfg = Config.from_env()
        setattr(state, STATE_ATTR, {"status": "running", "mode": cfg.mode})
        if cfg.mode == "off":
            res = {"status": "skipped", "mode": "off", "reason": "GLM53_SELFTEST=off"}
            logger.warning(
                "%s skipped: GLM53_SELFTEST=off -- this server's accuracy is "
                "NOT proven",
                LOG_TAG,
            )
            setattr(state, STATE_ATTR, res)
            return
        if engine_client is None:
            res = {
                "status": "skipped",
                "mode": cfg.mode,
                "reason": "no engine (render server)",
            }
            setattr(state, STATE_ATTR, res)
            return
        model_config = getattr(engine_client, "model_config", None)
        model_path = str(getattr(model_config, "model", "") or "")
        served = getattr(args, "served_model_name", None) or []
        served = list(served) if isinstance(served, (list, tuple)) else [served]
        served_name = str(served[0] if served else getattr(args, "model", "") or "")
        max_logprobs = getattr(model_config, "max_logprobs", None)
        try:
            pack = load_pack(cfg.pack_path)
            if not pack_matches_model(pack, model_path, served_name):
                res = {
                    "status": "skipped",
                    "mode": cfg.mode,
                    "reason": f"goldens pack is for {pack.get('model')}, "
                    f"served {served_name or model_path}",
                }
                logger.warning("%s skipped: %s", LOG_TAG, res["reason"])
                setattr(state, STATE_ATTR, res)
                return
            logger.info(
                "%s starting: mode %s, scoring %s, %d TF positions + 3 greedy "
                "smoke + 2 decode-after-prefill, budget %.0f s, "
                "goldens %s (sha256 %s)",
                LOG_TAG,
                cfg.mode,
                cfg.scoring,
                cfg.tf_steps,
                cfg.budget_s,
                pack["_path"],
                pack["_sha256"][:16],
            )
            res = await run_startup_selftest(
                engine_client,
                cfg,
                max_logprobs=max_logprobs,
                info={"model": served_name, "model_path": model_path},
            )
        except SelftestError as exc:
            res = {
                "status": "error",
                "pass": False,
                "mode": cfg.mode,
                "reasons": [str(exc)],
                "error": str(exc),
            }
            logger.error("%s ERROR: %s", LOG_TAG, exc)
            setattr(state, STATE_ATTR, res)
            if cfg.mode == "strict":
                raise SelftestFailed(f"startup self-test could not run: {exc}") from exc
            return
        setattr(state, STATE_ATTR, res)
        if cfg.out_path:
            try:
                write_verdict(res, cfg.out_path)
                logger.info("%s verdict written to %s", LOG_TAG, cfg.out_path)
            except OSError as exc:
                logger.warning("%s could not write %s: %s", LOG_TAG, cfg.out_path, exc)
        for line in render_table(res):
            logger.info("%s %s", LOG_TAG, line)
        if res.get("scoring") == "top1":
            logger.warning(
                "%s top-1-only scoring: %s -- top-5 is not measured on this "
                "launch, the gate is top-1 >= %.2f on clear-margin rows",
                LOG_TAG,
                res.get("scoring_reason"),
                cfg.thresholds.top1,
            )
        line = summary_line(res)
        if res["pass"]:
            logger.info("%s %s", LOG_TAG, line)
            return
        if cfg.mode == "warn":
            logger.warning(
                "%s %s (GLM53_SELFTEST=warn: serving anyway, /selftest reports 503)",
                LOG_TAG,
                line,
            )
            return
        logger.error(
            "%s %s -- NOT READY, refusing to serve (GLM53_SELFTEST=strict)",
            LOG_TAG,
            line,
        )
        raise SelftestFailed(line)


def create_plugin() -> Glm53SelftestEndpointPlugin:
    """``vllm.endpoint_plugins`` entry point factory."""
    return Glm53SelftestEndpointPlugin()


def install_endpoint_patch() -> None:
    """Append the self-test plugin to vLLM's endpoint plugin list even when the
    entry point is not installed / allowlisted (PYTHONPATH-first trees).
    Idempotent; only the API server process ever calls load_endpoint_plugins."""
    import vllm.plugins as vllm_plugins

    original = vllm_plugins.load_endpoint_plugins
    if getattr(original, "_glm53_selftest_patch", False):
        return

    @wraps(original)
    def load_endpoint_plugins(supported_tasks=None):
        plugins = list(original(supported_tasks))
        if any(getattr(p, "name", None) == PLUGIN_NAME for p in plugins):
            return plugins
        if supported_tasks is not None and "generate" not in supported_tasks:
            return plugins
        plugins.append(create_plugin())
        logger.info(
            "Loaded endpoint plugin %s (GLM-5.3-Flash startup self-test)", PLUGIN_NAME
        )
        return plugins

    load_endpoint_plugins._glm53_selftest_patch = True  # type: ignore[attr-defined]
    vllm_plugins.load_endpoint_plugins = load_endpoint_plugins
