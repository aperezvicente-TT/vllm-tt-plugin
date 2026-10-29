# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One-shot GLM-5.3-Flash serving self-test against a running OpenAI endpoint
(HTTP only; run under no-tt-devices). The same plan, goldens and scorer as the
startup hook. Exit 0 = PASS, 1 = FAIL, 2 = could not run.

    python -m vllm_tt_plugin.selftest --url http://127.0.0.1:8000 --out verdict.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from vllm_tt_plugin.selftest.runner import Config, run_http_selftest
from vllm_tt_plugin.selftest.score import (
    DEFAULT_PACK,
    SCORING_MODES,
    SelftestError,
    Thresholds,
    render_table,
    summary_line,
    write_verdict,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m vllm_tt_plugin.selftest", description=__doc__
    )
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--model", default=None, help="served model name (default: first of /v1/models)"
    )
    ap.add_argument("--goldens", default=str(DEFAULT_PACK))
    ap.add_argument("--tf-steps", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--logprobs", type=int, default=100)
    ap.add_argument(
        "--scoring",
        choices=SCORING_MODES,
        default="auto",
        help="logprobs ranks, top-1-only (served greedy token), or auto: "
        "logprobs unless the server refuses them (then the plan is re-run "
        "without logprobs)",
    )
    ap.add_argument(
        "--max-logprobs",
        type=int,
        default=None,
        help="the server's --max-logprobs, if known",
    )
    ap.add_argument(
        "--spec-identity",
        action="store_true",
        help="the server speculates (--speculative-config): add the MTP "
        "identity request (AIME smoke prompt without logprobs == plain path)",
    )
    ap.add_argument("--budget-s", type=float, default=60.0)
    ap.add_argument("--top1", type=float, default=Thresholds.top1)
    ap.add_argument("--top5", type=float, default=Thresholds.top5)
    ap.add_argument("--margin", type=float, default=Thresholds.margin)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--out", default=None, help="verdict JSON path")
    ap.add_argument(
        "--save-observations", default=None, help="record the raw observations (JSON)"
    )
    ap.add_argument(
        "--save-raw",
        default=None,
        help="record the AIME smoke request's raw /v1/completions payload (JSON)",
    )
    ap.add_argument("--quiet", action="store_true", help="no per-position table")
    a = ap.parse_args(argv)
    cfg = replace(
        Config(),
        mode="cli",
        tf_steps=a.tf_steps,
        budget_s=a.budget_s,
        concurrency=max(1, a.concurrency),
        logprobs=a.logprobs,
        pack_path=Path(a.goldens),
        thresholds=Thresholds(top1=a.top1, top5=a.top5, margin=a.margin),
        request_timeout_s=a.timeout_s,
        scoring=a.scoring,
    )
    try:
        res = run_http_selftest(
            a.url,
            cfg,
            model=a.model,
            max_logprobs=a.max_logprobs,
            save_observations=a.save_observations,
            save_raw=a.save_raw,
            speculative="declared by --spec-identity" if a.spec_identity else None,
        )
    except SelftestError as exc:
        print(f"SELFTEST ERROR: {exc}", file=sys.stderr)
        return 2
    if a.out:
        write_verdict(res, a.out)
    if not a.quiet:
        for line in render_table(res):
            print(line)
    print(summary_line(res))
    if a.out:
        print(json.dumps({"verdict": a.out, "status": res["status"]}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
