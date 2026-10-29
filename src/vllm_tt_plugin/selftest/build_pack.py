# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the self-test goldens pack (a few KB of token ids + fp32 margins) from
the program's fp32 GPU goldens. Needs torch; run once, commit the JSON.

    python -m vllm_tt_plugin.selftest.build_pack --ws <bringup repo> \
        --out goldens/glm53_aime24.json

Prompts: the AIME24 chat prompt (161 tokens; goldens_gpu_fp32_aime24: 100 greedy
continuation steps, continuation_logits [100, V] fp32) and the 128 / 512-token
golden prompts (goldens_gpu_fp32: 256 greedy steps). margin[t] = fp32 top1 -
top2 logit of the continuation position t (the prediction of golden[t] from
prompt + golden[:t]); the AIME margins are cross-checked against Stage 9's
doc/ttnn/09-vllm/scripts/aime24_fp32_margins.json when present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from vllm_tt_plugin.selftest.score import PACK_FORMAT


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry(tf_path: Path, logits_path: Path, n: int, torch) -> tuple[dict, dict]:
    tf = torch.load(tf_path, map_location="cpu", weights_only=False)
    lg = torch.load(logits_path, map_location="cpu", weights_only=False)
    prompt_ids = tf["prompt_ids"].reshape(-1).tolist()
    targets = tf["targets"].reshape(-1).tolist()
    cont_ids = lg["continuation_ids"].reshape(-1).tolist()
    cl = lg["continuation_logits"].float()
    if targets[: len(cont_ids)] != cont_ids[: len(targets)]:
        raise SystemExit(f"{tf_path}: targets != {logits_path} continuation_ids")
    n = min(n, len(cont_ids), cl.shape[0])
    top2 = torch.topk(cl[:n], 2, dim=-1).values
    argmax = cl[:n].argmax(-1).tolist()
    if argmax != cont_ids[:n]:
        raise SystemExit(
            f"{logits_path}: continuation_logits argmax != continuation_ids"
        )
    margins = [round(float(m), 4) for m in (top2[:, 0] - top2[:, 1]).tolist()]
    entry = {
        "prompt_ids": [int(x) for x in prompt_ids],
        "golden_ids": [int(x) for x in cont_ids[:n]],
        "margins": margins,
        "source": {
            "teacher_forcing": str(tf_path),
            "logits": str(logits_path),
        },
    }
    srcs = {str(tf_path): _sha(tf_path), str(logits_path): _sha(logits_path)}
    return entry, srcs


def _short_entry(tf_path: Path, logits_path: Path, n: int, torch) -> tuple[dict, dict]:
    """A short-prompt row (score.py "short"; program 2026-10-25T09:02:00Z (3)): the golden prompt's first n ids
    and the fp32 top-5 of its next token (prompt_top32 at position n - 1: the TF prediction from ids[:n]);
    golden_ids / margins = that one position (no greedy continuation exists for a prefix)"""
    tf = torch.load(tf_path, map_location="cpu", weights_only=False)
    lg = torch.load(logits_path, map_location="cpu", weights_only=False)
    ids = tf["prompt_ids"].reshape(-1).tolist()
    idx = lg["prompt_top32_indices"][n - 1].tolist()
    val = lg["prompt_top32_values"][n - 1].float().tolist()
    entry = {
        "prompt_ids": [int(x) for x in ids[:n]],
        "golden_ids": [int(idx[0])],
        "margins": [round(float(val[0] - val[1]), 4)],
        "first_top5": [int(x) for x in idx[:5]],
        "source": {"teacher_forcing": str(tf_path), "logits": str(logits_path), "position": n - 1},
    }
    srcs = {str(tf_path): _sha(tf_path), str(logits_path): _sha(logits_path)}
    return entry, srcs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ws", required=True, help="the bringup repo (goldens_gpu_fp32*/ under it)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--aime-steps",
        type=int,
        default=48,
        help="AIME golden tokens kept (>= 32 TF + 8 decode + 8)",
    )
    ap.add_argument(
        "--smoke-steps",
        type=int,
        default=16,
        help="golden tokens kept for the 128 / 512 prompts",
    )
    a = ap.parse_args(argv)
    import torch

    ws = Path(a.ws)
    prompts, sources = {}, {}
    e, s = _entry(
        ws / "goldens_gpu_fp32_aime24/teacher_forcing/161.pt",
        ws / "goldens_gpu_fp32_aime24/logits/161.pt",
        a.aime_steps,
        torch,
    )
    prompts["aime24_161"] = e
    sources.update(s)
    for T in (128, 512):
        e, s = _entry(
            ws / f"goldens_gpu_fp32/teacher_forcing/{T}.pt",
            ws / f"goldens_gpu_fp32/logits/{T}.pt",
            a.smoke_steps,
            torch,
        )
        prompts[f"g{T}"] = e
        sources.update(s)
    for n in (129, 257):  # the short-prompt rows (score.DEFAULT_SHORT_PROMPTS)
        e, s = _short_entry(
            ws / "goldens_gpu_fp32/teacher_forcing/2048.pt",
            ws / "goldens_gpu_fp32/logits/2048.pt",
            n,
            torch,
        )
        prompts[f"g{n}"] = e
        sources.update(s)
    check = ws / "doc/ttnn/09-vllm/scripts/aime24_fp32_margins.json"
    if check.exists():
        ref = json.loads(check.read_text())
        n = min(len(ref["margins"]), len(prompts["aime24_161"]["margins"]))
        diff = max(
            abs(x - y)
            for x, y in zip(ref["margins"][:n], prompts["aime24_161"]["margins"][:n])
        )
        if (
            diff > 1e-3
            or ref["continuation_ids"][:n] != prompts["aime24_161"]["golden_ids"][:n]
        ):
            raise SystemExit(f"AIME margins differ from {check} (max |diff| {diff})")
        sources[str(check)] = _sha(check)
    for key, e in prompts.items():
        if len(e["prompt_ids"]) != int(key.rsplit("_", 1)[-1].lstrip("g")):
            raise SystemExit(f"{key}: {len(e['prompt_ids'])} prompt ids")
    pack = {
        "format": PACK_FORMAT,
        "model": "zai-org/GLM-5.3-Flash",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what": "token ids + fp32 top1-top2 logit margins for the serving "
        "self-test (score.py); margin[t] = fp32 margin of the prediction of "
        "golden[t] from prompt + golden[:t]",
        "goldens_manifests": {
            k: str(ws / k / "manifest.json")
            for k in ("goldens_gpu_fp32_aime24", "goldens_gpu_fp32")
        },
        "sources_sha256": sources,
        "prompts": prompts,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=0) + "\n")
    print(
        f"wrote {out} ({out.stat().st_size} bytes): "
        + ", ".join(
            f"{k} {len(e['prompt_ids'])} prompt / {len(e['golden_ids'])} golden"
            for k, e in prompts.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
