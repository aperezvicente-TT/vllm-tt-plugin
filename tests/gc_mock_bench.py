# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""v3b5 GC mock (program 2026-10-29T00:12:19Z (2)): the SAME 200-step serving-shaped workload twice -- baseline
(GLM53_GC_THRESHOLD=off, GLM53_GC_REFREEZE=0: CPython 700,10,10, vLLM's one startup freeze) and v3b5 (the defaults) --
through vllm_tt_plugin.gc_tuning's real policy and accounting; prints gc_pause_s_total before / after.

The heap is shaped like the served engine's: a startup heap frozen once (vLLM's freeze_gc_heap), then a long-lived heap
built AFTER that freeze during the first steps (the first-use builds, connector / LMCache state: --late-objects
containers), per-step churn (--churn containers a step, 1 % kept in a 1,024-step history), a device wait per step
(--step-ms) and one idle gap (--idle-s) after the warm-up steps, as between requests.
usage: python tests/gc_mock_bench.py [--late-objects 4000000] [--churn 20000] [--steps 200] [--step-ms 20]
       (one mode per process: --mode baseline|v3b5; without --mode it runs both and prints the comparison)"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import subprocess
import sys
import time


def run(a) -> dict:
    if a.mode == "baseline":
        os.environ.update(GLM53_GC_THRESHOLD="off", GLM53_GC_REFREEZE="0")
    else:
        os.environ.update(GLM53_GC_REFREEZE_STEPS=str(a.warm_steps), GLM53_GC_IDLE_S=str(min(1.0, a.idle_s / 2)),
                          GLM53_GC_REFREEZE_MIN_S="0")  # fmt: skip
    from vllm_tt_plugin import gc_tuning

    startup = [[i] for i in range(a.startup_objects)]  # the model / engine objects vLLM freezes at init
    g = gc_tuning.install()  # (the worker's init_device: before vLLM's freeze)
    gc.collect()
    gc.freeze()  # vLLM's freeze_gc_heap at the end of EngineCore.__init__
    base = g.report()
    late: list = []
    history: collections.deque = collections.deque(maxlen=1024)
    t0 = time.perf_counter()
    for step in range(a.steps):
        g.step_begin()
        if step < a.warm_steps:  # the first-use builds and connector state: long-lived, created after the freeze
            late.extend([k, step] for k in range(a.late_objects // a.warm_steps))
        churn = [{"row": k, "ids": [k, k + 1]} for k in range(a.churn)]  # per-step request / output objects
        history.append(churn[:: 100])  # 1 % kept a while (the step history)
        time.sleep(a.step_ms / 1e3)  # the device
        g.step_end()
        if step == a.warm_steps:  # one idle gap between requests, after the warm-up
            time.sleep(a.idle_s)
    wall = time.perf_counter() - t0
    r = g.report()
    r = {k: (round(v - base.get(k, 0), 4) if isinstance(v, float) else v - base.get(k, 0)) if k != "gc_pause_s_max"
         else v for k, v in r.items()}  # fmt: skip
    r.update(mode=a.mode, wall_s=round(wall, 2), threshold=gc.get_threshold(), late_objects=len(late),
             startup_objects=len(startup))  # fmt: skip
    g.stop()
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("baseline", "v3b5"))
    ap.add_argument("--startup-objects", type=int, default=2000000)
    ap.add_argument("--late-objects", type=int, default=4000000)
    ap.add_argument("--churn", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warm-steps", type=int, default=20)
    ap.add_argument("--step-ms", type=float, default=20.0)
    ap.add_argument("--idle-s", type=float, default=3.0)
    a = ap.parse_args()
    if a.mode:
        print(json.dumps(run(a)))
        return 0
    out = {}
    for mode in ("baseline", "v3b5"):  # separate processes: one GC policy each
        cmd = [sys.executable, __file__, "--mode", mode] + [x for x in sys.argv[1:]]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out[mode] = json.loads(res.stdout.strip().splitlines()[-1])
    for mode in ("baseline", "v3b5"):
        r = out[mode]
        print(f"{mode:8s} gc_pause_s_total {r['gc_pause_s_total']:8.3f} s  gen2 {r['gc_collections_g2']:3d} "
              f"({r['gc_pause_s_g2']:.3f} s)  gen1 {r['gc_collections_g1']:4d}  gen0 {r['gc_collections_g0']:5d}  "
              f"max pause {r['gc_pause_s_max']:.3f} s  refreezes {r['gc_refreezes']} (at idle {r['gc_refreeze_pause_s']:.3f} s)  "
              f"wall {r['wall_s']} s  "
              f"(threshold {tuple(r['threshold'])})")  # fmt: skip
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
