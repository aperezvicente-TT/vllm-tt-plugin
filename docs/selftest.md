# Startup accuracy self-test (GLM-5.3-Flash)

`src/vllm_tt_plugin/selftest/` makes the served stack prove its accuracy before
the OpenAI endpoint is reachable, and refuses to serve on a miss. It exists
because a server restored from a stale DRAM image once served wrong prefill
logits for hours while every latency cell looked fine.

## What runs

After the engine is up (model loaded, warm-up and trace capture done) and
before `serve_http` binds the socket, the endpoint plugin's `init_state` sends
37 greedy requests through the engine client (the same AsyncLLM the OpenAI
handlers use: scheduler, prefill admission, host logprobs sampler):

| part | requests | what is checked |
|---|---|---|
| strict teacher forcing | 32 fresh prefills of `AIME prompt + golden[:t]`, t = 0..31, 2 tokens, top-100 logprobs | rank of `golden[t]` in the prefill-path logprobs; rank of `golden[t+1]` in the decode-path logprobs when position 0 hit |
| greedy smoke | 3 prompts (AIME 161, golden 128, golden 512), 8 tokens | the 8 greedy tokens equal the fp32 golden greedy, forks allowed only at near-ties |
| decode after prefill | 2 prompts (AIME + golden[:16] = 177, AIME + golden[:31] = 192), 8 tokens | ranks of the golden tokens over 7 decode steps while the sampled prefix is on golden |

Gate (program ruling 2026-10-22 07:19Z, margin-aware): over positions whose
fp32 top1-top2 logit margin is >= 0.5, prefill top-1 >= 0.85 and top-5 >= 0.98;
the same thresholds on the decode-path rows; every smoke prompt on golden or
forked at a near-tie; no request errors; at least 16 clear-margin rows.
Near-ties (margin < 0.5) are reported and excluded.

The goldens pack `selftest/goldens/glm53_aime24.json` (8 KB: token ids and
margins) is derived from the program's fp32 GPU goldens by
`python -m vllm_tt_plugin.selftest.build_pack`; the pack records the source
files' sha256.

## Knobs (environment)

| variable | default | meaning |
|---|---|---|
| `GLM53_SELFTEST` | `strict` | `strict`: a miss logs the table, keeps the server NOT READY (the socket never binds) and raises, so the process exits non-zero and the supervisor restarts or escalates. `warn`: log and serve anyway (`GET /selftest` -> 503). `off`: skip (a WARNING line). |
| `GLM53_SELFTEST_OUT` | unset | write the verdict JSON (rows, paths, timing) here |
| `GLM53_SELFTEST_GOLDENS` | the packaged pack | another goldens pack |
| `GLM53_SELFTEST_TF_STEPS` | 32 | teacher-forcing positions |
| `GLM53_SELFTEST_BUDGET_S` | 60 | reported as `within_budget`; not a verdict input |
| `GLM53_SELFTEST_CONCURRENCY` | 1 | in-flight requests (1 = Stage 9's sequential API TF) |
| `GLM53_SELFTEST_TOP1` / `_TOP5` / `_MARGIN` / `_MIN_ROWS` | 0.85 / 0.98 / 0.5 / 16 | the gate |
| `GLM53_SELFTEST_REQUEST_TIMEOUT_S` | 600 | per request |

The server must allow >= 5 logprobs per position (`--max-logprobs -1` as the
program serves); otherwise the self-test cannot run (an error: strict refuses).
For a model the pack does not describe the self-test skips itself with a
WARNING (the pack names `zai-org/GLM-5.3-Flash`).

## Log lines

```
[glm53-selftest] starting: mode strict, 32 TF positions + 3 greedy smoke + 2 decode-after-prefill, budget 60 s, goldens ... (sha256 ...)
[glm53-selftest]    t  len  golden   margin  prefill  sampled  decode   note
[glm53-selftest]    0  161    1654     0.08        1     xxxx       -   near-tie
...
[glm53-selftest] smoke aime24_161 (len 161): 8/8 == golden
[glm53-selftest] decode/len=177: decode ranks at positions [17, ..., 23] = [0, 0, 0, 0, 0, 0, 0]
[glm53-selftest] SELFTEST PASS top-1 0.933 / top-5 1.000 (prefill, n=30 of 32, 2 near-ties excluded) decode top-1 0.976 / top-5 1.000 (n=41 of 43) smoke 3/3; 37/37 requests in 18.3 s (budget 60 s)
```

On a miss the last line reads `SELFTEST FAIL ... -- <reasons> -- NOT READY,
refusing to serve (GLM53_SELFTEST=strict)` and the process exits non-zero after
a clean engine shutdown.

## One-shot CLI (same plan, goldens and scorer, over HTTP)

```
python -m vllm_tt_plugin.selftest --url http://127.0.0.1:8000 --out verdict.json [--save-observations obs.json]
```

Exit 0 = PASS, 1 = FAIL, 2 = could not run. `GET /selftest` on a running server
returns the startup verdict (200 pass / skipped, 503 fail).

## Loading

Two paths load the plugin: the `vllm.endpoint_plugins` entry point
`glm53_selftest` (installed plugin; vLLM loads endpoint plugins only when
`VLLM_PLUGINS` names them) and `_install_glm53_selftest_endpoint_patch`
(platform.py), installed with the general plugin, which appends the plugin from
`load_endpoint_plugins` for PYTHONPATH-first trees. The two dedupe by name.

Tests: `tests/test_selftest.py` (pure python: synthetic golden-consistent
observations, a fake OpenAI endpoint, a fake engine client, recorded mock
responses).
