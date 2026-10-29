# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Mixed prefill+decode steps through a REAL in-process vLLM engine.

The real V1 EngineCore, TTScheduler, TTWorker and TTModelRunner run over the
GLM-5.3-Flash adapter and the CPU MockModel generator of the tt-metal tree
(``tests/vllm/engine_harness.py``, the same harness as
``tests/lmcache/test_real_engine_join.py``); only the mesh open/close are
faked. The mock device poisons inactive decode rows, so a prefilling request
whose slot became an active row of the decode (or a decode row reading the
wrong slot) changes the tokens.

One request decodes while a long prompt arrives and prefills in chunks. With
mixed steps on, the decoding request must get a token on every engine step of
that prefill, and both requests' tokens must be bitwise those of the same run
with mixed steps off (and of the mock's unpaged greedy oracle).

Needs ``PYTHONPATH=<plugin>/src:<tt-metal tree>``; skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# The CPU mock stack serves 2,051 positions per request: pin the dense-exact
# contract before the adapter is imported (see test_real_engine_join.py).
_spec = importlib.util.find_spec("models.autoports.zai_org_glm_5_3_flash")
if _spec is not None and _spec.submodule_search_locations:
    _dense = (
        Path(list(_spec.submodule_search_locations)[0])
        / "tests/vllm/contracts/context_contract_dense_exact_2051.json"
    )
    if _dense.exists():
        os.environ.setdefault("GLM53_CONTEXT_CONTRACT", str(_dense))

H = pytest.importorskip(
    "models.autoports.zai_org_glm_5_3_flash.tests.vllm.engine_harness"
)
stack = pytest.importorskip(
    "models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack"
)

H.engine_env()

from vllm import SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402

from vllm_tt_plugin.config import MIXED_PREFILL_CAP_ENV, MIXED_STEPS_ENV  # noqa: E402
from vllm_tt_plugin.model_runner import TTModelRunner  # noqa: E402
from vllm_tt_plugin.scheduler import TTScheduler, is_tt_mixed_step  # noqa: E402

if not Path(H.SNAPSHOT).exists():
    pytest.skip("GLM-5.3-Flash snapshot config not available", allow_module_level=True)

BUDGET = 256  # max_num_batched_tokens: the long prompt needs five chunks
ALIGN = 128  # the default tt.prefill_chunk_alignment
DECODE_PROMPT = 100
LONG_PROMPT = 1200
DECODE_TOKENS = 24
LONG_TOKENS = 4



@pytest.fixture(autouse=True)
def _tail_split_on(monkeypatch):
    """these hits resume at the KDA-checkpoint tail split of a short prompt: GLM53_PF_TAIL_SPLIT_MIN off (under it a
    one-chunk prompt keeps no checkpoint by design; tests/lmcache/test_kda_ckpt_store_async.py covers its rule)"""
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT_MIN", "0")

@pytest.fixture(autouse=True)
def _mixed_knobs_from_the_test_only(monkeypatch):
    """The launch environment must not decide which mode a run exercises."""
    monkeypatch.delenv(MIXED_STEPS_ENV, raising=False)
    monkeypatch.delenv(MIXED_PREFILL_CAP_ENV, raising=False)


@pytest.fixture
def step_spies(monkeypatch):
    """Count the runner's mixed steps and record what each mixed schedule gave."""
    seen = {"mixed_runner_steps": 0, "mixed_schedules": []}
    run_mixed = TTModelRunner._execute_mixed_step
    schedule_mixed = TTScheduler._schedule_mixed

    def counting_execute(self, *args, **kwargs):
        seen["mixed_runner_steps"] += 1
        return run_mixed(self, *args, **kwargs)

    def recording_schedule(self, prefill_budget):
        out = schedule_mixed(self, prefill_budget)
        # The engine suffixes request ids ("d" -> "d-<hex>").
        scheduled = {
            req_id.split("-")[0]: n for req_id, n in out.num_scheduled_tokens.items()
        }
        seen["mixed_schedules"].append((scheduled, is_tt_mixed_step(out)))
        return out

    monkeypatch.setattr(TTModelRunner, "_execute_mixed_step", counting_execute)
    monkeypatch.setattr(TTScheduler, "_schedule_mixed", recording_schedule)
    return seen


def _run(*, mixed: bool, async_scheduling: bool):
    """Decode request ``d`` alone until it has three tokens, then add the long
    prompt ``p``; step the engine until both finish.

    Returns the per-step new-token counts, the step ``p`` arrived at, both
    requests' final outputs and the mock oracle's greedy tokens.
    """
    decode_prompt = stack.prompt_ids(7, DECODE_PROMPT)
    long_prompt = stack.prompt_ids(8, LONG_PROMPT)
    with H.fake_tt_mesh():
        llm = H.make_llm(
            max_num_seqs=4,
            max_num_batched_tokens=BUDGET,
            async_scheduling=async_scheduling,
            tt_config={"mixed_prefill_decode_steps": mixed},
        )
        try:
            eng = llm.llm_engine
            gen = H.live_generator()
            eng.add_request(
                "d",
                TokensPrompt(prompt_token_ids=decode_prompt),
                SamplingParams(
                    max_tokens=DECODE_TOKENS,
                    temperature=0.0,
                    ignore_eos=True,
                ),
            )
            counts = {"d": 0, "p": 0}
            per_step: list[dict[str, int]] = []
            finished = {}
            p_added_at = None
            step = 0
            while eng.has_unfinished_requests() or p_added_at is None:
                if p_added_at is None and counts["d"] >= 3:
                    eng.add_request(
                        "p",
                        TokensPrompt(prompt_token_ids=long_prompt),
                        SamplingParams(
                            max_tokens=LONG_TOKENS, temperature=0.0, ignore_eos=True
                        ),
                    )
                    p_added_at = step
                new = {}
                for out in eng.step():
                    n = len(out.outputs[0].token_ids)
                    new[out.request_id] = n - counts[out.request_id]
                    counts[out.request_id] = n
                    if out.finished:
                        finished[out.request_id] = out.outputs[0]
                per_step.append(new)
                step += 1
                assert step < 200, "workload did not finish"
            oracle = {
                "d": stack.reference_greedy(gen.backend, decode_prompt, DECODE_TOKENS),
                "p": stack.reference_greedy(gen.backend, long_prompt, LONG_TOKENS),
            }
        finally:
            del llm
    return per_step, p_added_at, finished, oracle


def _prefill_window(per_step, p_added_at):
    """Steps from the long prompt's arrival to its first token, inclusive."""
    first = next(
        i for i in range(p_added_at, len(per_step)) if per_step[i].get("p", 0) > 0
    )
    return range(p_added_at, first + 1)


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_decode_gets_a_token_every_step_while_a_long_prompt_prefills(
    async_scheduling, step_spies
):
    off_steps, off_added, off_out, oracle = _run(
        mixed=False, async_scheduling=async_scheduling
    )
    assert step_spies["mixed_runner_steps"] == 0
    assert step_spies["mixed_schedules"] == []
    on_steps, on_added, on_out, _ = _run(mixed=True, async_scheduling=async_scheduling)

    # Bitwise: the decoding request and the long prompt (its first token too)
    # are unchanged by mixing, and both match the unpaged oracle.
    for rid in ("d", "p"):
        assert list(on_out[rid].token_ids) == list(off_out[rid].token_ids), rid
        assert list(on_out[rid].token_ids) == oracle[rid], rid
    assert on_out["p"].token_ids[0] == off_out["p"].token_ids[0]

    # Mixed: one decode token on EVERY step of the long prompt's prefill.
    window = _prefill_window(on_steps, on_added)
    decode_tokens = [on_steps[i].get("d", 0) for i in window]
    assert decode_tokens == [1] * len(window), decode_tokens
    # The prompt really was prefilled over several mixed steps.
    mixed = [sched for sched, flagged in step_spies["mixed_schedules"] if flagged]
    p_chunks = [sched["p"] for sched in mixed if "p" in sched]
    assert len(p_chunks) >= LONG_PROMPT // BUDGET, step_spies
    assert sum(p_chunks) == LONG_PROMPT
    assert step_spies["mixed_runner_steps"] == len(mixed)
    # Every chunk but the last is a full, aligned prefill budget: the decode
    # rows ride on top of the prefill's share rather than shrinking it.
    assert all(n == BUDGET and n % ALIGN == 0 for n in p_chunks[:-1]), p_chunks
    assert all(sched["d"] == 1 for sched in mixed)

    # Control: without mixing the decode stalls on some step of the prefill.
    off_window = _prefill_window(off_steps, off_added)
    assert any(off_steps[i].get("d", 0) == 0 for i in off_window)


def _run_sampling_features(*, mixed: bool, async_scheduling: bool):
    """Host-sampled rows in mixed steps: ``d`` decodes under a regex grammar,
    ``e`` asks for logprobs, and the long prompt ``p`` (a choice grammar) gets
    its first token in a mixed step. Returns each request's final output."""
    with H.fake_tt_mesh():
        llm = H.make_llm(
            max_num_seqs=4,
            max_num_batched_tokens=BUDGET,
            async_scheduling=async_scheduling,
            tt_config={"mixed_prefill_decode_steps": mixed},
        )
        try:
            eng = llm.llm_engine
            eng.add_request(
                "d",
                TokensPrompt(prompt_token_ids=stack.prompt_ids(7, DECODE_PROMPT)),
                SamplingParams(
                    max_tokens=30,
                    temperature=0.0,
                    structured_outputs=StructuredOutputsParams(
                        regex="[0-9]{3}-[a-z]{4}-[0-9]{6}"
                    ),
                ),
            )
            eng.add_request(
                "e",
                TokensPrompt(prompt_token_ids=stack.prompt_ids(9, 90)),
                SamplingParams(
                    max_tokens=DECODE_TOKENS,
                    temperature=0.0,
                    ignore_eos=True,
                    logprobs=2,
                ),
            )
            outs = {}
            for step in range(200):
                if step == 4:
                    eng.add_request(
                        "p",
                        TokensPrompt(prompt_token_ids=stack.prompt_ids(8, LONG_PROMPT)),
                        SamplingParams(
                            max_tokens=8,
                            temperature=0.0,
                            structured_outputs=StructuredOutputsParams(
                                choice=["alpha", "beta", "gamma delta"]
                            ),
                        ),
                    )
                for out in eng.step():
                    if out.finished:
                        outs[out.request_id] = out.outputs[0]
                if step >= 4 and not eng.has_unfinished_requests():
                    break
            assert set(outs) == {"d", "e", "p"}
        finally:
            del llm
    return outs


def test_mixed_steps_keep_grammars_and_logprobs(step_spies):
    """Grammar rows of both parts (a decode's regex, a prompt's first token
    under a choice) and a decode's logprobs come out of mixed steps exactly as
    out of single-kind steps, under the sync and the async scheduler. Under
    the async one the regex request finishes through an output the scheduler
    has not processed when the next (mixed) step is scheduled: its row is
    absent from that step's grammar output and must not fail the step."""
    reference = _run_sampling_features(mixed=False, async_scheduling=False)
    assert step_spies["mixed_schedules"] == []
    for async_scheduling in (False, True):
        step_spies["mixed_schedules"].clear()
        outs = _run_sampling_features(mixed=True, async_scheduling=async_scheduling)
        assert any(flagged for _, flagged in step_spies["mixed_schedules"])
        for rid in ("d", "e", "p"):
            assert list(outs[rid].token_ids) == list(reference[rid].token_ids), rid
            assert outs[rid].text == reference[rid].text, rid
        assert outs["p"].text in ("alpha", "beta", "gamma delta")
        on_lp, off_lp = outs["e"].logprobs, reference["e"].logprobs
        assert on_lp is not None and len(on_lp) == len(off_lp) == DECODE_TOKENS
        for on_pos, off_pos in zip(on_lp, off_lp):
            assert {t: (lp.logprob, lp.rank) for t, lp in on_pos.items()} == {
                t: (lp.logprob, lp.rank) for t, lp in off_pos.items()
            }


@pytest.fixture
def mock_latent_pages(monkeypatch):
    """The 3-layer mock has one sparse_dsa layer standing in for all 11 of the
    spec (as in test_real_engine_join.py)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import (
        layer_kinds,
        text_config_of,
    )
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(H.SNAPSHOT)
    mla = [
        i for i, k in enumerate(layer_kinds(text_config_of(hf_config))) if k == "mla"
    ]

    def latent_pages(self, kv_cache):
        lat = next(
            c.latent for lay, c in zip(self.layers, kv_cache.caches) if not lay.is_kda
        )
        return {i: lat for i in mla}

    monkeypatch.setattr(stack.RecordingMockModel, "latent_pages", latent_pages)


def test_connector_store_and_hit_in_mixed_steps(
    tmp_path, mock_latent_pages, step_spies, monkeypatch
):
    """A prompt stored (pages + KDA chunk checkpoint) from the prefill part of
    a mixed step, then hit (pages + KDA import) by another prompt's prefill
    part, while a request decodes throughout: the connector sees each step
    once and every request keeps the oracle's tokens.

    ``a`` takes the row and the state slot of a request that just finished,
    below the decoding request's slot, so the mixed step's decode gather moves
    ``a``'s freshly prefilled state to another slot before its checkpoint is
    taken (deferred to the next step): the checkpoint must follow the state.

    The stored prompt prefills its checkpointed part in one step:
    ExampleConnector keys a store by the whole prompt at its first chunk, so a
    chunked store would hold only that chunk's pages under the prompt's key.
    """
    from vllm.config import KVTransferConfig

    from vllm_tt_plugin.kv_connector import ensure_tt_kv_transfer_shutdown

    remaps = []
    slot_remap = TTModelRunner._decode_state_slot_remap

    def recording_remap(self, row_req_ids):
        remap = slot_remap(self, row_req_ids)
        remaps.append(remap is not None)
        return remap

    monkeypatch.setattr(TTModelRunner, "_decode_state_slot_remap", recording_remap)
    ensure_tt_kv_transfer_shutdown(stop_lmcache=False)
    kv = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": str(tmp_path / "pages"),
            "tt_kda_checkpoint_dir": str(tmp_path / "kda"),
        },
    )
    prompt = stack.prompt_ids(51, 400)
    prompts = {
        "d1": stack.prompt_ids(7, DECODE_PROMPT),
        "d2": stack.prompt_ids(9, 90),
        "a": prompt,
        "b": prompt,
    }
    max_tokens = {"d1": 3, "d2": 40, "a": 4, "b": 4}
    greedy = dict(temperature=0.0, ignore_eos=True)
    with H.fake_tt_mesh():
        llm = H.make_llm(
            max_num_seqs=4,
            max_num_batched_tokens=512,
            async_scheduling=False,
            kv_transfer_config=kv,
            tt_config={"mixed_prefill_decode_steps": True},
        )
        try:
            eng = llm.llm_engine
            core = eng.engine_core.engine_core
            model = core.model_executor.driver_worker.worker.model_runner.model
            gen = H.live_generator()
            outs = {}

            def add(rid):
                eng.add_request(
                    rid,
                    TokensPrompt(prompt_token_ids=prompts[rid]),
                    SamplingParams(max_tokens=max_tokens[rid], **greedy),
                )

            def step_until(done):
                for _ in range(100):
                    if done():
                        return
                    for out in eng.step():
                        if out.finished:
                            outs[out.request_id] = out
                raise AssertionError("workload did not finish")

            add("d1")
            add("d2")
            step_until(lambda: "d1" in outs)
            add("a")  # into d1's row and state slot, below d2's
            step_until(lambda: "a" in outs)
            exports = model.counters["kda_exports"]
            assert model.counters["kda_imports"] == 0
            add("b")
            step_until(lambda: not eng.has_unfinished_requests())
            for rid in prompts:
                assert list(outs[rid].outputs[0].token_ids) == stack.reference_greedy(
                    gen.backend, prompts[rid], max_tokens[rid]
                ), rid
            # a's first chunk stopped on its last 128-token boundary and was
            # checkpointed there; b resumed from that checkpoint, and both
            # prefill parts ran beside d2's decode.
            assert exports >= 1
            assert outs["b"].num_cached_tokens == 384
            assert model.counters["kda_imports"] == 1
            mixed = [
                sched for sched, flagged in step_spies["mixed_schedules"] if flagged
            ]
            assert {"d2": 1, "a": 384} in mixed
            assert {"d2": 1, "b": 400 - 384} in mixed
            assert any(remaps), "no decode gather moved a slot"
        finally:
            del llm
            ensure_tt_kv_transfer_shutdown(stop_lmcache=False)
