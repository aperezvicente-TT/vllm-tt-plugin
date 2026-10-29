# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Mixed prefill+decode steps in the scheduler (host-only).

With ``tt.mixed_prefill_decode_steps`` (or ``GLM53_MIXED_STEPS=1``) a step that
has prefill work pending while requests decode carries both: every running
decode gets its token and the prefill gets a capped, 128-aligned share of the
budget. These tests drive the real ``TTScheduler`` (real vLLM base scheduler,
KV cache manager and requests) and feed back the outputs a TT step would
produce; the runner side is covered end to end by
``test_mixed_steps_engine.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import (
    MIXED_PREFILL_CAP_ENV,
    MIXED_STEPS_ENV,
    get_tt_mixed_prefill_cap,
    get_tt_mixed_steps_enabled,
    store_tt_output_tokens_per_step,
)
from vllm_tt_plugin.model_runner import _merge_part_logprobs
from vllm_tt_plugin.scheduler import (
    TTDecodeInterleavePolicy,
    TTScheduler,
    is_tt_mixed_step,
)

BLOCK = 64
ALIGN = 128
BUDGET = 512
MAX_MODEL_LEN = 4096
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


class _ChunkedStubModel:
    """Declares chunked prefill, so the platform keeps the scheduler's budget."""

    model_capabilities = {"supports_chunked_prefill": True}


@contextmanager
def _stub_model_resolution():
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(_ChunkedStubModel, None),
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _no_launch_knobs(monkeypatch):
    monkeypatch.delenv(MIXED_STEPS_ENV, raising=False)
    monkeypatch.delenv(MIXED_PREFILL_CAP_ENV, raising=False)


def _scheduler(
    *, mixed: bool | None = True, async_scheduling: bool = False, **tt_keys
) -> TTScheduler:
    if mixed is not None:
        tt_keys["mixed_prefill_decode_steps"] = mixed
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=BUDGET,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={"tt": tt_keys},
        )
    assert config.scheduler_config.enable_chunked_prefill
    config.scheduler_config.async_scheduling = async_scheduling
    store_tt_output_tokens_per_step(config, 1)
    num_blocks = 4 * MAX_MODEL_LEN // BLOCK
    cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(request_id: str, prompt_len: int, max_tokens: int = 64, **params):
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, **params)
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=[3 + i % 50 for i in range(prompt_len)],
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _commit(scheduler: TTScheduler, out, token: int = 9) -> None:
    """Feed back what the TT runner returns for ``out``: one token for every
    row whose prompt is complete, none for an intermediate chunk."""
    req_ids = list(out.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        done = request.num_computed_tokens >= request.num_tokens
        sampled.append([token] if done else [])
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _decoding(
    scheduler: TTScheduler, req_id: str = "d", prompt_len: int = 100, **params
):
    """Admit ``req_id`` and run it to its first decode-ready state (requests
    already decoding may advance beside it)."""
    scheduler.add_request(_request(req_id, prompt_len, **params))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[req_id] == prompt_len
    assert all(
        n == 1 for other, n in out.num_scheduled_tokens.items() if other != req_id
    )
    _commit(scheduler, out)
    assert not scheduler.requests[req_id].is_prefill_chunk


# --- Scheduling ---------------------------------------------------------------


def test_mixed_step_schedules_every_decode_beside_the_prefill():
    scheduler = _scheduler()
    _decoding(scheduler, "d1")
    _decoding(scheduler, "d2")  # itself a mixed step: d1 decodes beside it
    assert scheduler.requests["d1"].num_output_tokens == 2
    scheduler.add_request(_request("p", 1500))

    out = scheduler.schedule()

    # The decodes ride on top of a full prefill share of the budget.
    assert out.num_scheduled_tokens == {"d1": 1, "d2": 1, "p": BUDGET}
    assert out.total_num_scheduled_tokens == BUDGET + 2
    assert is_tt_mixed_step(out)
    _commit(scheduler, out)
    # The continuation is mixed too, and the decodes advanced one token.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"d1": 1, "d2": 1, "p": BUDGET}
    assert is_tt_mixed_step(out)
    assert scheduler.requests["d1"].num_output_tokens == 3


def test_long_prompt_prefills_in_mixed_steps_until_its_last_chunk():
    scheduler = _scheduler()
    _decoding(scheduler)
    scheduler.add_request(_request("p", 1300))
    chunks, decode_tokens = [], 0
    while scheduler.requests["p"].is_prefill_chunk or not chunks:
        out = scheduler.schedule()
        assert is_tt_mixed_step(out)
        chunks.append(out.num_scheduled_tokens["p"])
        decode_tokens += out.num_scheduled_tokens["d"]
        _commit(scheduler, out)
    assert chunks == [512, 512, 276]
    assert decode_tokens == len(chunks)
    # The prompt is done: the next step is an ordinary decode step.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"d": 1, "p": 1}
    assert not is_tt_mixed_step(out)


@pytest.mark.parametrize("cap, chunk", [(300, 256), (128, 128), (640, BUDGET)])
def test_prefill_part_is_capped_and_ends_on_the_alignment_grid(monkeypatch, cap, chunk):
    monkeypatch.setenv(MIXED_PREFILL_CAP_ENV, str(cap))
    scheduler = _scheduler()
    _decoding(scheduler)
    scheduler.add_request(_request("p", 2000))

    out = scheduler.schedule()

    assert out.num_scheduled_tokens == {"d": 1, "p": chunk}
    assert chunk % ALIGN == 0
    assert is_tt_mixed_step(out)


def test_the_cap_is_shared_by_every_prefill_row_of_the_step(monkeypatch):
    monkeypatch.setenv(MIXED_PREFILL_CAP_ENV, "300")
    scheduler = _scheduler()
    _decoding(scheduler)
    scheduler.add_request(_request("p1", 1000))
    scheduler.add_request(_request("p2", 40))
    scheduler.add_request(_request("p3", 200))

    out = scheduler.schedule()

    # p1 takes the aligned 256; p2's whole (final, unaligned) prompt fits the
    # remaining 44; p3 cannot reach the grid in the 4 left and waits.
    assert out.num_scheduled_tokens == {"d": 1, "p1": 256, "p2": 40}
    assert sum(n for r, n in out.num_scheduled_tokens.items() if r != "d") <= 300


def test_a_skipped_decode_does_not_grow_the_prefill_past_the_cap(monkeypatch):
    """Async: a decode whose placeholder already reaches max_tokens is skipped
    by upstream; its token must not go to the prefill (a 301-token final chunk
    would cross the 300 cap)."""
    monkeypatch.setenv(MIXED_PREFILL_CAP_ENV, "300")
    scheduler = _scheduler(async_scheduling=True)
    scheduler.add_request(_request("d", 100, max_tokens=2))
    _commit(scheduler, scheduler.schedule())
    in_flight = scheduler.schedule()  # d's last token, not committed yet
    assert in_flight.num_scheduled_tokens == {"d": 1}
    scheduler.add_request(_request("p", 301))

    out = scheduler.schedule()

    assert out.num_scheduled_tokens == {"p": 256}
    # Only the prefill was scheduled: not a mixed step for the runner.
    assert not is_tt_mixed_step(out)


# --- Mixed steps off / not eligible ---------------------------------------------


def test_default_keeps_prefill_and_decode_in_separate_steps():
    scheduler = _scheduler(mixed=None)
    assert scheduler._mixed_steps is False
    _decoding(scheduler)
    _commit(scheduler, scheduler.schedule())  # an ordinary decode step
    scheduler.add_request(_request("p", 1500))

    phases = []
    for _ in range(3):
        out = scheduler.schedule()
        assert not is_tt_mixed_step(out)
        phases.append("D" if set(out.num_scheduled_tokens) == {"d"} else "P")
        if phases[-1] == "P":
            assert set(out.num_scheduled_tokens) == {"p"}
        _commit(scheduler, out)
    # The decode-interleave cadence (2 prefill steps, then 1 decode step).
    assert phases == ["P", "P", "D"]


def test_environment_overrides_the_config_key(monkeypatch):
    on = SimpleNamespace(additional_config={"tt": {"mixed_prefill_decode_steps": True}})
    off = SimpleNamespace(additional_config={"tt": {}})
    assert get_tt_mixed_steps_enabled(on) is True
    assert get_tt_mixed_steps_enabled(off) is False
    monkeypatch.setenv(MIXED_STEPS_ENV, "0")
    assert get_tt_mixed_steps_enabled(on) is False
    monkeypatch.setenv(MIXED_STEPS_ENV, "1")
    assert get_tt_mixed_steps_enabled(off) is True
    monkeypatch.setenv(MIXED_STEPS_ENV, "yes")
    with pytest.raises(ValueError, match=MIXED_STEPS_ENV):
        get_tt_mixed_steps_enabled(off)
    bad = SimpleNamespace(additional_config={"tt": {"mixed_prefill_decode_steps": 1}})
    monkeypatch.delenv(MIXED_STEPS_ENV)
    with pytest.raises(ValueError, match="mixed_prefill_decode_steps"):
        get_tt_mixed_steps_enabled(bad)


def test_env_enables_mixed_steps_on_a_launch_without_the_key(monkeypatch):
    monkeypatch.setenv(MIXED_STEPS_ENV, "1")
    scheduler = _scheduler(mixed=None)
    _decoding(scheduler)
    scheduler.add_request(_request("p", 1000))
    assert is_tt_mixed_step(scheduler.schedule())


def test_cap_knob_validation(monkeypatch):
    assert get_tt_mixed_prefill_cap() is None
    monkeypatch.setenv(MIXED_PREFILL_CAP_ENV, "2048")
    assert get_tt_mixed_prefill_cap() == 2048
    for bad in ("0", "-5", "lots"):
        monkeypatch.setenv(MIXED_PREFILL_CAP_ENV, bad)
        with pytest.raises(ValueError, match=MIXED_PREFILL_CAP_ENV):
            get_tt_mixed_prefill_cap()


def test_a_cap_below_the_alignment_is_refused_at_start_up(monkeypatch):
    monkeypatch.setenv(MIXED_PREFILL_CAP_ENV, "64")
    with pytest.raises(ValueError, match="below the prefill chunk alignment"):
        _scheduler()
    # The knob is irrelevant (and not validated) with mixed steps off.
    assert _scheduler(mixed=False)._mixed_steps is False


def test_running_speculative_drafts_keep_the_step_single_kind():
    scheduler = _scheduler()
    _decoding(scheduler)
    scheduler.add_request(_request("p", 1000))
    scheduler.requests["d"].spec_token_ids = [11, 12]

    out = scheduler.schedule()

    assert not is_tt_mixed_step(out)
    assert set(out.num_scheduled_tokens) == {"p"}


def _bare_scheduler(**model_config) -> TTScheduler:
    scheduler = TTScheduler.__new__(TTScheduler)
    scheduler.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    scheduler._is_block_output_model = False
    scheduler.num_spec_tokens = 0
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=None,
        additional_config={},
        parallel_config=SimpleNamespace(data_parallel_size=1),
        model_config=SimpleNamespace(**model_config),
    )
    return scheduler


def test_configurations_whose_runner_paths_cannot_split_a_step_stay_off():
    assert _bare_scheduler()._mixed_steps_unsupported_reason() is None

    spec = _bare_scheduler()
    spec.vllm_config.speculative_config = object()
    spec.num_spec_tokens = 3
    assert "speculative" in spec._mixed_steps_unsupported_reason()

    block = _bare_scheduler()
    block._is_block_output_model = True
    assert "block-output" in block._mixed_steps_unsupported_reason()

    unchunked = _bare_scheduler()
    unchunked.scheduler_config.enable_chunked_prefill = False
    assert "chunked prefill" in unchunked._mixed_steps_unsupported_reason()

    assert "mrope" in _bare_scheduler(uses_mrope=True)._mixed_steps_unsupported_reason()
    assert (
        "logits processors"
        in _bare_scheduler(
            logits_processors=["my.Proc"]
        )._mixed_steps_unsupported_reason()
    )


def test_a_live_batch_indexed_sampling_request_keeps_steps_single_kind():
    scheduler = _scheduler()
    _decoding(scheduler, "d", min_p=0.1)
    _decoding(scheduler, "d2")
    scheduler.add_request(_request("p", 2000))

    for _ in range(3):
        out = scheduler.schedule()
        assert not is_tt_mixed_step(out)
        assert set(out.num_scheduled_tokens) in ({"p"}, {"d", "d2"})
        _commit(scheduler, out)
    assert scheduler.requests["p"].is_prefill_chunk
    # Once that request is gone, the step mixes again.
    scheduler.finish_requests(["d"], RequestStatus.FINISHED_ABORTED)
    out = scheduler.schedule()
    assert is_tt_mixed_step(out)
    assert set(out.num_scheduled_tokens) == {"d2", "p"}


def test_a_mixed_step_resets_the_interleave_counters():
    policy = TTDecodeInterleavePolicy(
        SimpleNamespace(
            additional_config={"tt": {"decode_interleave_prefill_steps": 2}}
        )
    )
    for _ in range(2):
        policy.record_step(is_decode=False, prefill_pending=True)
    assert policy.wants_decode_step(has_pending_prefill=True, has_running_decode=True)
    policy.note_mixed_step()
    assert not policy.wants_decode_step(
        has_pending_prefill=True, has_running_decode=True
    )


# --- Output assembly -------------------------------------------------------------


def _lp(token_ids, width):
    rows = len(token_ids)
    ids = torch.tensor([[t] + [100 + c for c in range(width - 1)] for t in token_ids])
    return LogprobsTensors(
        ids.to(torch.int32),
        torch.arange(rows * width, dtype=torch.float32).reshape(rows, width),
        torch.arange(rows, dtype=torch.int32),
    )


def test_part_logprobs_merge_into_batch_rows():
    req_ids = ("d0", "p0", "d1")
    index = {r: i for i, r in enumerate(req_ids)}
    merged = _merge_part_logprobs(
        req_ids, index, [(["d0", "d1"], _lp([5, 6], 3)), (["p0"], _lp([7], 2))]
    )
    assert merged.logprob_token_ids.shape == (3, 3)
    assert merged.logprob_token_ids[:, 0].tolist() == [5, 7, 6]
    # The narrower part is zero-padded past its own width.
    assert merged.logprob_token_ids[1].tolist() == [7, 100, 0]
    assert np.array_equal(merged.logprobs[2], np.array([3.0, 4.0, 5.0], np.float32))
    assert merged.sampled_token_ranks.tolist() == [0, 0, 1]
    assert _merge_part_logprobs(req_ids, index, []) is None


# --- GLM53_FIRST_TOKEN_YIELD (cf3h-d lever A's stopgap) ----------------------------


def test_first_token_yield_puts_an_empty_step_after_a_finishing_prefill(monkeypatch):
    """The step after the one that finishes p's prompt schedules nothing, so vLLM's batch queue pops p's prefill
    output (token 0) before the first decode is built; the decode follows on the next call. A non-finishing chunk
    and a decode step are not followed by an empty step."""
    monkeypatch.setenv("GLM53_FIRST_TOKEN_YIELD", "1")
    scheduler = _scheduler(mixed=None, async_scheduling=True)
    scheduler.add_request(_request("p", 1000))
    steps = []
    for _ in range(5):
        out = scheduler.schedule()
        steps.append(dict(out.num_scheduled_tokens))
        _commit(scheduler, out)
    assert steps == [{"p": 512}, {"p": 488}, {}, {"p": 1}, {"p": 1}]
    assert scheduler.first_token_yields == 1


def test_first_token_yield_carries_finished_ids_and_is_off_by_default(monkeypatch):
    monkeypatch.delenv("GLM53_FIRST_TOKEN_YIELD", raising=False)
    scheduler = _scheduler(mixed=None, async_scheduling=True)
    scheduler.add_request(_request("p", 100))
    out = scheduler.schedule()
    _commit(scheduler, out)
    assert scheduler.schedule().num_scheduled_tokens == {"p": 1}  # off: the decode right away
    assert scheduler.first_token_yields == 0

    monkeypatch.setenv("GLM53_FIRST_TOKEN_YIELD", "1")
    scheduler = _scheduler(mixed=None, async_scheduling=True)
    scheduler.add_request(_request("a", 100, max_tokens=1))
    scheduler.add_request(_request("b", 300))
    out = scheduler.schedule()  # both prompts finish in this step; a's single token finishes a
    assert out.num_scheduled_tokens == {"a": 100, "b": 300}
    _commit(scheduler, out)
    empty = scheduler.schedule()
    assert empty.total_num_scheduled_tokens == 0 and "a" in empty.finished_req_ids
    assert scheduler.schedule().num_scheduled_tokens == {"b": 1}

