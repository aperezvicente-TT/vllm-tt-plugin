# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import os
from typing import TYPE_CHECKING, Any

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_tt_plugin.spec_decode import SpecPlan

logger = init_tt_logger(__name__)


def _extract_tt_config(
    config: dict[str, Any], config_name: str
) -> tuple[dict[str, Any], bool]:
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a JSON object")
    if "tt" not in config:
        return {}, False
    tt_config = config["tt"]
    if not isinstance(tt_config, dict):
        raise ValueError(f"{config_name}['tt'] must be a JSON object")
    return tt_config, True


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT config from vLLM's generic additional config namespace."""
    additional_config, _ = _extract_tt_config(
        getattr(vllm_config, "additional_config", {}) or {}, "additional_config"
    )
    return dict(additional_config)


# Internal keys recording resolved platform state. Stored at the top level of
# additional_config -- deliberately outside the user "tt" namespace -- so they
# never collide with user config and read as platform-derived state rather than
# user input. Every value here stays JSON-encodable, because
# VllmConfig.compute_hash calls json.dumps over the whole dictionary.
# Written by store_tt_lane_count, store_tt_output_tokens_per_step and
# store_tt_spec_plan; read by get_tt_data_parallel_size,
# get_tt_output_tokens_per_step and get_tt_spec_plan.
_RESOLVED_LANE_COUNT_KEY = "_tt_resolved_lane_count"
_OUTPUT_TOKENS_PER_STEP_KEY = "_tt_output_tokens_per_step"
_SUPPORTS_DEVICE_GRAMMAR_KEY = "_tt_supports_device_grammar"
_SPEC_PLAN_KEY = "_tt_spec_plan"


def get_tt_data_parallel_size(vllm_config: "VllmConfig") -> int:
    """Effective TT lane count for batching, KV sizing, and merged execution.

    Standard multi-process DP runs one independent TT mesh per rank, so the TT
    model itself sees no internal DP and the effective TT lane count remains 1.
    With a single engine (``data_parallel_size == 1``) the value is the lane
    count resolved by the Galaxy DP-to-lanes conversion (see ``platform.py``)
    and recorded via ``store_tt_lane_count``; absent that, the count is 1.
    Not user-facing.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_RESOLVED_LANE_COUNT_KEY, 1))


def store_tt_lane_count(vllm_config: "VllmConfig", lanes: int) -> None:
    """Record the resolved in-process TT lane count on the config.

    Writes an internal, top-level key into ``additional_config`` (kept out of
    the user "tt" namespace) so ``get_tt_data_parallel_size`` observes it both
    here and in the worker subprocess -- ``additional_config`` is a declared
    config field, so it survives the copy/pickle to that process. Internal
    handoff from the Galaxy DP-to-lanes conversion; not user-facing.
    """
    if lanes < 1:
        raise ValueError(f"resolved TT lane count must be >= 1, got {lanes}")
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_RESOLVED_LANE_COUNT_KEY] = lanes


def get_tt_output_tokens_per_step(vllm_config: "VllmConfig") -> int:
    """Return the normalized model output width, defaulting to AR behavior.

    ``TTPlatform.check_and_update_config`` resolves the model capability once
    and stores it on the serializable vLLM config. Scheduler and worker
    construction therefore do not need to import model-loader code.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_OUTPUT_TOKENS_PER_STEP_KEY, 1))


def require_tt_output_tokens_per_step(vllm_config: "VllmConfig") -> int:
    """Return the resolved output width, failing if setup did not store it."""
    additional = getattr(vllm_config, "additional_config", None)
    if (
        not isinstance(additional, dict)
        or _OUTPUT_TOKENS_PER_STEP_KEY not in additional
    ):
        raise RuntimeError(
            "TT output_tokens_per_step was not initialized on VllmConfig"
        )
    return int(additional[_OUTPUT_TOKENS_PER_STEP_KEY])


def is_tt_block_output_model(vllm_config: "VllmConfig") -> bool:
    """Whether this config describes a model that commits multi-token blocks."""
    return get_tt_output_tokens_per_step(vllm_config) > 1


# Platform-derived, like _RESOLVED_LANE_COUNT_KEY above: written from the
# MODEL's declared capability, never read from operator input. The leading
# underscore keeps an operator-passed --additional-config entry of the same
# name from reading as configuration this code honours.
_ADAPTIVE_BLOCK_OUTPUT_KEY = "_tt_adaptive_block_output"


def is_tt_adaptive_block_output_model(vllm_config: "VllmConfig") -> bool:
    """Whether the model commits a block ONLY when it decodes alone (batch==1).

    A plain block-output model owns a single request state and requires
    ``max_num_seqs 1`` / no data-parallelism. An ADAPTIVE block-output model
    emits its multi-token block only on steps that schedule exactly one decode
    request, and falls back to plain 1-token baseline decode whenever two or
    more requests batch together. That lifts the ``max_num_seqs 1`` and
    data-parallel restrictions: at low concurrency each request gets the block
    speedup, at higher concurrency the server is a plain batched baseline
    (never worse). ``TTScheduler._update_after_schedule`` owns the reservation
    predicate and reserves the K-token placeholder block only when the step is
    solo, is a decode, and the request owns the model's speculative session --
    which is where ``tt_adaptive_block_max_prompt_tokens`` was already applied,
    at the prefill that armed that session.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional.get(_ADAPTIVE_BLOCK_OUTPUT_KEY, False))


def store_tt_adaptive_block_output(vllm_config: "VllmConfig", flag: bool) -> None:
    """Record the model's adaptive block-output capability on the config.

    Internal platform-to-runtime handoff, not user-facing: the value comes from
    the model's ``tt_adaptive_block_output`` capability, which
    ``TTPlatform.check_and_update_config`` has already validated against
    ``output_tokens_per_step > 1``.
    """
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_OUTPUT_KEY] = bool(flag)


# Platform-derived; see _ADAPTIVE_BLOCK_OUTPUT_KEY.
_ADAPTIVE_BLOCK_MAX_PROMPT_KEY = "_tt_adaptive_block_max_prompt_tokens"


def get_tt_adaptive_block_max_prompt_tokens(vllm_config: "VllmConfig") -> int:
    """Prompt-length frontier for the adaptive block path (0 = no limit).

    An adaptive block model whose fused capture cannot fit beyond some prompt
    length serves longer prompts as plain baseline (width-1 steps) for their
    whole lifetime. The scheduler must reserve width-1 for those requests even
    on solo decode steps, so the model declares the SAME frontier it gates on.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_ADAPTIVE_BLOCK_MAX_PROMPT_KEY, 0))


def store_tt_adaptive_block_max_prompt_tokens(
    vllm_config: "VllmConfig", limit: int
) -> None:
    """Record the adaptive block path's prompt-length frontier on the config.

    Internal platform-to-runtime handoff, not user-facing: the value comes from
    the model's ``tt_adaptive_block_max_prompt_tokens`` capability. Validated
    here, next to its sibling ``store_tt_output_tokens_per_step``, rather than
    in the platform -- the pairing rule that a non-zero frontier requires the
    adaptive capability stays in the platform, because only the platform sees
    both capabilities.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(
            "resolved TT adaptive_block_max_prompt_tokens must be an integer "
            f">= 0, got {limit!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_ADAPTIVE_BLOCK_MAX_PROMPT_KEY] = int(limit)


# Platform-derived; see _ADAPTIVE_BLOCK_OUTPUT_KEY.
_BLOCK_KV_EXTENT_KEY = "_tt_block_kv_extent_tokens"


def get_tt_block_kv_extent_tokens(vllm_config: "VllmConfig") -> int:
    """Total KV positions ONE block-output decode step may touch (0 = undeclared).

    A block-output step runs internal verify iterations against vLLM-owned KV, so
    the positions it writes run past the tokens it commits. The emitted width does
    not bound that: the physical extent is the committed block PLUS the last
    iteration's verification rows plus any accepted tokens carried beyond the
    emitted width, and a model may be configured with a verification width LARGER
    than its output block (dFlash at ``SERVE_BLOCK=2``, ``VERIFY=7`` emits 2 and
    writes 8 verification rows). Reserving a multiple of the output width then
    under-allocates and the step writes positions with no request block
    (vllm-tt-plugin#118 review, finding 2).

    So the model declares the extent it actually touches and the scheduler
    reserves at least that much lookahead. 0 means the model did not declare one,
    and the scheduler falls back to its output-width multiple.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_BLOCK_KV_EXTENT_KEY, 0))


def store_tt_block_kv_extent_tokens(vllm_config: "VllmConfig", extent: int) -> None:
    """Record the block-output step's physical KV extent on the config.

    Internal platform-to-runtime handoff: the value comes from the model's
    ``tt_block_kv_extent_tokens`` capability. Validated here alongside its
    siblings; the rule that it must cover the output width lives in the platform,
    which is the only place that sees both.
    """
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
        raise ValueError(
            "resolved TT block_kv_extent_tokens must be an integer >= 0, got "
            f"{extent!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_BLOCK_KV_EXTENT_KEY] = int(extent)


# Platform-derived; see _ADAPTIVE_BLOCK_OUTPUT_KEY.
_REQUEST_CONTEXT_LIMIT_KEY = "_tt_request_context_limit"


def get_tt_request_context_limit(vllm_config: "VllmConfig") -> int:
    """Largest prompt + output length one request may reach (0 = max_model_len).

    A model can advertise ``max_model_len`` while some of its layers compute only
    a shorter context faithfully for now (a staged bring-up directive rather than
    a device limit). It declares that shorter length as
    ``tt_request_context_limit``; the platform rejects longer requests at
    admission and resolves default ``max_tokens`` against it, so such a request
    gets an error instead of reaching the device.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    return int(additional.get(_REQUEST_CONTEXT_LIMIT_KEY, 0))


def resolve_tt_request_context_limit(
    declared: object, max_model_len: int, output_tokens_per_step: int
) -> int:
    """Validate a model's ``tt_request_context_limit`` declaration.

    Returns the limit to enforce: 0 when undeclared or when it does not lie
    below ``max_model_len`` (``max_model_len`` already bounds every request).
    """
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        raise ValueError(
            f"tt_request_context_limit must be an integer >= 0; got {declared!r}"
        )
    if declared >= max_model_len:
        return 0
    if declared and output_tokens_per_step > 1:
        raise ValueError(
            "tt_request_context_limit is not supported for block-output "
            f"models (output_tokens_per_step={output_tokens_per_step})"
        )
    return declared


def check_tt_request_context_limit(prompt_len: int, requested: int, limit: int) -> None:
    """Refuse a request of ``prompt_len`` + ``requested`` tokens over ``limit``
    (0 = no limit)."""
    if limit and prompt_len + requested > limit:
        raise ValueError(
            f"Request of {prompt_len} prompt + {requested} output tokens "
            f"exceeds this model's request context limit of {limit} tokens "
            "(model_capabilities['tt_request_context_limit']; max_model_len "
            "is the advertised context, not the limit served today)."
        )


def store_tt_request_context_limit(vllm_config: "VllmConfig", limit: int) -> None:
    """Record the resolved per-request context limit on the config.

    Internal platform-to-runtime handoff: the value comes from the model's
    ``tt_request_context_limit`` capability, already reduced to 0 when it does
    not lie below ``max_model_len``.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(
            f"resolved TT request_context_limit must be an integer >= 0, got {limit!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_REQUEST_CONTEXT_LIMIT_KEY] = int(limit)


def store_tt_output_tokens_per_step(
    vllm_config: "VllmConfig", output_tokens_per_step: int
) -> None:
    """Store the validated per-request output width on the vLLM config."""
    if (
        isinstance(output_tokens_per_step, bool)
        or not isinstance(output_tokens_per_step, int)
        or output_tokens_per_step < 1
    ):
        raise ValueError(
            "resolved TT output_tokens_per_step must be an integer >= 1, got "
            f"{output_tokens_per_step!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_OUTPUT_TOKENS_PER_STEP_KEY] = output_tokens_per_step


def get_tt_supports_device_grammar(vllm_config: "VllmConfig") -> bool:
    """Return the resolved model-side device grammar capability."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional.get(_SUPPORTS_DEVICE_GRAMMAR_KEY, False))


def is_tt_device_grammar_preload_eligible(
    vllm_config: "VllmConfig",
    *,
    trace_mode: str,
    enable_model_warmup: bool,
) -> bool:
    """Whether config-time state permits loading device grammar support.

    Grammar-on sampling traces must be compiled before traced execution starts.
    The loaded runner separately checks sampler/API and mesh-layout support.
    """
    return get_tt_supports_device_grammar(vllm_config) and (
        trace_mode == "none" or enable_model_warmup
    )


def store_tt_supports_device_grammar(
    vllm_config: "VllmConfig", supported: bool
) -> None:
    """Store device grammar support on the serializable vLLM config."""
    if not isinstance(supported, bool):
        raise ValueError(
            f"resolved TT supports_device_grammar must be a boolean, got {supported!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_SUPPORTS_DEVICE_GRAMMAR_KEY] = supported


def get_tt_spec_plan(vllm_config: "VllmConfig") -> "SpecPlan | None":
    """Return the admitted speculative plan, or ``None``.

    ``None`` covers two cases that a caller needing to tell them apart should
    resolve with ``require_tt_spec_plan``: no speculation was requested, and
    admission has not run in this process.

    ``TTModelRunner.__init__`` reads it to learn the draft length and whether
    the model serves a narrow decode, and ``TTPlatform.validate_request`` reads
    it to decide whether a request's sampling parameters can be served.
    Admission runs once per process, not once per launch:
    ``TTWorker.init_device`` re-runs ``TTPlatform.check_and_update_config``, so
    each process rebuilds its own plan from the same declarations.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    stored = additional.get(_SPEC_PLAN_KEY)
    if stored is None:
        return None
    from vllm_tt_plugin.spec_decode import SpecPlan

    # Rebuilt rather than returned, so SpecPlan.__post_init__ validates on the
    # way out and additional_config stays JSON-encodable for vLLM's config
    # hashing.
    return SpecPlan(**stored)


def require_tt_spec_plan(vllm_config: "VllmConfig") -> "SpecPlan | None":
    """Return the admitted plan, failing if admission did not run."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict) or _SPEC_PLAN_KEY not in additional:
        raise RuntimeError("TT speculative admission did not run on VllmConfig")
    return get_tt_spec_plan(vllm_config)


def store_tt_spec_plan(vllm_config: "VllmConfig", plan: "SpecPlan | None") -> None:
    """Store the admitted speculative plan, or ``None`` for no speculation.

    Stored as a plain dictionary because every other key in
    ``additional_config`` is JSON-encodable and ``VllmConfig.compute_hash``
    calls ``json.dumps`` over the whole dictionary.
    """
    from dataclasses import asdict

    from vllm_tt_plugin.spec_decode import SpecPlan

    if plan is not None and not isinstance(plan, SpecPlan):
        raise ValueError(
            f"TT speculative plan must be a SpecPlan or None, got {plan!r}"
        )
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    additional[_SPEC_PLAN_KEY] = None if plan is None else asdict(plan)


def get_tt_max_batch_size(vllm_config: "VllmConfig") -> int:
    """Return the global TT batch capacity for model/KV sizing.

    Standard DP is per-rank and single-process lane mode already stores the
    global engine capacity in ``max_num_seqs`` after the Galaxy conversion, so
    the TT model should always size itself to the visible engine-local batch.
    """
    return int(vllm_config.scheduler_config.max_num_seqs)


def get_tt_per_lane_max_num_seqs(vllm_config: "VllmConfig") -> int:
    """Return the per-lane/per-rank scheduling and wire-format capacity.

    Outside lane mode the global ``max_num_seqs`` is already the per-rank
    capacity. In single-process lane mode it is the validated per-lane split
    (see ``validate_tt_lane_config``).
    """
    if not uses_tt_lane_coordinator(vllm_config):
        return int(vllm_config.scheduler_config.max_num_seqs)
    return validate_tt_lane_config(vllm_config)


def validate_tt_lane_config(vllm_config: "VllmConfig") -> int:
    """Validate single-process lane-mode batch sizing; return per-lane capacity.

    Lane mode partitions the global ``max_num_seqs`` evenly across the lanes
    (one in-process DP replica each), so the global value must be a positive
    multiple of the lane count; raises ``ValueError`` otherwise. Assumes lane
    mode is active (callers gate on ``uses_tt_lane_coordinator``).

    Exposed as a named helper so ``platform.check_and_update_config`` can run
    this check at config time -- calling it for its raising side effect so a
    misconfiguration fails fast with a clear message -- rather than calling the
    per-lane getter and discarding its result.
    """
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    lanes = get_tt_data_parallel_size(vllm_config)
    if max_num_seqs % lanes != 0:
        raise ValueError(
            "max_num_seqs must be divisible by the TT lane count in "
            f"single-process lane mode; got max_num_seqs={max_num_seqs}, "
            f"lanes={lanes}."
        )
    per_lane = max_num_seqs // lanes
    if per_lane < 1:
        raise ValueError(
            "max_num_seqs must provide at least one request per TT lane; got "
            f"max_num_seqs={max_num_seqs}, lanes={lanes}."
        )
    return per_lane


def uses_tt_lane_coordinator(vllm_config: "VllmConfig") -> bool:
    return (
        vllm_config.parallel_config.data_parallel_size == 1
        and get_tt_data_parallel_size(vllm_config) > 1
    )


# Decode-interleave policy. TT executes a step as all-prefill or all-decode, so
# a run of prefill steps stalls every running decode for its whole duration.
# These bound that run. Read by the scheduler and by the lane coordinator.
_DECODE_INTERLEAVE_ENABLED_KEY = "decode_interleave_enabled"
_DECODE_INTERLEAVE_PREFILL_STEPS_KEY = "decode_interleave_prefill_steps"
_DECODE_INTERLEAVE_DECODE_STEPS_KEY = "decode_interleave_decode_steps"

_DECODE_INTERLEAVE_ENABLED_DEFAULT = True
_DECODE_INTERLEAVE_PREFILL_STEPS_DEFAULT = 2
_DECODE_INTERLEAVE_DECODE_STEPS_DEFAULT = 1


def _read_tt_bool(tt_config: dict[str, Any], key: str, default: bool) -> bool:
    value = tt_config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(
            f"additional_config['tt']['{key}'] must be a boolean, got {value!r}"
        )
    return value


def _read_tt_positive_int(tt_config: dict[str, Any], key: str, default: int) -> int:
    value = tt_config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"additional_config['tt']['{key}'] must be an integer >= 1, got {value!r}"
        )
    return value


_REQUEST_TIMEOUT_KEY = "request_timeout_s"


def get_tt_request_timeout_s(vllm_config: "VllmConfig") -> float:
    """Per-request wall-clock budget in seconds, counted from the request's
    arrival at the engine (queueing included); 0 (the default) disables it.

    Enforced by ``TTScheduler`` at the step boundary: a request older than the
    budget is finished with ``FinishReason.ABORT`` and ``stop_reason
    "timeout"``, its KV blocks and device state slot are released like a
    client abort, and the other requests of the step are untouched.
    """
    tt_config = get_tt_config(vllm_config)
    value = tt_config.get(_REQUEST_TIMEOUT_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(
            f"additional_config['tt']['{_REQUEST_TIMEOUT_KEY}'] must be a number "
            f">= 0 (0 disables), got {value!r}"
        )
    return float(value)


_PREFILL_CHUNK_ALIGNMENT_KEY = "prefill_chunk_alignment"
# TT chunked prefill walks a prompt in SDPA chunks of 128 query rows; a chunk
# that starts off that grid is realigned down by the model and the whole
# chunk is then padded up to the next power of two (4096 + delta -> 8192).
_PREFILL_CHUNK_ALIGNMENT_DEFAULT = 128


def get_tt_prefill_chunk_alignment(vllm_config: "VllmConfig") -> int:
    """Token alignment every partial prefill chunk must end on (0 disables).

    A request admitted on leftover budget would otherwise start every later
    chunk on an odd offset; aligning chunk ends keeps the model's chunk walk on
    its native grid for every model served through this plugin.
    """
    tt_config = get_tt_config(vllm_config)
    value = tt_config.get(
        _PREFILL_CHUNK_ALIGNMENT_KEY, _PREFILL_CHUNK_ALIGNMENT_DEFAULT
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"additional_config['tt']['{_PREFILL_CHUNK_ALIGNMENT_KEY}'] must be an "
            f"integer >= 0 (0 disables), got {value!r}"
        )
    return value


_PREFILL_CHUNK_RUN_KEY = "prefill_chunk_run"


def get_tt_prefill_chunk_run(vllm_config: "VllmConfig") -> int:
    """Chunks of ``max_num_batched_tokens`` one prefill step may give a request
    that is already mid-prompt (1 = one chunk per step, the default).

    A long prompt is otherwise one scheduler step per chunk, each with its own
    host round trip (scheduler, input build, submit, completion wait, sampler
    skip). With a run of K the continuation step carries K consecutive chunks
    of the SAME request; the model submits them back to back (K traced bucket
    replays, one wait at the end). The admission budget for new requests is
    unchanged, so burst TTFT is unaffected (doc/ttnn/09-vllm/chunk_pipelining.md).
    """
    tt_config = get_tt_config(vllm_config)
    value = tt_config.get(_PREFILL_CHUNK_RUN_KEY, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"additional_config['tt']['{_PREFILL_CHUNK_RUN_KEY}'] must be an "
            f"integer >= 1, got {value!r}"
        )
    return value


_PREFILL_SEGMENTS_KEY = "prefill_segments"
PREFILL_SEGMENTS_MAX = 16


def get_tt_prefill_segments(vllm_config: "VllmConfig") -> int:
    """``tt.prefill_segments`` = S (segmented traces, doc/ttnn/14b-persistent-stack/segtrace/runner_design.md section 4,
    change item 11; default 1 = off): a segmentable chunk (the 2,048 bucket, decoders running, mixing allowed) runs as S
    trace segments over S engine steps, and the decoders advance once per segment; 1 = one replay per chunk (today).
    1 <= S <= PREFILL_SEGMENTS_MAX. The GLM53_DECODE_INTERLEAVE_FILE control file's ``prefill_segments`` key overrides
    it at run time (TTScheduler._prefill_segments), so one resident A/Bs S = 1 / 3 / 4. The scheduler reads S ONCE per
    chunk, at its first step, and the consumer checks it against the model's ``prefill_segments`` capability (the S it
    captured)."""
    tt_config = get_tt_config(vllm_config)
    value = tt_config.get(_PREFILL_SEGMENTS_KEY, 1)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= PREFILL_SEGMENTS_MAX:
        raise ValueError(
            f"additional_config['tt']['{_PREFILL_SEGMENTS_KEY}'] must be an integer in [1, {PREFILL_SEGMENTS_MAX}], "
            f"got {value!r}"
        )
    return value


def get_tt_decode_interleave_config(
    vllm_config: "VllmConfig",
) -> tuple[bool, int, int]:
    """Return ``(enabled, prefill_steps, decode_steps)`` for decode interleave.

    ``prefill_steps`` is the number of consecutive prefill steps allowed before
    a decode-only step is inserted; ``decode_steps`` is how many decode-only
    steps that insertion runs before prefill is required again. Both are step
    counts rather than token counts because the interleave granularity on TT is
    a whole step: the device cannot mix prefill and decode rows in one batch.
    """
    tt_config = get_tt_config(vllm_config)
    return (
        _read_tt_bool(
            tt_config,
            _DECODE_INTERLEAVE_ENABLED_KEY,
            _DECODE_INTERLEAVE_ENABLED_DEFAULT,
        ),
        _read_tt_positive_int(
            tt_config,
            _DECODE_INTERLEAVE_PREFILL_STEPS_KEY,
            _DECODE_INTERLEAVE_PREFILL_STEPS_DEFAULT,
        ),
        _read_tt_positive_int(
            tt_config,
            _DECODE_INTERLEAVE_DECODE_STEPS_KEY,
            _DECODE_INTERLEAVE_DECODE_STEPS_DEFAULT,
        ),
    )


# Mixed prefill+decode steps. With the mode on, a step that has prefill work
# pending while requests decode carries both: the prefill keeps a prefill
# step's budget (capped by GLM53_MIXED_PREFILL_CAP) and every running decode
# advances one token beside it, so a decoding request is no longer stalled for
# a whole run of prefill chunks (the decode interleave policy above is the
# alternative it replaces for those steps).
_MIXED_STEPS_KEY = "mixed_prefill_decode_steps"
_MIXED_STEPS_DEFAULT = False
MIXED_STEPS_ENV = "GLM53_MIXED_STEPS"
MIXED_PREFILL_CAP_ENV = "GLM53_MIXED_PREFILL_CAP"


def _read_env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def get_tt_mixed_steps_enabled(vllm_config: "VllmConfig") -> bool:
    """Whether the scheduler may put prefill and decode work in one step.

    ``additional_config['tt']['mixed_prefill_decode_steps']`` (bool, default
    off); the ``GLM53_MIXED_STEPS`` environment variable (``0`` / ``1``), when
    set, overrides the key, so one launch configuration can be A/B'd without
    editing it.
    """
    from_env = _read_env_flag(MIXED_STEPS_ENV)
    if from_env is not None:
        return from_env
    return _read_tt_bool(
        get_tt_config(vllm_config), _MIXED_STEPS_KEY, _MIXED_STEPS_DEFAULT
    )


# Stage 12 (program 2026-10-26T21:47:35Z (1), arm C): with mixed steps on, a decoding request advances one token
# per mixed step, and a mixed step lasts a whole prefill chunk (~0.45 s at 2,048 rows), so 8 decoders beside 8 long
# prefills read TPOT ~0.6 s. GLM53_MIXED_DECODE_FOLLOW=K (default 0 = off): after every mixed step, the next K steps
# with running decodes are decode-only (~40 ms each), i.e. K + 1 tokens per prefill chunk at +K x 40 ms per chunk.
# GLM53_DECODE_INTERLEAVE_FILE=<path> (program 2026-10-26T21:58:59Z (a); alias GLM53_SCHED_CTL_FILE): an optional
# 'key=value' file re-read when its mtime changes (one stat per step), so one resident runs every scheduling arm without a
# relaunch. Keys: decode_interleave_enabled (0/1), decode_interleave_prefill_steps / decode_interleave_decode_steps
# (>= 1), mixed_steps (0/1: off, or back on when it was on at start-up), mixed_decode_follow (K >= 0), prefill_segments
# (1 .. PREFILL_SEGMENTS_MAX: get_tt_prefill_segments), prefill_segment_decodes (D >= 1:
# get_tt_prefill_segment_decodes); an absent key (or file) = the start-up value.
MIXED_DECODE_FOLLOW_ENV = "GLM53_MIXED_DECODE_FOLLOW"
SCHED_CTL_FILE_ENV = "GLM53_DECODE_INTERLEAVE_FILE"
SCHED_CTL_FILE_ALIAS_ENV = "GLM53_SCHED_CTL_FILE"


PREFILL_SEGMENT_DECODES_ENV = "GLM53_PREFILL_SEGMENT_DECODES"
PREFILL_SEGMENT_MIN_TOKENS_ENV = "GLM53_PREFILL_SEGMENT_MIN_TOKENS"


def _env_int(name: str, default: int, lo: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        n = int(value)
    except ValueError:
        n = lo - 1
    if n < lo:
        raise ValueError(f"{name} must be an integer >= {lo}, got {value!r}")
    return n


def get_tt_prefill_segment_decodes() -> int:
    """D, the decode steps per segment of a segmented chunk (``GLM53_PREFILL_SEGMENT_DECODES``, >= 1; unset = 1): the
    segment's own step plus D - 1 decode-only steps before the next segment (foldin/step_loop.md, the S x D table).
    The control file's ``prefill_segment_decodes`` overrides it at run time; read once per chunk."""
    return _env_int(PREFILL_SEGMENT_DECODES_ENV, 1, 1)


def get_tt_prefill_segment_min_tokens() -> int:
    """The shortest chunk that is segmented (``GLM53_PREFILL_SEGMENT_MIN_TOKENS``, >= 2; unset = 513): the model
    segments its 2,048-bucket keys only, and a chunk of <= 512 tokens replays the 512 bucket whole."""
    return _env_int(PREFILL_SEGMENT_MIN_TOKENS_ENV, 513, 2)


def get_tt_mixed_decode_follow() -> int:
    """Decode-only steps after every mixed step (``GLM53_MIXED_DECODE_FOLLOW``, an integer >= 0; unset = 0)."""
    value = os.environ.get(MIXED_DECODE_FOLLOW_ENV)
    if value is None or value == "":
        return 0
    try:
        k = int(value)
    except ValueError:
        k = -1
    if k < 0:
        raise ValueError(f"{MIXED_DECODE_FOLLOW_ENV} must be an integer >= 0, got {value!r}")
    return k


def get_tt_mixed_prefill_cap() -> int | None:
    """Most prompt tokens one mixed step may schedule (``GLM53_MIXED_PREFILL_CAP``).

    ``None`` (unset) means the scheduler's normal per-step prefill limit. The
    cap bounds the prefill part only, shared by every prefill row of the step;
    the step's decode rows (one token each) come on top of it.
    """
    value = os.environ.get(MIXED_PREFILL_CAP_ENV)
    if value is None or value == "":
        return None
    try:
        cap = int(value)
    except ValueError:
        cap = 0
    if cap < 1:
        raise ValueError(
            f"{MIXED_PREFILL_CAP_ENV} must be an integer >= 1, got {value!r}"
        )
    return cap
