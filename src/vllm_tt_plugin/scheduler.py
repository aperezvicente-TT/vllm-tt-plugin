# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import itertools
import os
import sys
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_tt_plugin.config import (
    get_tt_adaptive_block_max_prompt_tokens,
    get_tt_block_kv_extent_tokens,
    get_tt_decode_interleave_config,
    get_tt_mixed_prefill_cap,
    get_tt_mixed_steps_enabled,
    get_tt_mixed_decode_follow,
    get_tt_prefill_segment_decodes,
    get_tt_prefill_segment_min_tokens,
    get_tt_prefill_segments,
    PREFILL_SEGMENTS_MAX,
    SCHED_CTL_FILE_ENV,
    SCHED_CTL_FILE_ALIAS_ENV,
    get_tt_output_tokens_per_step,
    get_tt_prefill_chunk_alignment,
    get_tt_prefill_chunk_run,
    get_tt_request_timeout_s,
    get_tt_spec_plan,
    is_tt_adaptive_block_output_model,
    is_tt_block_output_model,
    uses_tt_lane_coordinator,
)
from vllm_tt_plugin.kv_connector import (
    KDACheckpointStore,
    TTPageAlignedConnector,
    kda_ckpt_at_end_mode,
    kda_ckpt_in_replay,
    lmcache_min_store_tokens,
    tail_split_min,
)
from vllm_tt_plugin import ttft_phases
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.spec_admission import MODEL_OWNED_DRAFT_METHODS

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_tt_logger(__name__)


# ``SchedulerOutput`` has no backend metadata field. Like lane step state, this
# TT-only signal is attached to the mutable output object and therefore follows
# it through the executor serialization boundary. Counts, rather than a set of
# request IDs, identify exactly how many newest in-flight frames a wholesale
# prefix-cache reset made stale.
_TT_FORCED_RESET_DISCARD_COUNTS_ATTR = "_tt_forced_reset_discard_counts"
_TT_OUTPUT_FRAME_REQ_IDS_ATTR = "_tt_output_frame_req_ids"


def set_tt_forced_reset_discard_counts(
    scheduler_output: SchedulerOutput, counts: dict[str, int]
) -> None:
    if counts:
        setattr(scheduler_output, _TT_FORCED_RESET_DISCARD_COUNTS_ATTR, dict(counts))


def get_tt_forced_reset_discard_counts(
    scheduler_output: SchedulerOutput,
) -> dict[str, int]:
    return dict(getattr(scheduler_output, _TT_FORCED_RESET_DISCARD_COUNTS_ATTR, {}))


# Per-step block-output reconciliation decisions, keyed by request id. Attached
# to the ``SchedulerOutput`` -- the ONLY object that re-associates a step's
# output with the scheduling decision that produced it: under async scheduling
# ``schedule()`` for step K+1 (which mutates per-``Request`` state) runs BEFORE
# ``update_from_output`` commits step K, so a single overwritten ``Request``
# slot would hand step K's commit the K+1 decision. The engine core holds each
# in-flight step's ``SchedulerOutput`` in its batch queue and passes it back to
# ``update_from_output``, so a map carried here always matches the output being
# committed (in sync mode the pairing is trivially the same step). Mirrors the
# forced-reset-count pattern above.
_TT_BLOCK_STEP_DECISIONS_ATTR = "_tt_block_step_decisions"


def set_tt_block_step_decisions(
    scheduler_output: SchedulerOutput, decisions: dict[str, bool]
) -> None:
    setattr(scheduler_output, _TT_BLOCK_STEP_DECISIONS_ATTR, dict(decisions))


def get_tt_block_step_decisions(
    scheduler_output: SchedulerOutput,
) -> dict[str, bool]:
    return dict(getattr(scheduler_output, _TT_BLOCK_STEP_DECISIONS_ATTR, {}))


# Marks a step that carries prefill rows AND decode rows (mixed mode, see
# ``TTScheduler._schedule_mixed``). Set only when the base scheduler actually
# scheduled both kinds, so the runner splits exactly the steps that need it and
# every other step keeps its single-kind path. Attached like the attributes
# above so it survives the executor serialization boundary.
_TT_MIXED_STEP_ATTR = "_tt_mixed_step"


def set_tt_mixed_step(scheduler_output: SchedulerOutput) -> None:
    setattr(scheduler_output, _TT_MIXED_STEP_ATTR, True)


def is_tt_mixed_step(scheduler_output: SchedulerOutput) -> bool:
    return bool(getattr(scheduler_output, _TT_MIXED_STEP_ATTR, False))


# Segmented prefill traces (doc/ttnn/14b-persistent-stack/segtrace/runner_design.md section 4, item 8): the first step
# of a segmented chunk carries {req_id: S}, S > 1 (its device step runs segment 0 of S). The request's next prefill row
# (the 1-token tail of the chunk, step S - 1) is its release. The KV connector host parks the chunk's save until then
# (TTKVConnectorHost._segment_park). Attached like the attributes above.
_TT_SEGMENTED_ATTR = "_tt_segmented"


def set_tt_segmented(scheduler_output: SchedulerOutput, segments: dict[str, int]) -> None:
    setattr(scheduler_output, _TT_SEGMENTED_ATTR, {str(rid): int(s) for rid, s in segments.items()})


def get_tt_segmented(scheduler_output: SchedulerOutput) -> dict[str, int]:
    return dict(getattr(scheduler_output, _TT_SEGMENTED_ATTR, None) or {})


# Every step that runs segments of a segmented chunk, for the runner: {"req_id", "segments" (the segment indices this
# step runs, in order), "num_segments" S, "end" (the chunk's device end e: step 0 schedules [s, e - 1), the runner
# runs [s, e))}. Step 0: [0]. A decode-only step of the hold: [s]. The release (the request's 1-token tail): every
# segment not run yet, [next .. S - 1], plus the head.
_TT_SEGMENT_STEP_ATTR = "_tt_segment_step"


def set_tt_segment_step(
    scheduler_output: SchedulerOutput, req_id: str, segments: list[int], num_segments: int, end: int
) -> None:
    setattr(scheduler_output, _TT_SEGMENT_STEP_ATTR, {
        "req_id": str(req_id), "segments": [int(x) for x in segments], "num_segments": int(num_segments),
        "end": int(end)})  # fmt: skip


def get_tt_segment_step(scheduler_output: SchedulerOutput) -> dict | None:
    v = getattr(scheduler_output, _TT_SEGMENT_STEP_ATTR, None)
    return dict(v) if v else None


_NO_VALUES: dict = {}


class _SchedCtl:
    """GLM53_DECODE_INTERLEAVE_FILE (config.SCHED_CTL_FILE_ENV): 'key=value' lines re-read when the file's mtime changes
    (one stat per step), so a running server's scheduling knobs can be switched for an A/B on one resident; absent /
    unreadable = no overrides. snapshot() returns the SAME dict object until the file changes (cheap change test)."""

    def __init__(self, path: str | None) -> None:
        self.path, self.mtime, self.values = path or None, None, _NO_VALUES

    def snapshot(self) -> dict:
        self.get("")
        return self.values

    def get(self, key: str):
        if not self.path:
            return None
        try:
            mtime = os.stat(self.path).st_mtime_ns
        except OSError:
            self.mtime, self.values = None, _NO_VALUES
            return None
        if mtime != self.mtime:
            values = {}
            try:
                with open(self.path) as f:
                    for line in f:
                        k, sep, v = line.strip().partition("=")
                        if sep and k.strip() and not k.lstrip().startswith("#"):
                            values[k.strip()] = v.strip()
            except OSError:
                values = {}
            self.mtime, self.values = mtime, values
        return self.values.get(key)


class TTSchedulingMode(Enum):
    DEFAULT = "default"
    DECODE_ONLY = "decode_only"
    PREFILL_ONLY = "prefill_only"

    @classmethod
    def from_prefill_intent(cls, prefill_intent: int) -> "TTSchedulingMode":
        if prefill_intent == 0:
            return cls.DECODE_ONLY
        if prefill_intent == 1:
            return cls.PREFILL_ONLY
        raise ValueError(f"Invalid TT scheduling intent: {prefill_intent}")


def spec_lookahead_tokens(plan, num_spec_tokens: int, method: str | None) -> int:
    """KV slots to reserve past a step's own tokens for a model-owned drafter.

    Such a drafter proposes for the next step inside the current one: after
    the accept walk commits, its fused body runs over the anchor plus every
    draft and writes K/V for those K+1 positions, none of which the current
    step's allocation covers. Upstream reserves lookahead only for the drafter
    methods it knows, and ``custom_class`` is not one of them, so without this
    reservation the rows that cross into the next block land in the null block
    and the first token that reads them diverges from plain decode. ``mtp`` is
    one of them, but upstream sizes it for its own proposer at ``K``, one short
    of what the model-owned drafter serving ``mtp`` on TT writes.

    An ngram launch also carries an admitted plan, but its drafts come from the
    plugin and its target verifies them inside the step's own allocation, so it
    reserves nothing here.
    """
    if plan is None or num_spec_tokens <= 0 or method not in MODEL_OWNED_DRAFT_METHODS:
        return 0
    return num_spec_tokens + 1


def prefill_short_first_tokens() -> int:
    """GLM53_PREFILL_SHORT_FIRST = T tokens (default 0 = off): see TTScheduler._short_first_hold"""
    try:
        return max(0, int(os.environ.get("GLM53_PREFILL_SHORT_FIRST", "0") or 0))
    except ValueError:
        return 0


def prefill_short_first_max() -> int:
    """GLM53_PREFILL_SHORT_FIRST_MAX = K (default 4): held steps in a row before the long gets one"""
    try:
        return max(1, int(os.environ.get("GLM53_PREFILL_SHORT_FIRST_MAX", "4") or 4))
    except ValueError:
        return 4


def prefill_short_first_retry() -> int:
    """GLM53_PREFILL_SHORT_FIRST_RETRY = steps (default 8) before a short a held step could not admit counts again"""
    try:
        return max(1, int(os.environ.get("GLM53_PREFILL_SHORT_FIRST_RETRY", "8") or 8))
    except ValueError:
        return 8


def prefill_short_first_inflight() -> int:
    """GLM53_PREFILL_SHORT_FIRST_INFLIGHT = N (default 2; 0 = off; acts only with GLM53_PREFILL_SHORT_FIRST = T): the
    head-less prefill chunk runs the runner keeps enqueued on the device; after enqueuing one it waits for the run N - 1
    behind it (v3b4, vllm_tt_plugin.prefill_pacer: the scheduler runs once per device run)."""
    try:
        return max(0, int(os.environ.get("GLM53_PREFILL_SHORT_FIRST_INFLIGHT", "2") or 0))
    except ValueError:
        return 2


def prefill_short_first_kv_overtake() -> bool:
    """GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE (default 1; acts only with GLM53_PREFILL_SHORT_FIRST = T and
    GLM53_ADMIT_KV_BUDGET=1): see TTScheduler._short_first_kv_hide. 0 = the gate's plain FIFO stop."""
    return os.environ.get("GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE", "1") == "1"


_ADMIT_GAUGES: dict = {}


def _admit_gauges() -> dict | None:
    """the admission-headroom gauges (program 2026-10-28T16:15:39Z / 16:15:53Z: router v6 routes on them). The API
    server's /metrics carries a gauge set in this EngineCore process only through PROMETHEUS_MULTIPROC_DIR
    (serve_window.sh sets it, as for the step meter); unset, or GLM53_ADMIT_HEADROOM_METRIC=0: none"""
    if "g" not in _ADMIT_GAUGES:
        g = None
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR") and os.environ.get("GLM53_ADMIT_HEADROOM_METRIC", "1") == "1":
            try:
                from prometheus_client import Gauge  # noqa: PLC0415

                g = {k: Gauge(f"glm53_admit_{k}_tokens", doc, multiprocess_mode="livemostrecent") for k, doc in (
                    ("pool", "KV pool of the GLM53_ADMIT_KV_BUDGET gate, tokens (blocks less the null block)"),
                    ("reserved", "KV reserves of the resident requests (running + waiting for remote KV), tokens"),
                    ("headroom", "pool - reserved: the largest reserve the gate admits now, tokens"),
                    ("queued", "KV reserves of the waiting requests not yet admitted, tokens"),
                )}  # fmt: skip
            except (ImportError, ValueError):
                g = None  # (no prometheus_client, or registered twice in one process: tests)
        _ADMIT_GAUGES["g"] = g
    return _ADMIT_GAUGES["g"]


def interleave_tail_exempt_tokens() -> int:
    """GLM53_INTERLEAVE_TAIL_EXEMPT=1 (default 0; program 2026-10-28T06:51:35Z (2)): a request's LAST prefill chunk,
    when it holds at most GLM53_INTERLEAVE_TAIL_EXEMPT_TOKENS (default 128) tokens, runs in the same prefill run as its
    predecessor: the decode interleave does not insert its D decode-only steps before it. Under k4d5p's P1/D24 the
    storing tail split's 128-token last chunk of a 32,768-token arrival cost one whole D window (24 x 0.156 s = 3.7 s of
    its 67.3-s TTFT with 8 decoders). The D window follows the exempt chunk as usual, so the decoders lose one short
    chunk's time, never more. Same chunks in the same order: bitwise. 0 = off."""
    if os.environ.get("GLM53_INTERLEAVE_TAIL_EXEMPT", "0") != "1":
        return 0
    try:
        return max(1, int(os.environ.get("GLM53_INTERLEAVE_TAIL_EXEMPT_TOKENS", "128") or 128))
    except ValueError:
        return 128


class TTDecodeInterleavePolicy:
    """Bounds how many consecutive prefill steps may stall running decodes.

    A TT step carries either prefill rows or decode rows, never both, so a
    prompt split into N chunks occupies N consecutive prefill steps and every
    running decode waits for all of them. This policy inserts decode-only steps
    into such a run. ``docs/SCHEDULING.md`` covers why, how the two counts
    interact, and what the defaults rest on.

    Three invariants the code depends on:

    - Reaching the decode allowance never forces prefill. It only stops the
      policy choosing decode; whether prefill then runs is up to the scheduler
      and to whether prefill work is pending.
    - ``has_running_decode`` must exclude partial-prefill continuations. A
      decode step cannot advance one, so interleaving on its account trades a
      productive prefill step for an empty one.
    - A decode step taken with no prefill pending is an ordinary decode, not
      part of an insertion, and clears both counters.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        (
            self._enabled,
            self._prefill_steps,
            self._decode_steps,
        ) = get_tt_decode_interleave_config(vllm_config)
        self._prefill_run = 0
        self._decode_run = 0

    def wants_decode_step(
        self, *, has_pending_prefill: bool, has_running_decode: bool
    ) -> bool:
        """Whether to spend this step on decode although prefill work is pending.

        ``has_running_decode`` must exclude partial-prefill continuations: a
        decode step cannot advance one, so interleaving on its account would
        trade a productive prefill step for an empty one.
        """
        if not self._enabled or not has_pending_prefill or not has_running_decode:
            return False
        return (
            self._prefill_run >= self._prefill_steps
            and self._decode_run < self._decode_steps
        )

    def record_step(self, *, is_decode: bool, prefill_pending: bool) -> None:
        """Advance the counters with the phase the step actually ran.

        Called for every step, including one that scheduled no tokens. A
        decode-only step that schedules nothing (upstream's running loop skips a
        request whose async placeholders have already reached ``max_tokens``)
        still consumes its decode allowance; leaving the counters untouched
        there would re-pick decode on every following step and livelock the
        engine on empty steps.

        ``prefill_pending`` says whether any prefill work existed when the step
        was chosen. A decode step taken with nothing pending is an ordinary
        decode, not part of an insertion, so it clears both counters: counting
        it would leave ``_prefill_run`` at its bound with part of the decode
        allowance already spent, and the next arriving prompt would then lose
        its first prefill step to an insertion it was never part of.
        """
        if not prefill_pending:
            self._prefill_run = 0
            self._decode_run = 0
            return
        if is_decode:
            self._decode_run += 1
            if self._decode_run >= self._decode_steps:
                # Allowance spent: require a fresh run of prefill steps before
                # the next insertion, which is what stops the policy from
                # holding the device in decode while a prompt waits.
                self._decode_run = 0
                self._prefill_run = 0
        else:
            self._prefill_run += 1
            self._decode_run = 0

    def note_mixed_step(self) -> None:
        """A mixed step advanced the running decodes alongside the prefill.

        Nothing stalled them, so there is no run of prefill steps to bound:
        both counters clear, as after an ordinary decode. A later step that
        cannot mix (see ``TTScheduler._mixed_prefill_budget``) starts a fresh
        run instead of inheriting one the mixed steps never were.
        """
        self._prefill_run = 0
        self._decode_run = 0


def _effective_prefill_step_limit(scheduler_config) -> tuple[int, str]:
    """The most tokens one scheduling step can hand a single new request, named.

    Resolved exactly as vLLM's ``Scheduler.__init__`` resolves its per-step
    token budget: ``max_num_scheduled_tokens`` when it is set, otherwise
    ``max_num_batched_tokens``. A positive ``long_prefill_token_threshold``
    then caps that limit. Returns ``(limit, name_of_the_binding_cap)``; the
    limit is ``sys.maxsize`` when nothing is configured.
    """
    scheduled = getattr(scheduler_config, "max_num_scheduled_tokens", None)
    if scheduled is not None:
        limit, name = int(scheduled), "max_num_scheduled_tokens"
    else:
        batched = int(getattr(scheduler_config, "max_num_batched_tokens", 0) or 0)
        limit, name = (
            (batched, "max_num_batched_tokens")
            if batched > 0
            else (
                sys.maxsize,
                "max_num_batched_tokens",
            )
        )
    threshold = int(getattr(scheduler_config, "long_prefill_token_threshold", 0) or 0)
    if 0 < threshold < limit:
        limit, name = threshold, "long_prefill_token_threshold"
    return limit, name


def _plan_cut_hint(request) -> int:
    """DEV ONLY (Stage 12 diagnostic, program 2026-10-23T12:18:35Z): kv_transfer_params {"tt.plan_cut": N} makes a
    request's prefill plan cut at token N and then follow a connector hit's plan, as if N tokens had been loaded: no
    chunk crosses N, and the chunk that starts at N gets a fresh request's first-chunk budget (one step, not the
    prefill_chunk_run run). With {"tt.cold": true} a cold request then reproduces the chunk geometry of an N-token hit +
    continuation without any hit. Absent (the default) or outside (0, prompt length): 0, no effect. The split applies
    it only on the prefill alignment grid (an off-grid N is ignored there: no aligned chunk could end at it)."""
    kv = getattr(request, "kv_transfer_params", None)
    if kv is None:
        sp = getattr(request, "sampling_params", None)
        kv = (getattr(sp, "extra_args", None) or {}).get("kv_transfer_params")
    try:
        n = int((kv or {}).get("tt.plan_cut") or 0)
    except (TypeError, ValueError):
        return 0
    return n if 0 < n < int(getattr(request, "num_prompt_tokens", 0) or 0) else 0


def _stores_nothing(request, mode: str) -> bool:
    """Does the request store nothing a later hit could use (GLM53_PF_TAIL_SPLIT_SKIP, see TTScheduler.__init__)?
    "floor": its prompt is under GLM53_LMCACHE_MIN_STORE_TOKENS (no LMCache store, no KDA checkpoint, for every request
    of that length). "1": also any request the connector marked lmcache.skip_save (tt.cold probes); the mark lives in
    kv_transfer_params, on the request or in its sampling params' extra_args (where the connector writes it)."""
    floor = lmcache_min_store_tokens()
    if floor and int(getattr(request, "num_prompt_tokens", 0) or 0) < floor:
        return True
    if mode != "1":
        return False
    for kv in (getattr(request, "kv_transfer_params", None),
               (getattr(getattr(request, "sampling_params", None), "extra_args", None) or {}).get("kv_transfer_params")):
        if (kv or {}).get("lmcache.skip_save"):
            return True
    return False


def _chunk_cap_hint(request) -> int:
    """DEV ONLY (a prefill-plan diagnostic): kv_transfer_params {"tt.chunk_cap": N} caps every prefill chunk of the
    request at N tokens (N a multiple of the prefill alignment), so its prefill runs in buckets <= N (512: no chunk
    reaches the 2,048 bucket). Absent (the default), non-positive or unparsable: 0, no effect. The split applies it only
    as a positive multiple of the alignment (any other N is ignored there: a cap off the grid leaves no aligned chunk)."""
    kv = getattr(request, "kv_transfer_params", None)
    if kv is None:
        sp = getattr(request, "sampling_params", None)
        kv = (getattr(sp, "extra_args", None) or {}).get("kv_transfer_params")
    try:
        n = int((kv or {}).get("tt.chunk_cap") or 0)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _validate_prefill_chunk_alignment(align: int, scheduler_config) -> None:
    """Reject caps that can never let a new request reach the alignment grid.

    ``_align_prefill_chunk_end`` defers a new request whose chunk falls short
    of the grid; that only converges if a later step can offer a larger chunk.
    The binding cap is the one vLLM schedules with (``max_num_scheduled_tokens``
    when set, else ``max_num_batched_tokens``), further capped by a positive
    ``long_prefill_token_threshold``.
    """
    limit, name = _effective_prefill_step_limit(scheduler_config)
    if limit < align:
        raise ValueError(
            f"additional_config['tt']['prefill_chunk_alignment']={align} cannot be "
            f"reached: scheduler {name}={limit} is a permanent per-step cap "
            "below the "
            "alignment. Raise the cap to at least the alignment or set "
            "prefill_chunk_alignment to 0."
        )



# GLM53_ADMIT_KV_BUDGET: the statuses whose requests hold KV blocks (TTScheduler._kv_budget_defers)
_KV_RESIDENT_STATUSES = frozenset({RequestStatus.RUNNING, RequestStatus.WAITING_FOR_REMOTE_KVS})


def _admit_reserve_tokens() -> int:
    """GLM53_ADMIT_RESERVE_TOKENS (default 32,768; 0 = no cap): the output tokens a request's KV-budget reserve counts
    at most (program 2026-10-27T00:21:37Z 14 [M])"""
    try:
        return max(int(os.environ.get("GLM53_ADMIT_RESERVE_TOKENS", "32768") or 0), 0)
    except ValueError:
        return 32768

class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - By default no mixed prefill+decode batches: each batch is either
      all-prefill or all-decode. With mixed steps on
      (``tt.mixed_prefill_decode_steps`` / ``GLM53_MIXED_STEPS=1``) a step
      with prefill work pending while requests decode carries both: the
      prefill gets a prefill step's (capped) budget and every running decode
      its token on top (``_schedule_mixed``); the runner executes the two
      parts back to back as one step. Configurations whose runner paths
      cannot split a step keep the single-kind steps
      (``_mixed_steps_unsupported_reason``).
    - Token-chunked prefill is allowed: a long prefill may be split across
      steps. After a partial chunk the base scheduler keeps the request in
      ``running`` with ``is_prefill_chunk=True``; later prefill steps schedule
      the next chunk until the prompt is fully computed. Those continuations
      count as prefill work here, never as decodes.

    The base scheduler holds temporarily blocked prefill requests (e.g. while
    a structured-output grammar compiles) in ``skipped_waiting`` rather than
    ``waiting``. Its prefill loop revisits that queue and promotes requests
    whose dependency is ready. Decode-only scheduling must hide both queues so
    that the base scheduler cannot admit a prefill into a decode step.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap
    - under async scheduling a preempted request keeps the tokens it had
      already scheduled but not yet received, and needs no TT-side handling
      for them. Those tokens are valid: the forward that produced them ran to
      completion before the preempt freed any block, device submits form a
      strict queue, and every async op is forced to complete before the next
      prefill, so no later write can reach the KV they were computed against.
      The base class appends them on arrival and the resumed prefill replays
      them. ``Request.async_tokens_to_discard`` serves the wholesale
      ``reset_prefix_cache`` teardown only; wiring ordinary preemption into it
      drops valid tokens and silently truncates the response.
      A preempted request with an outstanding output placeholder stays in the
      waiting queue until that valid frame is accounted for. Resuming earlier
      would replay and sample against the old token history, producing a second
      physical frame for the same single placeholder and advancing seeded RNG
      state for a token that must be discarded.

    Supports ``set_forced_mode`` for lane coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when there is no pending prefill work).
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer prefill
      when pending prefill work exists, but fall back to decode-only if
      prefill cannot make progress and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm_tt_plugin.kv_connector import _COUNTER_SOURCES  # noqa: PLC0415  (the counters line)

        _COUNTER_SOURCES["sched"].add(self)
        self._forced_mode = TTSchedulingMode.DEFAULT
        # Read only in DEFAULT mode. A lane coordinator sets a forced mode
        # before every one of its lanes' schedule() calls and owns the one
        # policy instance for the whole step, so this one stays dormant there.
        self._decode_interleave = TTDecodeInterleavePolicy(self.vllm_config)
        self._pending_forced_reset_discard_counts: dict[str, int] = {}
        # Block-step decisions for the step currently being committed, read off
        # its SchedulerOutput in update_from_output so _update_request_with_output
        # sees the decision that produced THIS output (not a later schedule).
        self._committing_block_step_decisions: dict[str, bool] = {}
        # Request id that owns the adaptive model's SINGLE speculative
        # session, mirrored from scheduling-side facts (see
        # _mirror_spec_session). Only its owner can emit a block.
        self._spec_session_owner: str | None = None
        self._pending_async_output_frames: dict[str, int] = {}
        self._widest_decode_batch_size = 0
        # Per-request wall-clock budget (tt.request_timeout_s; 0 = none). The
        # clock is an attribute so a host test can drive it deterministically.
        self._request_timeout_s = get_tt_request_timeout_s(self.vllm_config)
        self._tt_clock = time.time
        self._prefill_chunk_alignment = (
            get_tt_prefill_chunk_alignment(self.vllm_config)
            if self.scheduler_config.enable_chunked_prefill
            else 0
        )
        if self._prefill_chunk_alignment > 1:
            _validate_prefill_chunk_alignment(
                self._prefill_chunk_alignment, self.scheduler_config
            )
            logger.info(
                "TT scheduler: partial prefill chunks end on %d-token boundaries",
                self._prefill_chunk_alignment,
            )
        # tt.prefill_chunk_run = K: a continuation step of a lone mid-prompt
        # request may carry K chunks (see _prefill_chunk_run_budget).
        self._prefill_chunk_run = (
            get_tt_prefill_chunk_run(self.vllm_config)
            if self.scheduler_config.enable_chunked_prefill
            else 1
        )
        if self._prefill_chunk_run > 1:
            logger.info(
                "TT scheduler: a lone mid-prompt request gets up to %d chunks per prefill step",
                self._prefill_chunk_run,
            )
        # GLM53_PF_RESUME_GRID ("auto" = max_num_batched_tokens, the model's prefill chunk; "0" = off): plan-aligned
        # resume cuts. The model splits a chunk into prefill-chunk-sized sub-chunks counted from the chunk's start, and
        # a cold prompt's chunks start on that grid; a resumed hit's continuation starts at the hit length, so all its
        # sub-chunks were offset from the cold plan's and its numerics differed from cold by the chunk-plan sensitivity.
        # A chunk that starts off the grid now ends at the next grid boundary: from there on a resumed request runs
        # the cold plan's sub-chunks, and only the first partial unit differs.
        spec = os.environ.get("GLM53_PF_RESUME_GRID", "auto").strip().lower()
        self._resume_grid = (
            int(self.scheduler_config.max_num_batched_tokens) if spec == "auto" else max(int(spec or 0), 0)
        )
        if self._resume_grid:
            logger.info(
                "TT scheduler: a chunk starting off the %d-token prefill grid ends at the next grid boundary",
                self._resume_grid,
            )
        # GLM53_PF_TAIL_SPLIT_SKIP: the KDA-checkpoint tail split (a prompt-finishing chunk stops on the last page
        # boundary so a later hit can restore that state) is skipped for a request that stores nothing, which then runs
        # ONE replay instead of two (at 1,000 tokens: 896 + 104). "floor" (default; program 2026-10-24T06:21:05Z):
        # only prompts under GLM53_LMCACHE_MIN_STORE_TOKENS, which nobody stores -- a tt.cold probe keeps the plan a
        # first-time request of its length runs, so a cold ladder row stays a real first-time TTFT and the hit gates
        # keep their stored-plan reference; "1": any lmcache.skip_save request (tt.cold too; lever B as first ruled,
        # 05:28:50Z); "0": never (cf3h-c).
        self._tail_split_skip = os.environ.get("GLM53_PF_TAIL_SPLIT_SKIP", "floor").strip().lower()
        # GLM53_PF_TAIL_SPLIT_MIN = N (default 8,192; 0 = off; program 2026-10-28T02:28:53Z (2)): a prompt shorter
        # than N tokens gets NO KDA-checkpoint tail split, cold or storing. The split cost every short storing request
        # one extra replay (+0.18 s at 2K: k4d8's ladder); without it the plan no longer depends on storing below N.
        # The worker checkpoints such a prompt at its LAST chunk end instead (tail_split_min() in kv_connector), so a
        # repeat recomputes at most one chunk
        self._tail_split_min = tail_split_min()
        self._ckpt_at_end = kda_ckpt_at_end_mode()  # the split's counterpart in the worker's checkpoint rule
        self._ckpt_in_replay = kda_ckpt_in_replay()  # the one-replay store: no split for any storing prompt
        if self._tail_split_skip not in ("0", "1", "floor"):
            raise ValueError(f"GLM53_PF_TAIL_SPLIT_SKIP={self._tail_split_skip!r}: expected 0, 1 or floor")
        # GLM53_FIRST_TOKEN_YIELD = 1 (cf3h-d lever A's stopgap, program 2026-10-24T05:28:50Z; the served cf3h-d
        # target sets it, the default "0" keeps every other construction's step sequence): the step
        # after one that finishes a prompt schedules nothing. vLLM's batch queue (EngineCore.step_with_batch_queue,
        # depth 2) pops a step's output only once the NEXT step is submitted, and the TT runner builds and submits a
        # decode on the engine thread (the page-table publish and the input writes, ~50 ms at 1M), so the request's
        # token 0 waited for its own first decode's submit. An empty step submits in microseconds: the prefill's output
        # (token 0) is popped and streamed first, and the first decode is scheduled on the next call.
        self._first_token_yield = os.environ.get("GLM53_FIRST_TOKEN_YIELD", "0") == "1"
        self._yield_pending = False
        self.first_token_yields = 0
        self._output_tokens_per_step = get_tt_output_tokens_per_step(self.vllm_config)
        self._is_block_output_model = is_tt_block_output_model(self.vllm_config)
        self._init_mixed_steps()
        # GLM53_MIXED_DECODE_FOLLOW / GLM53_SCHED_CTL_FILE (config.get_tt_mixed_decode_follow): decode-only steps after
        # each mixed step; counters for the step log
        self._mixed_follow_env = get_tt_mixed_decode_follow()
        self._sched_ctl = _SchedCtl(os.environ.get(SCHED_CTL_FILE_ENV) or os.environ.get(SCHED_CTL_FILE_ALIAS_ENV))
        self._mixed_follow_left = 0
        self.mixed_follow_steps = 0
        # tt.prefill_segments (runner_design.md item 11): S of the segmented-trace form; the control file overrides it
        self._prefill_segments_init = get_tt_prefill_segments(self.vllm_config)
        self._prefill_segment_decodes_init = get_tt_prefill_segment_decodes()
        self._segment_min_tokens = get_tt_prefill_segment_min_tokens()
        self._segment_hold: dict | None = None  # the segmented chunk in flight (_segment_begin .. its release)
        self._segment_pass, self._segment_start, self._segment_pass_rows = False, None, 0
        self._segment_follow_left = 0  # D - 1 decode-only steps after a release (the last segment's D decodes)
        # the start-up values the control file overrides (and falls back to when a key / the file goes away)
        self._mixed_init = bool(self._mixed_steps)
        pol = self._decode_interleave
        self._interleave_init = (pol._enabled, pol._prefill_steps, pol._decode_steps)
        self._ctl_applied = None
        # KV connector (LMCache / example connectors): hits end on a page, and
        # with the KDA checkpoint policy on a checkpointed page (finding J3);
        # the policy also ends a prompt's prefill on its last page boundary so
        # the slot's KDA state can be checkpointed there (page_contract.md §5).
        self._kda_checkpoint_store = KDACheckpointStore.from_vllm_config(
            self.vllm_config
        )
        # The checkpoint / hit grid: a page, and at least the prefill
        # continuation alignment -- a hit resumes the prompt there, and a
        # sparse-regime GLM-5.3-Flash prefill only starts on 128-token
        # boundaries (a 1,000-token prompt checkpointed at the 64-token page 960
        # left a tail chunk starting at 960: 2x4 mock 2026-10-22 22:37Z).
        self._checkpoint_grid = max(
            int(self.cache_config.block_size), int(self._prefill_chunk_alignment or 0)
        )
        if self.connector is not None:
            self.connector = TTPageAlignedConnector(
                self.connector,
                self._checkpoint_grid,
                self._kda_checkpoint_store,
            )
        # Model-side prefix cache (vllm_tt_plugin.prefix_oracle): the model keeps
        # snapshots of its non-paged state at boundaries of `oracle.boundary`
        # tokens. vLLM's local page hit is truncated to the longest snapshotted
        # boundary of the prompt (the model cannot continue a KDA recurrence
        # from a page that has no state), and the final prefill chunk of every
        # prompt stops on its last boundary so that snapshot exists for later
        # hits (the tail runs as the next chunk). Installed only when the model
        # registered an oracle before this scheduler was built (in-process
        # executor: the model loads first) and prefix caching is on.
        self._prefix_oracle = None
        if self.cache_config.enable_prefix_caching:
            from vllm_tt_plugin import prefix_oracle as _po

            self._prefix_oracle = _po.get()
            if self._prefix_oracle is not None:
                self._install_prefix_oracle(self._prefix_oracle)
                logger.info(
                    "TT scheduler: model prefix oracle installed (boundary %d): "
                    "page hits truncate to snapshotted boundaries",
                    int(self._prefix_oracle.boundary),
                )
        # Adaptive: emit the block only on a solo decode step; batch >1 decodes
        # as plain baseline. Lets max_num_seqs>1 coexist with block-output.
        self._is_adaptive_block = is_tt_adaptive_block_output_model(self.vllm_config)
        # Prompt-length frontier for the block path (0 = none): a longer prompt
        # is served as plain baseline by the model for its whole lifetime, so
        # its steps reserve width-1 even when solo.
        self._adaptive_block_max_prompt = get_tt_adaptive_block_max_prompt_tokens(
            self.vllm_config
        )
        speculative_config = self.vllm_config.speculative_config
        self.num_lookahead_tokens = max(
            self.num_lookahead_tokens,
            spec_lookahead_tokens(
                get_tt_spec_plan(self.vllm_config),
                self.num_spec_tokens,
                speculative_config.method if speculative_config is not None else None,
            ),
        )
        if self._is_block_output_model:
            # KV pages for the WHOLE step, not just the one token upstream
            # counts. A block-output decode runs several internal verify
            # iterations against vLLM-owned KV before it returns, and those
            # iterations write past the position upstream allocated for:
            # schedule() calls allocate_slots with num_lookahead_tokens, which
            # is 0 here because there is no vLLM speculative_config, and
            # raising Request.num_output_placeholders afterwards accounts for
            # pending OUTPUT tokens without allocating anything. The model's
            # refresh_page_tables then pads the missing columns with zero and
            # the verify reads and writes the null block.
            #
            # Reserved as twice the emitted width: the committed block, plus
            # headroom for the final iteration's verify positions and for
            # accepted tokens carried past the emitted width. Both are bounded
            # by the packed-verify width, which is far below the block width
            # (6 against 64 at the shipped default), so this is generous and
            # costs a page or two per request.
            #
            # Twice the width is NOT a bound in general, though: it only covers
            # the physical extent while the block is at least as wide as the
            # verification. A model configured the other way round -- dFlash at
            # GEMMA4_DFLASH_SERVE_BLOCK=2 with GEMMA4_DFLASH_VERIFY=7 emits 2
            # tokens and writes 8 verification rows -- needs 2 + 8 while this
            # reserves 4, and the step then writes positions that have no
            # request block (vllm-tt-plugin#118 review, finding 2). So a model
            # may declare the extent it actually touches and we honour whichever
            # is larger.
            declared_extent = get_tt_block_kv_extent_tokens(self.vllm_config)
            self.num_lookahead_tokens = max(
                int(getattr(self, "num_lookahead_tokens", 0) or 0),
                2 * int(self._output_tokens_per_step),
                int(declared_extent),
            )
            assert self.num_sampled_tokens_per_step == 1, (
                "Block-output accounting requires upstream to reserve exactly "
                "one sampled-token placeholder"
            )
            assert not self.kv_cache_manager.enable_caching, (
                "Block-output models bypass the kv_cache_manager.cache_blocks "
                "call made by AsyncScheduler._update_request_with_output and "
                "must disable prefix caching"
            )

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    # --- Mixed prefill+decode steps ---------------------------------------

    def _init_mixed_steps(self) -> None:
        """Resolve the mixed-step mode and its prefill cap once, at start-up.

        The cap defaults to the normal per-step prefill limit, so a mixed step
        hands the prefill what a prefill step would (see
        ``_mixed_prefill_budget``). A cap below the chunk alignment could never
        carry a partial chunk, so the prefill would only advance once every
        decode had finished; it is refused instead.
        """
        # Requests whose sampling uses logits processors that hold per-row
        # state over the whole persistent batch (min_p, logit_bias,
        # min_tokens). A mixed step samples its prefill and decode rows as two
        # sub-batches, which those processors cannot index, so no step mixes
        # while one is live.
        self._mixed_ineligible_req_ids: set[str] = set()
        self._mixed_prefill_room: int | None = None
        self._mixed_decode_req_ids: frozenset[str] = frozenset()
        self._mixed_prefill_cap = 0
        self._mixed_steps = get_tt_mixed_steps_enabled(self.vllm_config)
        if not self._mixed_steps:
            return
        reason = self._mixed_steps_unsupported_reason()
        if reason is not None:
            logger.warning(
                "TT scheduler: mixed prefill+decode steps were requested but stay "
                "off: %s; steps remain all-prefill or all-decode",
                reason,
            )
            self._mixed_steps = False
            return
        step_limit, _ = _effective_prefill_step_limit(self.scheduler_config)
        cap = get_tt_mixed_prefill_cap()
        self._mixed_prefill_cap = step_limit if cap is None else cap
        align = self._prefill_chunk_alignment
        if align > 1 and self._mixed_prefill_cap < align:
            raise ValueError(
                f"GLM53_MIXED_PREFILL_CAP={self._mixed_prefill_cap} is below the "
                f"prefill chunk alignment ({align}): a mixed step could never "
                "carry a partial prefill chunk. Raise the cap to at least the "
                "alignment."
            )
        logger.info(
            "TT scheduler: mixed prefill+decode steps on: every running decode "
            "advances in each prefill step; prefill part capped at %d prompt "
            "tokens per step, decode rows on top",
            min(self._mixed_prefill_cap, int(self.max_num_scheduled_tokens)),
        )

    def _mixed_steps_unsupported_reason(self) -> str | None:
        """Why this configuration cannot run mixed steps, or ``None``.

        Each case is a runner path that assumes one step kind per step:
        multi-token outputs and drafts are reconciled per step, lanes merge
        whole steps, and rope deltas / multimodal inputs are gathered over the
        whole persistent batch.
        """
        if not self.scheduler_config.enable_chunked_prefill:
            # The mixed prefill share is held to its cap by cutting chunks,
            # which a model that cannot resume a prefill must never get.
            return "chunked prefill is disabled"
        if self._is_block_output_model:
            return "block-output models commit a multi-token canvas per step"
        if (
            self.vllm_config.speculative_config is not None
            or self.num_spec_tokens > 0
            or get_tt_spec_plan(self.vllm_config) is not None
        ):
            return "speculative decoding is configured"
        if uses_tt_lane_coordinator(self.vllm_config):
            return "lane mode coordinates whole prefill or decode steps across lanes"
        model_config = self.vllm_config.model_config
        if getattr(model_config, "uses_mrope", False) or getattr(
            model_config, "is_multimodal_model", False
        ):
            return "multimodal / mrope inputs are gathered over the whole batch"
        if getattr(model_config, "logits_processors", None):
            return "custom logits processors keep per-row state over the whole batch"
        return None

    @staticmethod
    def _uses_batch_indexed_sampling(request: Request) -> bool:
        """Whether the request activates a builtin logits processor whose state
        is indexed by persistent-batch row (see ``_init_mixed_steps``)."""
        params = request.sampling_params
        if params is None:
            return False
        return (
            bool(params.min_p and params.min_p > 0)
            or bool(params.logit_bias)
            or bool(params.min_tokens and params.min_tokens > 0)
        )

    def _apply_sched_ctl(self) -> None:
        """GLM53_DECODE_INTERLEAVE_FILE: apply the control file's decode-interleave / mixed-steps keys once per change
        (mixed steps can go off at run time, and back on only when they were on at start-up: the cap and the support
        checks are resolved then)."""
        ctl = getattr(self, "_sched_ctl", None)
        if ctl is None or not ctl.path:
            return
        vals = ctl.snapshot()
        if vals is getattr(self, "_ctl_applied", None):
            return

        def _int(key, default, lo):
            try:
                v = int(vals[key])
            except (KeyError, ValueError):
                return default
            return v if v >= lo else default

        en, ps, ds = self._interleave_init
        pol = self._decode_interleave
        pol._enabled = bool(_int("decode_interleave_enabled", int(en), 0))
        pol._prefill_steps = _int("decode_interleave_prefill_steps", ps, 1)
        pol._decode_steps = _int("decode_interleave_decode_steps", ds, 1)
        self._mixed_steps = bool(_int("mixed_steps", int(self._mixed_init), 0)) and self._mixed_init
        self._ctl_applied = vals
        logger.info(
            "TT scheduler: control file %s applied: decode interleave %s P%d/D%d, mixed steps %s, mixed decode "
            "follow %d, prefill segments %d x %d decodes", ctl.path, "on" if pol._enabled else "off", pol._prefill_steps,
            pol._decode_steps, "on" if self._mixed_steps else "off", self._mixed_decode_follow(), self._prefill_segments(),
            self._prefill_segment_decodes(),
        )

    def _prefill_segments(self) -> int:
        """S for the segmented-trace form (9's hold / release, runner_design.md items 7-10, read it once per chunk at its
        step 0): the control file's prefill_segments when set and in [1, PREFILL_SEGMENTS_MAX], else tt.prefill_segments"""
        ctl = getattr(self, "_sched_ctl", None)
        v = ctl.get("prefill_segments") if ctl is not None else None
        if v is not None:
            try:
                n = int(v)
            except ValueError:
                n = 0
            if 1 <= n <= PREFILL_SEGMENTS_MAX:
                return n
        return int(getattr(self, "_prefill_segments_init", 1))

    def _prefill_segment_decodes(self) -> int:
        """D, the decode steps per segment (config.get_tt_prefill_segment_decodes): the control file's
        prefill_segment_decodes when set and >= 1, else GLM53_PREFILL_SEGMENT_DECODES; read once per chunk"""
        ctl = getattr(self, "_sched_ctl", None)
        v = ctl.get("prefill_segment_decodes") if ctl is not None else None
        if v is not None:
            try:
                n = int(v)
            except ValueError:
                n = 0
            if n >= 1:
                return n
        return int(getattr(self, "_prefill_segment_decodes_init", 1))

    def _mixed_decode_follow(self) -> int:
        """K decode-only steps after each mixed step: the control file's mixed_decode_follow when set, else the env's"""
        ctl = getattr(self, "_sched_ctl", None)
        v = ctl.get("mixed_decode_follow") if ctl is not None else None
        if v is not None:
            try:
                return max(0, int(v))
            except ValueError:
                pass
        return int(getattr(self, "_mixed_follow_env", 0) or 0)

    def _mixed_prefill_budget(self) -> int:
        """Prompt tokens the prefill part of this step may take; 0 = do not mix.

        The prefill part gets what a prefill step would, the step budget,
        capped by ``GLM53_MIXED_PREFILL_CAP``; the decode rows (one token each)
        come on top of it (``_schedule_mixed``). TT submits the two parts as
        separate device calls, so each stays within the shapes a single-kind
        step of its kind uses, and a prompt's chunks keep their size however
        many requests decode beside it: taking the decode tokens out of the
        prefill's share would cut every partial chunk down a whole alignment
        step (2,047 tokens round to 1,920) for a single decoder. A budget below
        one aligned chunk (start-up validation rules it out) would not mix, so
        a prefill can never starve behind the decodes. Speculative drafts keep
        the step out: a verify's width is reconciled per step.
        """
        if not getattr(self, "_mixed_steps", False):
            return 0
        if not any(not r.is_prefill_chunk for r in self.running):
            return 0
        if any(r.spec_token_ids for r in self.running):
            return 0
        ineligible = getattr(self, "_mixed_ineligible_req_ids", None)
        if ineligible:
            ineligible.intersection_update(self.requests)
            if ineligible:
                return 0
        budget = min(int(self._mixed_prefill_cap), int(self.max_num_scheduled_tokens))
        if budget < max(1, int(getattr(self, "_prefill_chunk_alignment", 0) or 0)):
            return 0
        return budget

    def _schedule_mixed(self, prefill_budget: int) -> SchedulerOutput:
        """One base-scheduler pass over decodes, partial prefills and waiting
        requests, so the KV-connector metadata, block allocation and async
        placeholders are built once for the whole step.

        The step budget is the decode rows plus ``prefill_budget``, so a mixed
        step may total up to one token per decode more than the configured step
        budget (nothing on TT sizes a buffer by the step total; each device
        call keeps its single-kind shape). The prefill rows alone are held to
        ``prefill_budget`` through the block-aligned-split hook
        (``_mamba_block_aligned_split``), which runs for every request before
        its blocks are allocated. A decode that upstream skips (async
        placeholders already at ``max_tokens``) therefore leaves its token
        unused rather than growing a prefill chunk past the cap, and the cap
        holds whatever order ``running`` is in.
        """
        decode_ids = frozenset(
            r.request_id for r in self.running if not r.is_prefill_chunk
        )
        saved_budget = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = len(decode_ids) + int(prefill_budget)
        self._mixed_prefill_room = int(prefill_budget)
        self._mixed_decode_req_ids = decode_ids
        self._segment_pass_open(bool(decode_ids))
        self._split_trace = []
        hidden = self._short_first_kv_hide()  # GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE
        try:
            result = super().schedule()
        finally:
            self._short_first_kv_restore(hidden)
            self.max_num_scheduled_tokens = saved_budget
            self._mixed_prefill_room = None
            self._mixed_decode_req_ids = frozenset()
            start = self._segment_pass_close()
        self._segment_begin(result, start, mixed=True)
        self._decode_interleave.note_mixed_step()
        scheduled = result.num_scheduled_tokens
        decode_rows = sum(1 for req_id in scheduled if req_id in decode_ids)
        if 0 < decode_rows < len(scheduled):
            set_tt_mixed_step(result)
        return self._finalize_scheduler_output(
            result, is_decode=decode_rows > 0, decode_rows=decode_rows
        )

    # Host-only sampling controls and their neutral values: any of these
    # forces the step onto host sampling (check_perform_device_sampling),
    # which cannot construct a multi-token canvas and kills the engine.
    # Penalties do not force host sampling, but a non-neutral value flips
    # InputBatch.no_penalties and makes every block decode step build (and
    # discard) penalty token tensors that grow with the committed session
    # length; the frontend neutralizes them, so mirror that here for
    # prebuilt requests that bypassed it.
    _BLOCK_HOST_ONLY_SAMPLING_NEUTRAL = (
        ("min_p", 0.0),
        ("min_tokens", 0),
        ("logit_bias", None),
        ("allowed_token_ids", None),
        ("bad_words", None),
        ("_bad_words_token_ids", None),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
    )

    def add_request(self, request: Request) -> None:
        if self._is_block_output_model:
            existing = self.requests.get(request.request_id)
            if existing is not None and existing.streaming_queue is None:
                # A continuation (next input chunk or the closing sentinel) of
                # a session whose resumable flag the neutralization below
                # scrubbed. The base scheduler asserts on the missing
                # streaming_queue, which would tear down EngineCore. The
                # session cannot accept more input, so drop the message; the
                # live request finishes and notifies its client on its own.
                logger.warning(
                    "Dropping streaming-input continuation for request %s: "
                    "block-output models do not support resumable sessions",
                    request.request_id,
                )
                return
            self._truncate_unservable_block_prompt(request)
            self._align_block_output_max_tokens(request)
            self._neutralize_block_output_host_sampling(request)
        self._raw_prompt_reasoning_state(request)
        if getattr(self, "_mixed_steps", False) and self._uses_batch_indexed_sampling(
            request
        ):
            self._mixed_ineligible_req_ids.add(request.request_id)
        super().add_request(request)

    def _raw_prompt_reasoning_state(self, request: Request) -> None:
        """A structured request from /v1/completions under --reasoning-parser
        is constrained from its first token when its prompt opens no reasoning
        section.

        The chat frontend hands the engine the template's kwargs
        (reasoning_parser_kwargs, e.g. enable_thinking); the completions
        frontend hands nothing, so the reasoner is built thinking-on and the
        structured-output gate waits for a reasoning-end marker that a raw
        prompt never contains. The grammar then never engages and the request
        answers HTTP 200 with unconstrained text (review round 5 P2-4: the
        choice [red, blue] answered '\\n它的999'). No template opened a
        reasoning section in a raw prompt that holds neither marker, so its
        continuation is the constrained answer, as without a reasoning
        parser. A raw prompt that carries a marker keeps vLLM's own check (the
        last marker decides); chat requests are untouched.
        """
        so = request.structured_output_request
        if so is None or so.reasoning_parser_kwargs is not None or so.reasoning_ended is not None:
            return
        mgr = getattr(self, "structured_output_manager", None)
        if mgr is None or getattr(mgr, "reasoner_cls", None) is None or getattr(mgr, "enable_in_reasoning", False):
            return
        markers = self._reasoning_marker_ids(mgr, request)
        if not markers:
            return  # unknown markers: vLLM's default stands
        prompt = request.prompt_token_ids or []
        if not any(t in markers for t in prompt):
            so.reasoning_ended = True

    @staticmethod
    def _reasoning_marker_ids(mgr, request: Request) -> set[int]:
        """the reasoner's start / end marker token ids (empty when unknown)"""
        try:
            reasoner = mgr._get_reasoner(request)
        except Exception:  # a parser that cannot be built here: leave the request to vLLM
            logger.debug("reasoner build failed for %s", request.request_id, exc_info=True)
            return set()
        eng = getattr(reasoner, "_parser_engine", reasoner)
        ids = {getattr(eng, "_reasoning_start_token_id", None), getattr(eng, "_reasoning_end_token_id", None)}
        ids.discard(None)
        if not ids:
            vocab = getattr(mgr, "tokenizer", None)
            vocab = vocab.get_vocab() if vocab is not None else {}
            ids = {vocab[m] for m in ("<think>", "</think>") if m in vocab}
        return {int(i) for i in ids}

    def _truncate_unservable_block_prompt(self, request: Request) -> None:
        """Contain a bypassed prompt that leaves no room for a whole canvas.

        Frontend validation rejects such prompts; a prebuilt EngineCoreRequest
        skips it, and neither upstream EngineCore.add_request nor the base
        scheduler re-checks prompt length. Admitted untouched, the prompt
        either can never be scheduled (chunked prefill is disabled and the
        prompt exceeds the token budget: parked in WAITING forever,
        head-of-line blocking every later request), overflows the worker's
        max_model_len-wide token buffer, or — even when it fits
        max_model_len — trips the adapter's own canvas-capacity validation,
        which raises out of execute_model in eager (no-upfront-capture) mode
        where there is no graceful stop-canvas rejection. Truncate to the
        largest tile-aligned prompt that still fits one whole canvas below
        the tile-floored max_model_len, so every admitted request is
        genuinely servable and finishes through the normal notifying path.
        Prefix caching is disabled for block models (__init__ asserts it), so
        the request's stale block hashes stay inert.
        """
        from vllm_tt_plugin.platform import _TT_TOKEN_TILE_SIZE

        if request.mm_features:
            # A text-only block model has a zero encoder budget: a feature at
            # offset 0 forces zero-token schedules forever (head-of-line
            # stall), and an interior offset carves a partial prefill chunk
            # that flips the step onto host sampling and kills the engine.
            # Dropped, the placeholder positions decode as ordinary tokens.
            logger.warning(
                "Request %s bypassed frontend validation with multimodal "
                "features a block-output model cannot encode; dropping them",
                request.request_id,
            )
            request.mm_features = []
        if request.prompt_token_ids is None and request.num_prompt_tokens > 0:
            # The frontend rejects prompt_embeds for every TT model; admitted
            # bare, the worker's request-state builder raises
            # NotImplementedError out of execute_model. Replace with
            # placeholder tokens; an embeds-only Request already carries
            # [0] * num_prompt_tokens in _all_token_ids, so the replacement
            # keeps every derived view consistent.
            logger.warning(
                "Request %s bypassed frontend validation with a "
                "prompt_embeds-only prompt the TT backend does not support; "
                "replacing with %d placeholder tokens",
                request.request_id,
                request.num_prompt_tokens,
            )
            request.prompt_token_ids = [0] * request.num_prompt_tokens
            request.prompt_embeds = None
        if request.prompt_embeds is not None:
            # Mixed-mode prompts (token ids + embeds + prompt_is_token_ids
            # mask) carry placeholder ids at the embed positions, so the TT
            # model decodes them as ordinary tokens either way; scrub the
            # embeds so the prompt_len x hidden_size tensor is not pinned for
            # the request lifetime, and warn like the embeds-only branch.
            logger.warning(
                "Request %s bypassed frontend validation with mixed "
                "token/embeds prompt content the TT backend does not "
                "support; dropping the embeds (placeholder ids decode as "
                "ordinary tokens)",
                request.request_id,
            )
            request.prompt_embeds = None
            request.prompt_is_token_ids = None
        if request.num_prompt_tokens == 0:
            # The frontend rejects empty prompts; admitted bare, the waiting
            # loop schedules zero new tokens and upstream's num_new_tokens
            # assert tears down the engine. Pad to one placeholder token so
            # the request schedules and finishes through the normal path.
            logger.warning(
                "Request %s bypassed frontend validation with an empty "
                "prompt; padding to one placeholder token",
                request.request_id,
            )
            request.prompt_token_ids = [0]
            request.prompt_embeds = None
            request._all_token_ids.append(0)
            request.num_prompt_tokens = 1
            return
        tile = _TT_TOKEN_TILE_SIZE
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        aligned_max_model_len = max_model_len // tile * tile
        keep = (aligned_max_model_len - self._output_tokens_per_step) // tile * tile
        if request.num_prompt_tokens <= keep:
            return
        logger.warning(
            "Request %s bypassed frontend validation with a %d-token prompt "
            "that leaves no room for a whole %d-token canvas within "
            "max_model_len; truncating to %d tokens so the request can "
            "finish length-capped",
            request.request_id,
            request.num_prompt_tokens,
            self._output_tokens_per_step,
            keep,
        )
        if request.prompt_token_ids is not None:
            request.prompt_token_ids = request.prompt_token_ids[:keep]
        if getattr(request, "prompt_embeds", None) is not None:
            request.prompt_embeds = request.prompt_embeds[:keep]
        del request._all_token_ids[keep:]
        request.num_prompt_tokens = keep

    def _align_block_output_max_tokens(self, request: Request) -> None:
        """Clamp max_tokens so a bypassed EngineCoreRequest cannot overshoot.

        Frontend validation rejects an oversized limit. Prebuilt requests skip
        that path; the last canvas would then be applied past max_model_len and
        kill the engine. Shrink the logical cap to the largest whole-canvas
        budget that still fits. Lane mode reaches this through each lane.
        """
        from vllm_tt_plugin.platform import _fit_block_output_max_tokens

        prompt_ids = request.prompt_token_ids
        if prompt_ids is None or request.sampling_params is None:
            return
        fitted = _fit_block_output_max_tokens(
            len(prompt_ids),
            request.max_tokens,
            self._output_tokens_per_step,
            int(self.vllm_config.model_config.max_model_len),
        )
        if fitted == request.max_tokens:
            return
        # The prompt truncation above guarantees at least one whole canvas
        # fits, so a clamp always leaves a positive, servable budget.
        logger.debug(
            "Clamping block-output max_tokens from %s to %s for request %s "
            "so physical canvases fit max_model_len",
            request.max_tokens,
            fitted,
            request.request_id,
        )
        request.max_tokens = fitted
        request.sampling_params.max_tokens = fitted

    def _neutralize_block_output_host_sampling(self, request: Request) -> None:
        """Strip controls a block-output model cannot honor from a bypassed
        request: host-sampling forcers, structured outputs, and resumable
        streaming-input sessions.

        Frontend validation rejects these; a prebuilt EngineCoreRequest skips
        it, and any of them flips the step to host sampling mid-flight (which
        cannot construct a multi-token canvas) or parks the request forever.
        Raising here is no safer: an add_request exception also tears down
        EngineCore, so neutralize instead, the way the sampling controls are
        neutralized at the frontend.
        """
        params = request.sampling_params
        if params is None:
            return
        stripped = []
        if request.structured_output_request is not None:
            request.structured_output_request = None
            stripped.append("structured_outputs")
            # Request.__init__ parked the request on grammar compilation; with
            # the structured-output request gone nothing would ever promote it
            # out of skipped_waiting, so restore schedulability.
            if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
                request.status = RequestStatus.WAITING
        if params.structured_outputs is not None:
            params.structured_outputs = None
        if request.resumable:
            # A resumable session parks the stopped request to wait for more
            # input instead of finishing it, permanently leaking the
            # model-owned state slot. The frontend rejects it for block models.
            request.resumable = False
            stripped.append("resumable")
        for field, neutral in self._BLOCK_HOST_ONLY_SAMPLING_NEUTRAL:
            if getattr(params, field, neutral) not in (neutral, [], {}):
                setattr(params, field, neutral)
                stripped.append(field.lstrip("_"))
        if stripped:
            logger.warning(
                "Request %s bypassed frontend validation; stripped "
                "controls unsupported by block-output models: %s",
                request.request_id,
                ", ".join(stripped),
            )

    def _interleave_tail_exempt(self) -> bool:
        """GLM53_INTERLEAVE_TAIL_EXEMPT: True when the pending prefill is one running partial prefill whose remaining
        prompt tokens -- its last chunk -- are at most the knob's bound (interleave_tail_exempt_tokens), and nothing
        is waiting for admission (a waiting prompt's first chunk is not such a tail)"""
        lim = interleave_tail_exempt_tokens()
        if not lim or self.waiting or getattr(self, "skipped_waiting", False):
            return False
        parts = [r for r in self.running if getattr(r, "is_prefill_chunk", False)]
        if len(parts) != 1:
            return False
        n, done = getattr(parts[0], "num_prompt_tokens", None), getattr(parts[0], "num_computed_tokens", None)
        if n is None or done is None:
            return False
        return 0 < int(n) - int(done) <= lim

    def _has_pending_prefill(self) -> bool:
        """Whether any request still needs prefill work.

        A request in ``skipped_waiting`` still needs a future prefill pass:
        that is where the base scheduler retries promotion after its dependency
        becomes ready. In decode-only mode this check also ensures both waiting
        queues are hidden from the base scheduler.

        A running ``is_prefill_chunk`` request is a partial prefill whose next
        chunk can only be scheduled by a prefill step.
        """
        return (
            bool(self.waiting)
            or bool(getattr(self, "skipped_waiting", False))
            or any(request.is_prefill_chunk for request in self.running)
        )

    def _take_preempted_requests_with_pending_outputs(self) -> RequestQueue | None:
        """Temporarily remove resumes that still own an in-flight output.

        Ordinary preemption preserves already-submitted frames. Waiting for the
        corresponding placeholder to be consumed makes the accepted token part
        of request history before resumed prefill is built, so replay samples
        the next logical token and receives a fresh placeholder.

        The temporary queue follows upstream's skipped-waiting convention:
        ``prepend_request`` while collecting and ``prepend_requests`` while
        restoring preserve FCFS order, while priority queues reorder by their
        normal priority key.
        """
        deferred = [
            request
            for request in self.waiting
            if request.status == RequestStatus.PREEMPTED
            and request.num_output_placeholders > 0
        ]
        if not deferred:
            return None

        deferred_queue = create_request_queue(self.policy)
        for request in deferred:
            deferred_queue.prepend_request(request)
        self.waiting.remove_requests(deferred)
        return deferred_queue

    # vLLM runs ``_mamba_block_aligned_split`` on every prefill chunk when
    # ``need_mamba_block_aligned_split`` is set; the TT scheduler reuses that
    # hook so chunk ends can be aligned before any block is allocated, and so a
    # mixed step's prefill rows can be held to their share of the budget.
    @property
    def need_mamba_block_aligned_split(self) -> bool:
        return (
            bool(getattr(self, "_base_need_mamba_split", False))
            or getattr(self, "_prefill_chunk_alignment", 0) > 1
            or getattr(self, "_kda_checkpoint_store", None) is not None
            or getattr(self, "_prefix_oracle", None) is not None
            or getattr(self, "_mixed_prefill_room", None) is not None
            or os.environ.get("GLM53_ADMIT_KV_BUDGET", "0") == "1"  # the admission gate lives in this hook too
        )

    @need_mamba_block_aligned_split.setter
    def need_mamba_block_aligned_split(self, value: bool) -> None:
        self._base_need_mamba_split = bool(value)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        # Mixed step: every request that is not one of the step's decodes is
        # prefill work and shares the prefill room. Clamped before the
        # alignment so a chunk the room cuts short is still rounded onto the
        # grid; charged with what the request is handed. The charge is never
        # given back: a request that ends up unscheduled after all (its block
        # allocation failed, or a priority preemption took it back) only
        # shrinks the room, so the cap holds either way.
        room = getattr(self, "_mixed_prefill_room", None)
        mixed_prefill = (
            room is not None and request.request_id not in self._mixed_decode_req_ids
        )
        if mixed_prefill:
            num_new_tokens = min(num_new_tokens, room)
        if getattr(self, "_base_need_mamba_split", False):
            num_new_tokens = super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        n = self._align_prefill_chunk_end(request, start, num_new_tokens)
        trace = getattr(self, "_split_trace", None)  # v3b: what the hook saw this pass (the stuck-step line)
        if trace is not None and len(trace) < 12:
            trace.append([getattr(request, "request_id", None), int(num_new_tokens), int(n), "ok"])
        if n > 0 and self._kv_budget_defers(request):
            if trace is not None and trace and trace[-1][0] == getattr(request, "request_id", None):
                trace[-1][3] = "kv_gate"
            return 0  # GLM53_ADMIT_KV_BUDGET: stays waiting; vLLM's waiting loop stops here, so nothing overtakes it
        if n > 0 and self._prefill_one_request() and start < request.num_prompt_tokens:
            # GLM53_PREFILL_ONE_REQUEST=1: the first request given prompt tokens owns this step's prefill; another
            # request's prompt chunk waits for a later step (the leftover budget stays unused). A multi-request prefill
            # step lets two slots' state writes share a 32-user tile inside one program -- the run-to-run race of the
            # concurrent 1M cell (2026-10-24). Decode rows are untouched.
            rid = getattr(request, "request_id", None)
            owner = getattr(self, "_step_prefill_owner", None)
            if owner is not None and owner != rid:
                self.prefill_one_request_deferred = getattr(self, "prefill_one_request_deferred", 0) + 1
                if trace is not None and trace and trace[-1][0] == rid:
                    trace[-1][3] = "one_request"
                return 0
            self._step_prefill_owner = rid
        if n > 0 and getattr(self, "_segment_pass", False) and (room is None or mixed_prefill):
            n = self._segment_decide(request, start, n)
        if mixed_prefill:
            self._mixed_prefill_room = room - n
        return n

    # -- segmented prefill traces (doc/ttnn/14b-persistent-stack/segtrace/runner_design.md section 4, items 7-9) ----
    #
    # Step 0 schedules a segmentable chunk [s, e) minus its last token, so the request stays is_prefill_chunk (its
    # output is [] and the decode filters hide it); the runner runs [s, e)'s segment 0. Then, while the chunk is held,
    # every step is decode-only and carries the next segment (D - 1 plain decode steps between two segments, D =
    # prefill_segment_decodes). The release is the request's 1-token tail [e - 1, e) with nothing else prefilling;
    # the runner runs every segment not run yet + the head, and reports the token of a final chunk. From there the
    # request is an ordinary decoder or partial prefill. No decoders at step 0 -> no segmentation; decoders gone
    # inside the hold -> an early release. Spec configs (no mixed steps) segment through the prefill-only step:
    # steps 0 and the release are plain prefill steps.

    def _segment_pass_open(self, decoders: bool) -> None:
        """a scheduling pass in which a segmentable chunk may start: decoders run, no chunk is held, S > 1"""
        self._segment_pass = (bool(decoders) and getattr(self, "_segment_hold", None) is None
                              and self._prefill_segments() > 1)  # fmt: skip
        self._segment_start, self._segment_pass_rows = None, 0

    def _segment_pass_close(self) -> dict | None:
        start, self._segment_start, self._segment_pass = getattr(self, "_segment_start", None), None, False
        return start

    def _segment_decide(self, request: Request, start: int, n: int) -> int:
        """the split hook's last rule in a segment pass: the pass's FIRST prefill row is segmented when its chunk is
        long enough (the model's 2,048-bucket keys, GLM53_PREFILL_SEGMENT_MIN_TOKENS) and its last position needs no
        KV block beyond the chunk minus one token's allocation; it gets n - 1 tokens, and every later prefill row of
        the pass gets 0 (a segmented chunk is the step's only prefill: one open chunk, its shared scratch)"""
        self._segment_pass_rows = getattr(self, "_segment_pass_rows", 0) + 1
        if getattr(self, "_segment_start", None) is not None:
            return 0
        if self._segment_pass_rows > 1:
            return n
        end = start + n
        if n < max(2, int(getattr(self, "_segment_min_tokens", 513))):
            return n
        if getattr(request, "mm_features", None) or getattr(request, "has_encoder_inputs", False):
            return n
        if (end - 1) % int(self.cache_config.block_size) == 0:
            # position e - 1 opens a block the n - 1 tokens did not allocate (a prompt end one past a page edge)
            self.segment_skipped_block_edge = getattr(self, "segment_skipped_block_edge", 0) + 1
            return n
        self._segment_start = {"req_id": request.request_id, "S": self._prefill_segments(), "start": start,
                               "end": end, "n": n - 1}  # fmt: skip
        return n - 1

    def _segment_begin(self, result: SchedulerOutput, start: dict | None, *, mixed: bool) -> None:
        """after a segment pass: the chunk it segmented was scheduled -> stamp step 0, hold the chunk"""
        if start is None:
            return
        rid = start["req_id"]
        if int(result.num_scheduled_tokens.get(rid, 0)) != int(start["n"]):
            return  # not scheduled after all (its blocks): no token of the chunk ran, it is decided again later
        d = self._prefill_segment_decodes()
        self._segment_hold = {"req_id": rid, "S": int(start["S"]), "next": 1, "end": int(start["end"]), "D": d,
                              "d_left": d - 1, "mixed": bool(mixed)}  # fmt: skip
        set_tt_segmented(result, {rid: int(start["S"])})
        set_tt_segment_step(result, rid, [0], int(start["S"]), int(start["end"]))
        self.segment_chunks = getattr(self, "segment_chunks", 0) + 1
        self._mixed_follow_left = 0  # the hold's decode steps replace the follow-ups

    def _schedule_segment_hold(self, has_running_decode: bool) -> SchedulerOutput | None:
        """the step while a segmented chunk is held: decode-only carrying the next segment (or one of the D - 1 plain
        decode steps between two segments), then the release; None = the hold broke (the request left), schedule
        as usual"""
        hold = self._segment_hold
        assert hold is not None
        rid = hold["req_id"]
        request = self.requests.get(rid)
        if request is None or request not in self.running or not request.is_prefill_chunk:
            # aborted / finished / preempted inside the hold: the runner drops its open chunk through the step's
            # finished / preempted ids, the connector its parked save
            self._segment_hold = None
            self.segment_holds_broken = getattr(self, "segment_holds_broken", 0) + 1
            logger.warning("TT scheduler: segmented chunk of %s left inside its hold (segment %d of %d next)", rid,
                           hold["next"], hold["S"])  # fmt: skip
            return None
        self._decode_interleave.note_mixed_step()  # every step of a segmented chunk advances the decoders
        self._mixed_follow_left = 0
        empty = None
        if has_running_decode and (hold["next"] < hold["S"] - 1 or hold["d_left"] > 0):
            result = self._schedule_decode_only()
            if result.total_num_scheduled_tokens > 0:
                if hold["d_left"] > 0:
                    hold["d_left"] -= 1  # D > 1: a plain decode step between two segments
                    self.segment_decode_steps = getattr(self, "segment_decode_steps", 0) + 1
                else:
                    set_tt_segment_step(result, rid, [hold["next"]], hold["S"], hold["end"])
                    hold["next"] += 1
                    hold["d_left"] = hold["D"] - 1
                    self.segment_steps = getattr(self, "segment_steps", 0) + 1
                return self._finalize_scheduler_output(result, is_decode=True)
            empty = result  # the decoders scheduled nothing (async placeholders at max_tokens): release now
        return self._schedule_segment_release(empty)

    def _schedule_segment_release(self, empty: SchedulerOutput | None) -> SchedulerOutput:
        """the held request's 1-token tail, with the waiting queues and every other partial prefill hidden; mixed
        with the decoders when the chunk started in a mixed step and decoders run, else a prefill-only step"""
        hold = self._segment_hold
        assert hold is not None
        rid, n_seg = hold["req_id"], hold["S"]
        segments = list(range(hold["next"], n_seg))
        others = [r for r in self.running if r.is_prefill_chunk and r.request_id != rid]
        saved_waiting = self.waiting
        saved_skipped = getattr(self, "skipped_waiting", None)
        self.waiting = create_request_queue(self.policy)
        if saved_skipped is not None:
            self.skipped_waiting = create_request_queue(self.policy)
        if others:
            hidden = {id(r) for r in others}
            self.running = [r for r in self.running if id(r) not in hidden]
        try:
            if hold["mixed"] and any(not r.is_prefill_chunk for r in self.running):
                result = self._schedule_mixed(1)
            else:
                result = self._finalize_scheduler_output(self._schedule_prefill_only(budget=1), is_decode=False)
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            if saved_skipped is not None:
                if self.skipped_waiting:
                    saved_skipped.prepend_requests(self.skipped_waiting)
                self.skipped_waiting = saved_skipped
            self.waiting = saved_waiting
            if others:
                self.running.extend(others)
        if empty is not None:  # an empty decode pass ran first this step: its cleanup events go out with this one
            result.finished_req_ids |= empty.finished_req_ids
            result.free_encoder_mm_hashes = empty.free_encoder_mm_hashes + result.free_encoder_mm_hashes
            if empty.preempted_req_ids:
                result.preempted_req_ids = (result.preempted_req_ids or set()) | empty.preempted_req_ids
        if int(result.num_scheduled_tokens.get(rid, 0)) == 1:
            set_tt_segment_step(result, rid, segments, n_seg, hold["end"])
            if segments != [n_seg - 1]:
                self.segment_early_releases = getattr(self, "segment_early_releases", 0) + 1
            self.segment_steps = getattr(self, "segment_steps", 0) + 1
            self._segment_hold = None
            # the last segment's D decodes: the release's own (mixed) + D - 1, or D after a prefill-only release
            self._segment_follow_left = hold["D"] - 1 if hold["mixed"] else hold["D"]
        else:
            self.segment_release_retries = getattr(self, "segment_release_retries", 0) + 1  # held: the next step
        return result

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """GLM53_PREEMPT_SAVE (program 2026-10-27T00:05:27Z (2); kv_connector._PREEMPT_SAVES): record the request's
        computed KV (n tokens, its block ids, its token ids) BEFORE vLLM frees the blocks, so the worker's begin_step
        stores it (and an exact KDA checkpoint) before this step's forward reuses them; the resume then hits n instead of
        re-prefilling its own generated tokens (the 8/16 x 32K divergence)."""
        from vllm_tt_plugin import kv_connector as _kvc  # noqa: PLC0415

        if _kvc._preempt_save_on() and getattr(self, "connector", None) is not None:
            # 14 B2: under async scheduling num_computed_tokens counts the in-flight step's scheduled drafts (vLLM
            # subtracts rejections only in update_from_output): never past the committed history
            n = min(int(request.num_computed_tokens), len(request.all_token_ids))
            try:
                blocks = self.kv_cache_manager.get_block_ids(request.request_id)
            except Exception:  # noqa: BLE001 - no blocks known: nothing to save (vLLM preempts as before)
                blocks = None
            if blocks and _kvc.record_preempt_save(request.request_id, n, blocks, request.all_token_ids):
                self.preempt_saves_recorded = getattr(self, "preempt_saves_recorded", 0) + 1
        super()._preempt_request(request, timestamp)

    def _kv_budget_defers(self, request: Request) -> bool:
        """GLM53_ADMIT_KV_BUDGET=1 (program 2026-10-27T00:05:27Z (3); default off): a WAITING request is admitted
        only while the KV the resident requests may grow to fits the pool, so decode growth never preempts. Today vLLM
        admits on the prompt's blocks alone and preempts later (the 8/16 x 32K cells' recompute divergence). The
        reserve of a request is blocks(min(prompt + min(max_tokens, GLM53_ADMIT_RESERVE_TOKENS), max_model_len) + the
        scheduler's lookahead (the speculative K / K + 1 every step allocates)). The cap (default 32,768; 0 = none):
        an OpenAI request without max_tokens reaches the scheduler with max_tokens = max_model_len - prompt, which
        would reserve the whole pool and serve such clients one at a time; a request that outgrows its capped
        reserve is preempted (GLM53_PREEMPT_SAVE is the net). Resident = the RUNNING and WAITING_FOR_REMOTE_KVS
        requests by status (both hold blocks; a preempted request holds none and waits at the head). A deferred
        request stays waiting, and vLLM's waiting loop stops at it (no smaller request overtakes it). Nothing resident
        = always admit (the budget cannot help, and the request must not wait forever). Counters
        admissions_deferred_kv, admit_reserve_capped (requests whose max_tokens the cap cut, once each); one log line
        per deferred request and per capped request. The gate runs in vLLM's split hook, which vLLM skips for an
        async KV load (load_kv_async): the TT connector never reports one (LMCacheConnectorV1 returns is_async False;
        TTPageAlignedConnector warns if it ever sees True with the gate on)."""
        if self._kv_fits(request):  # v3b: ONE predicate for the gate, the SHORT_FIRST hold and the overtake
            return False
        used, need, total, resident = self._kv_budget_room(request)
        self.admissions_deferred_kv = getattr(self, "admissions_deferred_kv", 0) + 1
        seen = self.__dict__.setdefault("_kv_budget_logged", set())
        rid = getattr(request, "request_id", None)
        if rid not in seen:
            seen.add(rid)
            if len(seen) > 4096:
                seen.clear()
            logger.info("TT scheduler: %s waits for KV (GLM53_ADMIT_KV_BUDGET): reserves %d blocks, %d of %d reserved "
                        "by %d resident", rid, need, used, total, resident)  # fmt: skip
        return True

    def _kv_reserve_blocks(self, r: Any) -> int:
        """GLM53_ADMIT_KV_BUDGET's reserve of one request: blocks(min(prompt + min(max_tokens, GLM53_ADMIT_RESERVE_TOKENS),
        max_model_len) + the scheduler's lookahead); a capped request is counted / logged once"""
        bs = int(self.cache_config.block_size)
        mml = int(self.max_model_len)
        cap = _admit_reserve_tokens()
        look = int(getattr(self, "num_lookahead_tokens", 0) or 0)
        mx = getattr(r, "max_tokens", None)
        out = int(mx) if mx is not None else mml
        if cap > 0 and out > cap:
            out = cap
            capped = self.__dict__.setdefault("_kv_budget_capped", set())
            rid_ = getattr(r, "request_id", None)
            if rid_ not in capped:
                capped.add(rid_)
                if len(capped) > 4096:
                    capped.clear()
                self.admit_reserve_capped = getattr(self, "admit_reserve_capped", 0) + 1
                logger.info("TT scheduler: %s reserves %d output tokens of its max_tokens %s "
                            "(GLM53_ADMIT_RESERVE_TOKENS)", rid_, cap, mx)  # fmt: skip
        return -(-(min(int(r.num_prompt_tokens) + out, mml) + look) // bs)

    def _kv_budget_room(self, request: Any = None, long_tokens: int = 0) -> tuple[int, int, int, int]:
        """(used, need, total, residents) in blocks: the reserves of the resident requests by STATUS (a prefill-only
        step schedules with self.running emptied of its decoders, so the list would read 0 there; long_tokens > 0:
        only the residents whose prompt is longer than it), `request`'s own reserve (0 without one), the pool's
        blocks less the null block"""
        resident = [r for r in self.requests.values() if r.status in _KV_RESIDENT_STATUSES
                    and (long_tokens <= 0 or int(r.num_prompt_tokens) > long_tokens)]  # fmt: skip
        total = int(self.kv_cache_manager.block_pool.num_gpu_blocks) - 1  # (the null block)
        used = sum(self._kv_reserve_blocks(r) for r in resident)
        need = self._kv_reserve_blocks(request) if request is not None and resident else 0
        return used, need, total, len(resident)

    def _prefill_one_request(self) -> bool:
        """GLM53_PREFILL_ONE_REQUEST (launch env, default off): at most one request's prompt chunk per step"""
        v = getattr(self, "_prefill_one_request_flag", None)
        if v is None:
            v = os.environ.get("GLM53_PREFILL_ONE_REQUEST", "0") == "1"
            self._prefill_one_request_flag = v
            if v:
                logger.info("TT scheduler: one request's prompt chunk per prefill step (GLM53_PREFILL_ONE_REQUEST)")
        return v

    def _plannable_prefix_hit(self, request: Request, oracle, ids, keep: int, block_size: int) -> int:
        """program 2026-10-26T17:23:42Z (M2-1): a DEVICE prefix hit's continuation [keep, prompt end) must PLAN on the
        model (vllm_tt_plugin.plan_check), as the connector's LMCache hits do -- else it steps down to the next lower
        snapshotted boundary (the oracle's aligned_hit) until it plans, or to no hit (a fresh prefill from 0)"""
        from vllm_tt_plugin import plan_check  # noqa: PLC0415
        from vllm_tt_plugin.prefix_oracle import aligned_hit  # noqa: PLC0415

        end = int(getattr(request, "num_prompt_tokens", 0) or len(ids))
        while keep > 0:
            err = plan_check.plan_error(keep, end - keep)
            if err is None:
                break
            lower = int(aligned_hit(oracle, ids, keep - 1, block_size))
            logger.info("prefix oracle: %s device hit at %d does not plan (%s); stepping down to %d",
                        request.request_id, keep, err, lower)  # fmt: skip
            self.__dict__["_prefix_hits_replanned"] = self.__dict__.get("_prefix_hits_replanned", 0) + 1
            if lower >= keep:
                break
            keep = lower
        return keep

    def _install_prefix_oracle(self, oracle) -> None:
        """Wrap the KV cache manager's local hit: computed blocks and tokens are
        cut to the longest boundary the model snapshotted for this prompt."""
        from vllm_tt_plugin.prefix_oracle import aligned_hit

        manager = self.kv_cache_manager
        inner = manager.get_computed_blocks
        block_size = int(self.cache_config.block_size)

        def get_computed_blocks(request: Request):
            result = inner(request)
            blocks, num_computed = result[0], int(result[1])
            if num_computed <= 0:
                return result
            ids = request.prompt_token_ids or request.all_token_ids
            keep = aligned_hit(oracle, ids, num_computed, block_size)
            keep = self._plannable_prefix_hit(request, oracle, ids, keep, block_size)
            if keep == num_computed:
                return result
            n_blocks = keep // block_size
            cut = manager.create_kv_cache_blocks(
                tuple(list(group[:n_blocks]) for group in blocks.blocks)
            )
            if keep != num_computed:
                logger.debug(
                    "prefix oracle: %s page hit %d -> %d tokens (snapshotted boundary)",
                    request.request_id,
                    num_computed,
                    keep,
                )
            self._prefix_hits_truncated = getattr(self, "_prefix_hits_truncated", 0) + 1
            return (cut, keep) + tuple(result[2:])

        manager.get_computed_blocks = get_computed_blocks  # type: ignore[method-assign]

    def _free_request(self, request: Request, *args, **kwargs):
        """A finished request: the model's prefix cache keys its trailing state
        snapshot by the request's ids now (``note_finished``), before the runner's
        release reaches the model one step later, so a chat turn arriving right
        after the previous one hits that snapshot. The base's other arguments
        (vLLM 0.26 ``finish_requests`` passes ``delay_free_blocks=``) pass through."""
        oracle = getattr(self, "_prefix_oracle", None)
        note = getattr(oracle, "note_finished", None) if oracle is not None else None
        if callable(note):
            try:
                note(list(request.all_token_ids))
            except Exception as exc:  # bookkeeping only: never fail the engine step
                logger.warning("prefix oracle note_finished failed for %s: %s", request.request_id, exc)
        ineligible = getattr(self, "_mixed_ineligible_req_ids", None)
        if ineligible:
            ineligible.discard(request.request_id)
        return super()._free_request(request, *args, **kwargs)

    def _align_prefill_chunk_end(
        self, request: Request, start: int, num_new_tokens: int
    ) -> int:
        """Round a *partial* prefill chunk down so it ends on the alignment grid.

        Chunks that finish the prompt, decode steps and multimodal prompts are
        left alone. A request whose aligned chunk would be empty -- new, running
        (a continuation after an LMCache / prefix hit or an earlier chunk),
        resumed or preempted alike -- waits for the next step (vLLM skips a
        request handed 0 tokens), unless no full step could reach the grid. The
        model starts every continuation on the grid: a running request given
        this step's leftover (e.g. a 640-token hit, then a 9-token chunk to 649
        with 32 live rows) would make the next chunk start off the 64-token
        latent page, and the TT step raises.
        """
        # The dev hints (tt.plan_cut / tt.chunk_cap, client kv_transfer_params) act only on the alignment grid: an
        # off-grid cut or cap leaves no aligned chunk, so the request would be handed 0 tokens on every step (a running
        # one never progresses; a waiting one blocks every admission behind it)
        grid = max(int(getattr(self, "_prefill_chunk_alignment", 0) or 0), 1)
        cut = _plan_cut_hint(request) if num_new_tokens > 0 else 0
        cut = 0 if cut % grid else cut
        if cut and start < cut < start + num_new_tokens:
            num_new_tokens = cut - start  # tt.plan_cut (dev): no chunk crosses the hinted hit length
        cap = _chunk_cap_hint(request) if num_new_tokens > 0 else 0
        cap = 0 if cap % grid else cap
        if cap and num_new_tokens > cap:
            num_new_tokens = cap  # tt.chunk_cap (dev): the prefill runs in buckets <= cap
        rg = int(getattr(self, "_resume_grid", 0) or 0)  # plan-aligned resume cuts (see __init__)
        rg_end = (start // rg + 1) * rg if rg else 0
        # the cut must land on the continuation-alignment grid too: a resume grid off it (a step budget that is not a
        # multiple of prefill_chunk_alignment, e.g. 150 on 128) ended the chunk at a boundary the alignment below then
        # rounded back to the start -> 0 tokens on every step, a livelock. A resume grid on the alignment grid keeps the cut.
        align_grid = int(getattr(self, "_prefill_chunk_alignment", 0) or 0)
        if (rg and num_new_tokens > 0 and start % rg and start + num_new_tokens > rg_end
                and (align_grid <= 1 or rg_end % align_grid == 0)):  # fmt: skip
            num_new_tokens = rg_end - start
        oracle = getattr(self, "_prefix_oracle", None)
        skip = getattr(self, "_tail_split_skip", "0")
        stores = getattr(self, "_kda_checkpoint_store", None) and not (skip != "0" and _stores_nothing(request, skip))
        if num_new_tokens > 0 and (stores or oracle is not None):
            # KDA checkpoint policy: a chunk that would finish an unaligned
            # prompt stops on the last page boundary first (the state after
            # exactly that many tokens is what a later prefix hit restores);
            # the tail runs as the next chunk.
            # The boundary is the hit length a later request gets for this
            # prompt (connectors report len - 1 matched tokens, rounded down
            # to the page): floor(len - 1), not floor(len), so a prompt that is
            # itself a multiple of the page checkpoints at len - page (J7).
            # The grid is the page, at least the prefill continuation alignment
            # (_checkpoint_grid), and with the model's prefix oracle its snapshot
            # boundary (128 for GLM-5.3-Flash).
            bs = getattr(self, "_checkpoint_grid", 0) or self.cache_config.block_size
            if oracle is not None:
                bs = max(bs, int(oracle.boundary))
            boundary = max(request.num_prompt_tokens - 1, 0) // bs * bs
            # GLM53_KDA_CKPT_AT_END (kv_connector.kda_ckpt_at_end_mode): the prompt is checkpointed at its END, so its
            # last chunk runs whole ("aligned": a prompt that ends on the grid; "all": every prompt). "both" (the
            # default since cf3h-e) and "0" keep the split; under "both" the worker also checkpoints an aligned end.
            # The prefix oracle keeps its snapshot boundary.
            at_end = getattr(self, "_ckpt_at_end", "0")
            if oracle is None and (getattr(self, "_ckpt_in_replay", False) or at_end == "all"
                                   or (at_end == "aligned" and not request.num_prompt_tokens % bs)):  # fmt: skip
                boundary = 0  # GLM53_KDA_CKPT_IN_REPLAY: the replay snapshots the cut itself
            tmin = getattr(self, "_tail_split_min", 0)
            if oracle is None and tmin and request.num_prompt_tokens < tmin:
                boundary = 0  # GLM53_PF_TAIL_SPLIT_MIN: a short prompt runs its last chunk whole (see __init__)
            if (
                start + num_new_tokens >= request.num_prompt_tokens
                and start < boundary < request.num_prompt_tokens
            ):
                num_new_tokens = boundary - start
        align = getattr(self, "_prefill_chunk_alignment", 0)
        if align <= 1 or num_new_tokens <= 0:
            return num_new_tokens
        end = start + num_new_tokens
        # The chunk that reaches every known token is the last of this prefill.
        # A preempted request re-prefills its prompt AND the tokens it generated
        # (num_tokens), so its chunks past the prompt end still need the grid
        # (a request re-prefilling [0, 150) would otherwise start its next chunk
        # off the latent page).
        known = max(
            int(request.num_prompt_tokens), int(getattr(request, "num_tokens", 0) or 0)
        )
        if end >= known:
            return num_new_tokens
        if getattr(request, "mm_features", None) or getattr(
            request, "has_encoder_inputs", False
        ):
            return num_new_tokens
        aligned_end = end // align * align
        if aligned_end <= start:
            # Any request waits when this step's leftover budget is what stopped
            # it. If a full step could not reach the grid either (a permanent cap
            # below the alignment), it must still make progress.
            return (
                0 if num_new_tokens < self._full_step_prefill_cap() else num_new_tokens
            )
        return aligned_end - start

    def _full_step_prefill_cap(self) -> int:
        """Most prompt tokens one scheduling step can give a single new request."""
        cfg = getattr(self, "scheduler_config", None)
        if cfg is None:
            return sys.maxsize
        limit, _ = _effective_prefill_step_limit(cfg)
        return limit

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        t_tp = time.perf_counter() if ttft_phases.on() else None  # GLM53_TTFT_PHASES: the cold first chunks' stamp
        out = self._schedule_tt(throttle_prefills)
        if t_tp is not None:
            ttft_phases.mark_scheduled(self, out, t_tp)
        self._publish_admit_headroom()
        return out

    def _publish_admit_headroom(self, force: bool = False) -> None:
        """glm53_admit_{pool,reserved,headroom,queued}_tokens (_admit_gauges), at most every
        GLM53_ADMIT_HEADROOM_EVERY_S (default 0.5) s: the reserve rule of GLM53_ADMIT_KV_BUDGET, in tokens. force (v3b6):
        at once -- the engine just drained (_publish_admit_headroom_if_drained)"""
        g = _admit_gauges()
        if g is None:
            return
        now = time.monotonic()
        try:
            every = float(os.environ.get("GLM53_ADMIT_HEADROOM_EVERY_S", "0.5") or 0.5)
        except ValueError:
            every = 0.5
        if not force and now - getattr(self, "_admit_headroom_t", 0.0) < every:
            return
        self._admit_headroom_t = now
        try:
            bs = int(self.cache_config.block_size)
            used, _, total, _ = self._kv_budget_room()
            queued = sum(self._kv_reserve_blocks(r) for q in (self.skipped_waiting, self.waiting) for r in (q or ())
                         if r.status not in _KV_RESIDENT_STATUSES)  # fmt: skip
            g["pool"].set(total * bs)
            g["reserved"].set(used * bs)
            g["headroom"].set(max(0, total - used) * bs)
            g["queued"].set(queued * bs)
        except Exception as e:  # noqa: BLE001 -- a metric never fails a step
            if not getattr(self, "_admit_headroom_warned", False):
                self._admit_headroom_warned = True
                logger.warning("TT scheduler: admission headroom metric not published: %s", e)

    def _publish_admit_headroom_if_drained(self) -> None:
        """v3b6 (program 2026-10-29T01:17:33Z): the gauges are set from schedule(), and a drained engine schedules
        nothing -- they kept their last BUSY value (a 1M reserve) until the next request, so router v6-v6c read an EMPTY
        block as full: 16 shorts all on blk3 while blk0 / blk1 sat idle, two 1M returns 12+ min in the router. The step
        that empties the engine (update_from_output) and an abort that empties it (finish_requests) publish at once."""
        if _admit_gauges() is None:
            return
        try:
            if self.get_num_unfinished_requests() == 0:
                self._publish_admit_headroom(force=True)
        except Exception:  # noqa: BLE001 -- a metric never fails a step
            pass

    def finish_requests(self, *args, **kwargs):
        out = super().finish_requests(*args, **kwargs)
        self._publish_admit_headroom_if_drained()  # v3b6: an abort (client gone) that empties the engine
        return out

    def _schedule_tt(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._step_prefill_owner = None  # GLM53_PREFILL_ONE_REQUEST: a new step, no prefill owner yet
        self._refuse_unplannable()
        if getattr(self, "_yield_pending", False):
            self._yield_pending = False
            return self._schedule_first_token_yield()
        deferred_resumes = self._take_preempted_requests_with_pending_outputs()
        try:
            out = self._schedule_without_pending_output_resumes(throttle_prefills)
        finally:
            if deferred_resumes:
                self.waiting.prepend_requests(deferred_resumes)
        if getattr(self, "_first_token_yield", False):
            self._yield_pending = self._finishes_a_prompt(out)
        return out

    def _finishes_a_prompt(self, scheduler_output: SchedulerOutput) -> bool:
        """a request whose last prompt chunk runs in this step (its first output token comes from it); the scheduler
        has already advanced num_computed_tokens by the scheduled tokens"""
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None or int(getattr(request, "num_output_tokens", 0) or 0):
                continue
            after = int(request.num_computed_tokens)
            if after - int(n) < int(request.num_prompt_tokens) <= after:
                return True
        return False

    def _schedule_first_token_yield(self) -> SchedulerOutput:
        """GLM53_FIRST_TOKEN_YIELD: an EMPTY step (token budget 0), built by the base scheduler so finished request ids,
        connector metadata and the other per-step bookkeeping still go out with it"""
        saved_budget = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = 0
        try:
            result = super().schedule()
        finally:
            self.max_num_scheduled_tokens = saved_budget
        self.first_token_yields += 1
        return self._finalize_scheduler_output(result, is_decode=False)

    def _schedule_without_pending_output_resumes(
        self, throttle_prefills: bool = False
    ) -> SchedulerOutput:
        # NOTE: `throttle_prefills` accepted for interface compatibility with the base
        #        scheduler but unused - TT separates prefill/decode explicitly.
        self._apply_sched_ctl()  # GLM53_DECODE_INTERLEAVE_FILE: once per change
        has_pending_prefill = self._has_pending_prefill()
        # A partial prefill occupies ``running`` but cannot produce a decode
        # token, so it must not make the decode fallback below look viable.
        has_running_decode = any(
            not request.is_prefill_chunk for request in self.running
        )
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            # Forced mode is shared by every lane. Return an empty prefill
            # result unchanged so the coordinator can decide whether all lanes
            # should fall back to decode together.
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result, is_decode=False)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide the waiting queues and partial prefills so the base
                # scheduler cannot admit prefill work.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result, is_decode=True)

        if getattr(self, "_segment_hold", None) is not None:
            # a segmented chunk in flight owns the step loop until its release (runner_design.md section 4)
            held = self._schedule_segment_hold(has_running_decode)
            if held is not None:
                return held
            has_pending_prefill = self._has_pending_prefill()
        if getattr(self, "_segment_follow_left", 0) > 0:
            if has_running_decode and has_pending_prefill:
                self._segment_follow_left -= 1
                self.segment_decode_steps = getattr(self, "segment_decode_steps", 0) + 1
                self._decode_interleave.note_mixed_step()
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            self._segment_follow_left = 0  # nothing to interleave with: an ordinary step

        # Default mode:
        # Prefer prefill whenever prefill work is pending, so new requests are
        # admitted and partial prefills advance.
        if has_pending_prefill:
            # Mixed mode: the running decodes ride along with the prefill, so
            # neither the interleave policy nor the decode fallback is needed.
            follow = self._mixed_decode_follow() if getattr(self, "_mixed_steps", False) else 0
            if follow and has_running_decode and getattr(self, "_mixed_follow_left", 0) > 0:
                # GLM53_MIXED_DECODE_FOLLOW: one of the K decode-only steps after a mixed step (a decoder's token
                # every ~40 ms instead of every prefill chunk); the counter always falls, so the prefill resumes
                self._mixed_follow_left -= 1
                self.mixed_follow_steps = getattr(self, "mixed_follow_steps", 0) + 1
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            mixed_budget = self._mixed_prefill_budget() if has_running_decode else 0
            if mixed_budget:
                out = self._schedule_mixed(mixed_budget)
                self._mixed_follow_left = follow if is_tt_mixed_step(out) else 0
                if get_tt_segment_step(out) is not None:
                    self._mixed_follow_left = 0  # a segmented chunk's hold decodes instead
                return out
            wants = self._decode_interleave.wants_decode_step(
                has_pending_prefill=True, has_running_decode=has_running_decode
            )
            if wants and self._interleave_tail_exempt():
                wants = False  # GLM53_INTERLEAVE_TAIL_EXEMPT: the last short chunk joins its predecessor's prefill run
                self.interleave_tail_exempts = getattr(self, "interleave_tail_exempts", 0) + 1
            if wants:
                # A run of prefill steps has reached its bound. Spend this step
                # on the running decodes so their inter-token latency does not
                # scale with the pending prompt's length; the pending prefill
                # resumes on the next step.
                self._decode_interleave.record_step(
                    is_decode=True, prefill_pending=True
                )
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result, is_decode=True)
            prefill_result = (  # S > 1 with decoders running: the pass may start a segmented chunk
                self._schedule_prefill_only(segmentable=True)
                if has_running_decode and getattr(self, "_segment_hold", None) is None and self._prefill_segments() > 1
                else self._schedule_prefill_only()
            )
            # If prefill cannot make progress (e.g. KV pressure), do not stall
            # decode. Fall back to decode-only so running requests can advance
            # and free capacity for a later prefill admission.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running_decode:
                self._decode_interleave.record_step(
                    is_decode=True, prefill_pending=True
                )
                result = self._schedule_decode_only()
                # Even an empty prefill pass drains upstream cleanup events.
                # The runner must receive them with the replacement decode.
                result.finished_req_ids |= prefill_result.finished_req_ids
                result.free_encoder_mm_hashes = (
                    prefill_result.free_encoder_mm_hashes
                    + result.free_encoder_mm_hashes
                )
                if prefill_result.preempted_req_ids:
                    result.preempted_req_ids = (
                        result.preempted_req_ids or set()
                    ) | prefill_result.preempted_req_ids
                return self._finalize_scheduler_output(result, is_decode=True)
            self._decode_interleave.record_step(is_decode=False, prefill_pending=True)
            return self._finalize_scheduler_output(prefill_result, is_decode=False)

        # No pending prefill work in default mode: run decode-only naturally.
        self._mixed_follow_left = 0  # follow-ups serve decoders between a pending prefill's chunks; none is pending
        self._decode_interleave.record_step(is_decode=True, prefill_pending=False)
        result = super().schedule()
        return self._finalize_scheduler_output(result, is_decode=True)

    def _finalize_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
        *,
        is_decode: bool,
        decode_rows: int | None = None,
    ) -> SchedulerOutput:
        """``decode_rows``: the decode rows of a mixed step (default: every
        scheduled row of a decode step)."""
        if is_decode:
            rows = (
                len(scheduler_output.num_scheduled_tokens)
                if decode_rows is None
                else decode_rows
            )
            if rows > getattr(self, "_widest_decode_batch_size", 0):
                self._widest_decode_batch_size = rows
                logger.info(
                    "TT scheduler: widest decode batch reached %d request row(s)",
                    rows,
                )
        pending_reset_discards = getattr(
            self, "_pending_forced_reset_discard_counts", {}
        )
        if pending_reset_discards:
            set_tt_forced_reset_discard_counts(scheduler_output, pending_reset_discards)
            self._pending_forced_reset_discard_counts = {}
        return scheduler_output

    def _schedule_prefill_only(self, segmentable: bool = False, budget: int | None = None) -> SchedulerOutput:
        """Schedule prefill work: waiting requests and partial continuations.

        Temporarily hides the running *decode* requests so the base scheduler's
        running loop only advances partial prefills, and its waiting loop
        admits new ones.  Adjusts max_num_running_reqs so the waiting loop
        respects the true capacity with the decodes hidden.

        ``segmentable``: decoders run, so the pass may start a segmented chunk
        (``_segment_decide``). ``budget``: the step's token budget (a segment
        release: 1).
        """
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]
        # GLM53_PREFILL_SHORT_FIRST: a long mid-prompt prefill sits this step out when a short prompt can take it (v3b4:
        # the runner paces a long's head-less chunk runs at the device, prefill_pacer, so this runs once per device run)
        held = self._short_first_hold(partial_prefills) if budget is None else []
        if held:
            partial_prefills = [r for r in partial_prefills if r not in held]

        saved_max = self.max_num_running_reqs
        saved_budget = self.max_num_scheduled_tokens
        self.running = cast(list[Request], partial_prefills)
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes) - len(held))
        self.max_num_scheduled_tokens = (
            self._prefill_chunk_run_budget(partial_prefills, pure_decodes, short_first=bool(held))
            if budget is None else int(budget)
        )  # fmt: skip
        if segmentable:
            self._segment_pass_open(bool(pure_decodes))
        start = None
        self._split_trace = []
        hidden = self._short_first_kv_hide() if budget is None else None  # gate-deferred requests a short may pass
        try:
            result = super().schedule()
        finally:
            self._short_first_kv_restore(hidden)
            self.running[:0] = held  # (their earlier admission order: ahead of anything admitted this step)
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
            self.max_num_scheduled_tokens = saved_budget
            if segmentable:
                start = self._segment_pass_close()
        if held:
            self._short_first_after(result, held)
        self._segment_begin(result, start, mixed=False)
        return result

    # -- GLM53_PREFILL_SHORT_FIRST (program 2026-10-28T13:27:02Z / 13:32:43Z: DP4 load run 1, a block's prefill was
    # FIFO -- a 2K arrival behind an in-flight 1M waited the whole 1M, 255-325 s) ---------------------------------------
    def _short_first_hold(self, partial_prefills) -> list:
        """the long mid-prompt prefills (remaining > T = GLM53_PREFILL_SHORT_FIRST tokens) this prefill step holds back,
        so a short prompt (remaining <= T: a running partial or an admissible waiting request) owns the step instead
        (with GLM53_PREFILL_ONE_REQUEST the long would own every step until its last chunk). At most K =
        GLM53_PREFILL_SHORT_FIRST_MAX (default 4) held steps in a row, then the long gets one step (no starvation). A
        waiting short that a held step could not admit (a deferred lookup, the KV budget) is not held for again for
        GLM53_PREFILL_SHORT_FIRST_RETRY (default 8) steps. One request per prefill step stays the rule."""
        T = prefill_short_first_tokens()
        if T <= 0 or not partial_prefills:
            return []
        rem = lambda r: int(r.num_prompt_tokens) - int(r.num_computed_tokens)  # noqa: E731
        longs = [r for r in partial_prefills if rem(r) > T]
        if not longs:
            return []
        self._short_first_steps = getattr(self, "_short_first_steps", 0) + 1
        stuck = getattr(self, "_short_first_stuck", None)
        if stuck is None:
            stuck = self._short_first_stuck = {}
        for rid in [k for k, v in stuck.items() if self._short_first_steps - v >= prefill_short_first_retry()]:
            stuck.pop(rid, None)
        if not getattr(self, "_short_first_knobs_logged", False):  # v3b: the in-engine proof of the knobs
            self._short_first_knobs_logged = True
            logger.info("TT scheduler: SHORT_FIRST T=%d MAX=%d RETRY=%d INFLIGHT=%d KV_OVERTAKE=%s ADMIT_KV_BUDGET=%s "
                        "ONE_REQUEST=%s policy=%s", T, prefill_short_first_max(), prefill_short_first_retry(),
                        prefill_short_first_inflight(), prefill_short_first_kv_overtake(),
                        os.environ.get("GLM53_ADMIT_KV_BUDGET", "0"), os.environ.get("GLM53_PREFILL_ONE_REQUEST", "0"),
                        getattr(self, "policy", None))  # fmt: skip
        shorts = [r for r in partial_prefills if rem(r) <= T]
        shorts += [r for _, r in self._waiting_scan()  # v3b: BOTH queues (a lookup that returned 'not yet' parks a short
                   # in skipped_waiting, which vLLM scans first)
                   if rem(r) <= T and getattr(r, "request_id", None) not in stuck and self._kv_fits(r)]  # fmt: skip
        if not shorts:
            self._short_first_run = 0
            waiting = [r for _, r in self._waiting_scan() if rem(r) <= T]
            if waiting:  # v3b3: a short waits beside a long mid-prefill and no hold is taken: say why
                why = []
                for r in waiting[:4]:
                    rid = getattr(r, "request_id", None)
                    if rid in stuck:
                        why.append(f"{rid}: stuck (retry in {prefill_short_first_retry() - (self._short_first_steps - stuck[rid])})")
                    else:
                        used, need, total, n = self._kv_budget_room(r)
                        why.append(f"{rid}: KV gate (reserves {used} + {need} of {total} blocks, {n} resident)")
                self._short_first_no_hold("; ".join(why), longs)
            return []
        if getattr(self, "_short_first_run", 0) >= prefill_short_first_max():
            self._short_first_run = 0  # the long's turn
            self.prefill_short_first_yields = getattr(self, "prefill_short_first_yields", 0) + 1
            self._short_first_no_hold(f"the long's turn after {prefill_short_first_max()} held steps", longs)
            return []
        self._short_first_run = getattr(self, "_short_first_run", 0) + 1
        self.prefill_short_first_holds = getattr(self, "prefill_short_first_holds", 0) + 1
        return longs

    def _short_first_no_hold(self, why: str, longs: list) -> None:
        """v3b3 (program 2026-10-28T19:09:14Z (1)): one INFO line per step (the first 30, then every 500th) on which a
        short waits beside a long mid-prefill and the long is not held, with the reason"""
        n = self.prefill_short_first_no_hold = getattr(self, "prefill_short_first_no_hold", 0) + 1
        if n <= 30 or n % 500 == 0:
            logger.info("TT scheduler: SHORT_FIRST no hold (%d): %s; long(s) %s", n, why,
                        [(getattr(r, "request_id", None), int(r.num_prompt_tokens) - int(r.num_computed_tokens))
                         for r in longs[:2]])  # fmt: skip

    def _short_first_after(self, result: SchedulerOutput, held: list) -> None:
        """a held step that scheduled no prompt tokens wasted the long's turn: the waiting shorts it held for are
        marked stuck (retried after GLM53_PREFILL_SHORT_FIRST_RETRY steps) and the run counter given back"""
        held_ids = {getattr(r, "request_id", None) for r in held}
        got = [rid for rid, n in (getattr(result, "num_scheduled_tokens", None) or {}).items()
               if n and rid not in held_ids]  # fmt: skip
        if got:
            return
        T = prefill_short_first_tokens()
        for _, r in self._waiting_scan():  # v3b: both queues (a short parked in skipped_waiting is retried later too)
            if int(r.num_prompt_tokens) - int(r.num_computed_tokens) <= T:
                self._short_first_stuck[getattr(r, "request_id", None)] = self._short_first_steps
        self._short_first_run = max(0, getattr(self, "_short_first_run", 1) - 1)
        self.prefill_short_first_stuck = getattr(self, "prefill_short_first_stuck", 0) + 1
        n = self.prefill_short_first_stuck
        if n <= 20 or n % 100 == 0:  # v3b: one line names why a held step admitted nothing
            try:
                q = [(qn[0], getattr(r, "request_id", None), int(r.num_prompt_tokens), int(r.num_computed_tokens),
                      getattr(getattr(r, "status", None), "name", None), self._kv_fits(r))
                     for qn, r in self._waiting_scan()[:8]]  # fmt: skip
            except Exception as e:  # noqa: BLE001 -- a diagnostic never fails a step
                q = f"unavailable: {e}"
            logger.info("TT scheduler: SHORT_FIRST held step admitted nothing (stuck %d): hide %s; queue (s=skipped / "
                        "w=waiting, id, prompt, computed, status, fits) %s; split hook (id, in, out, why) %s", n,
                        getattr(self, "_kv_hide_why", None), q, getattr(self, "_split_trace", None))  # fmt: skip

    def _kv_fits(self, request: Any) -> bool:
        """THE KV-budget predicate (v3b: one function for the gate, the SHORT_FIRST hold and the overtake): the gate off, a
        resident request, nothing resident, or the reserve fits beside the residents' (no counters, no log)"""
        if os.environ.get("GLM53_ADMIT_KV_BUDGET", "0") != "1" or getattr(request, "status", None) in _KV_RESIDENT_STATUSES:
            return True
        used, need, total, resident = self._kv_budget_room(request)
        return not resident or used + need <= total

    _kv_admissible = _kv_fits

    def _waiting_scan(self) -> list:
        """[(queue name, request)] in vLLM's FCFS scan order (skipped_waiting first, then waiting), without the
        resident and the blocked statuses (vLLM steps past those itself)"""
        out = []
        blocked = getattr(self, "_is_blocked_waiting_status", None) or (lambda st: False)  # (not in every vLLM build)
        for name in ("skipped_waiting", "waiting"):
            for r in list(getattr(self, name, None) or ()):
                st = getattr(r, "status", None)
                if st in _KV_RESIDENT_STATUSES or (st is not None and blocked(st)):
                    continue
                out.append((name, r))
        return out

    def _short_first_kv_hide(self) -> dict | None:
        """GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE (default 1; with GLM53_PREFILL_SHORT_FIRST = T and
        GLM53_ADMIT_KV_BUDGET=1; program 2026-10-28T16:15:39Z / 16:15:53Z, the DP4 acceptance's blk3: its shorts
        waited 318-330 s behind its 1M). The KV-budget gate stops vLLM's waiting loop at the first request it defers
        ('no smaller request overtakes it'). A resident 1M reserves all but ~9.4K tokens of a 1M pool, so ONE 32K /
        131K arrival at the head of the queue held every short behind it for the 1M's whole prefill. This pass hides
        the waiting requests the gate defers, so a short (prompt left <= T) behind them is admitted when its own reserve
        fits; they keep their queue positions (put back by _short_first_kv_restore).
        v3b (program 2026-10-28T18:11:24Z: on the Galaxy the overtake never fired, holds 23 / stuck 23): ORDER-FREE
        over BOTH queues. v3 walked skipped_waiting then waiting and stopped at the first request that fit; a short
        parked in skipped_waiting (its lookup returned 'not yet') ended the walk with nothing hidden, while the hold
        (waiting only) still held the long. Now: when any fitting short waits in either queue, every gate-deferred
        request is hidden, and so is a fitting request over T that sits behind a hidden one (only shorts overtake).
        No hide at all when a deferred request is blocked by short residents alone (they drain, it is admitted next: no
        starvation). Any queue policy (v3b2). Returns {queue name: (the queue's order before, the hidden requests)} or None;
        the reason of a None is kept in self._kv_hide_why (the stuck-step line prints it)."""
        T = prefill_short_first_tokens()
        if T <= 0 or not prefill_short_first_kv_overtake() or os.environ.get("GLM53_ADMIT_KV_BUDGET", "0") != "1":
            self._kv_hide_why = "off"
            return None
        # (v3b2: no policy test -- v3's 'self.policy != SchedulingPolicy.FCFS' is the one guard that depends on how the
        # vLLM build stores the policy; the hide and the restore work for either queue type)
        scan = self._waiting_scan()
        if len(scan) < 2:
            self._kv_hide_why = "fewer than 2 waiting"
            return None
        rem = lambda r: int(r.num_prompt_tokens) - int(r.num_computed_tokens)  # noqa: E731
        fit = {id(r): self._kv_fits(r) for _, r in scan}
        if not any(fit[id(r)] and rem(r) <= T for _, r in scan):
            self._kv_hide_why = "no fitting short waits"
            return None
        hide, deferred_seen = [], False
        for name, r in scan:
            if fit[id(r)]:
                if deferred_seen and rem(r) > T:
                    hide.append((name, r))  # a fitting long behind a deferred request does not overtake
                continue
            _used, need, total, _n = self._kv_budget_room(r)
            used_long, _, _, _ = self._kv_budget_room(r, long_tokens=T)
            if used_long + need <= total:
                self._kv_hide_why = f"{getattr(r, 'request_id', None)} is blocked by short residents only (FIFO stop kept)"
                return None
            deferred_seen = True
            hide.append((name, r))
        if not deferred_seen:
            self._kv_hide_why = "nothing deferred"
            return None
        before = {n: list(getattr(self, n)) for n in ("skipped_waiting", "waiting") if getattr(self, n, None)}
        out: dict = {}
        for hn, hr in hide:
            out.setdefault(hn, (before[hn], []))[1].append(hr)
            getattr(self, hn).remove_request(hr)
        self._kv_hide_why = f"hid {len(hide)}"
        self.prefill_short_first_kv_overtakes = getattr(self, "prefill_short_first_kv_overtakes", 0) + 1
        seen = self.__dict__.setdefault("_kv_overtake_logged", set())
        for _, hr in hide:
            rid = getattr(hr, "request_id", None)
            if rid not in seen:
                seen.add(rid)
                if len(seen) > 4096:
                    seen.clear()
                logger.info("TT scheduler: %s (%d tokens left) sits the waiting loop out "
                            "(GLM53_PREFILL_SHORT_FIRST_KV_OVERTAKE): a long resident blocks its KV reserve; the shorts "
                            "behind it are admitted first", rid, rem(hr))  # fmt: skip
        return out

    def _short_first_kv_restore(self, hidden: dict | None) -> None:
        """put the requests _short_first_kv_hide took out back into their queues at their old positions, relative to
        whatever the pass left there (a request added in the pass keeps its place)"""
        for name, (before, hid) in (hidden or {}).items():
            q = getattr(self, name)
            pos = {id(r): i for i, r in enumerate(before)}
            pending = sorted(hid, key=lambda r: pos[id(r)])
            cur = list(q)
            order = []
            for r in cur:
                i = pos.get(id(r))
                while i is not None and pending and pos[id(pending[0])] < i:
                    order.append(pending.pop(0))
                order.append(r)
            order += pending
            q.remove_requests(cur)
            for r in order:
                q.add_request(r)

    def _prefill_chunk_run_budget(self, partial_prefills, pure_decodes, short_first: bool = False) -> int:
        """Token budget of this prefill step: the configured step budget, or
        ``prefill_chunk_run`` times it when the step would hold exactly one
        request that is already mid-prompt and nothing else could use the
        budget (no running decodes, nothing waiting). The run reaches the model
        as ONE prefill call of up to K chunks; the model submits its K bucket
        replays back to back and waits once (head-less intermediate chunks).
        Every other step keeps the admission budget, so new requests and
        decoders see the step cadence they see today."""
        base = int(self.max_num_scheduled_tokens)
        run = int(getattr(self, "_prefill_chunk_run", 1) or 1)
        if run <= 1 or len(partial_prefills) != 1:
            return base
        request = partial_prefills[0]
        # GLM53_PREFILL_SHORT_FIRST: a short owner with <= 8K left takes its run even beside decoders / waiting work
        # (program 13:32:43Z: "8K-and-under arrivals take a chunk run; 32K arrivals take chunks")
        short_run = short_first and int(request.num_prompt_tokens) - int(request.num_computed_tokens) <= run * base
        if pure_decodes and not short_run:
            return base
        if (self.waiting or getattr(self, "skipped_waiting", None)) and not short_run:
            return base
        if int(request.num_computed_tokens) <= 0:
            return base  # the first chunk: a fresh request's TTFT chunk is one bucket
        if int(request.num_computed_tokens) == _plan_cut_hint(request):
            return base  # tt.plan_cut (dev): the hinted hit's first continuation chunk is one bucket too
        if getattr(request, "mm_features", None) or getattr(
            request, "has_encoder_inputs", False
        ):
            return base
        return base * run

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running decode requests.

        Temporarily hides both the ``waiting`` and ``skipped_waiting`` queues
        so the base scheduler's waiting loop is a no-op and cannot promote a
        grammar-ready structured-output request into this decode step, and
        hides partial prefills so their next chunk is not scheduled into it
        either.  Any requests that get preempted during decode scheduling are
        merged back into the original queues afterwards.
        """
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        saved_skipped = getattr(self, "skipped_waiting", None)
        self.waiting = create_request_queue(self.policy)
        if saved_skipped is not None:
            self.skipped_waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]
        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            if saved_skipped is not None:
                if self.skipped_waiting:
                    saved_skipped.prepend_requests(self.skipped_waiting)
                self.skipped_waiting = saved_skipped
            self.waiting = saved_waiting
            if partial_prefills:
                self.running.extend(partial_prefills)
        return result

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Backstop against a stale-canvas resume vLLM's AR reset cannot represent.

        The engine layer (``_install_block_output_reset_abort_patch``) aborts
        running block requests and notifies their clients before delegating,
        so a reset requested through ``EngineCoreProc`` succeeds. Live block
        requests reach this guard only from engines that cannot notify the
        owning clients (a bare in-process ``EngineCore``) or callers that
        bypassed the engine layer; refusing here beats silently removing a
        request someone is still waiting on.
        """
        if self._is_block_output_model and reset_running_requests and self.running:
            message = (
                "Cannot reset prefix cache while a block-output request is "
                "running; finish or abort the request first."
            )
            # pause_generation(mode="keep") reaches this through an unguarded
            # idle-state callback while intentionally retaining live requests.
            # Returning False preserves them without letting an exception
            # escape EngineCore and strand the callback's Future.
            if self.pause_state == PauseState.PAUSED_ALL:
                logger.error("%s", message)
                return False
            raise RuntimeError(message)
        # Upstream copies token reservations into ``async_tokens_to_discard``.
        # TT receives one output frame for a speculative reservation of 1+K
        # tokens, so publish frame counts to both the scheduler and runner. The
        # stale frames remain visible to the scheduler but do not enter runner
        # request state before resumed-prefill inputs are built.
        reset_candidates = (
            [
                (
                    request,
                    self._pending_async_output_frames.get(request.request_id, 0),
                )
                for request in self.running
                if request.num_output_placeholders > 0
            ]
            if reset_running_requests
            else []
        )
        try:
            return super().reset_prefix_cache(reset_running_requests, reset_connector)
        finally:
            for request, frame_count in reset_candidates:
                if request.status == RequestStatus.PREEMPTED:
                    request.async_tokens_to_discard = frame_count
                if request.status == RequestStatus.PREEMPTED and frame_count > 0:
                    pending = self._pending_forced_reset_discard_counts
                    pending[request.request_id] = (
                        pending.get(request.request_id, 0) + frame_count
                    )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Reserve the physical output each scheduled step will commit.

        The platform removes the upstream diffusion marker, so vLLM reserves its
        normal one sampled-token placeholder. A plain block-output model commits
        a K-token canvas on every step, so reserve the remaining K-1 positions.
        An adaptive block-output model commits K only on a step that satisfies
        the ``block_step`` predicate below, and one token on every other step,
        so the extra reservation is per step. Each step's decision is recorded
        on its own SchedulerOutput because update_from_output must reconcile a
        commit against the decision that produced it.
        """
        super()._update_after_schedule(scheduler_output)
        output_frame_req_ids = tuple(
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if not self.requests[req_id].is_prefill_chunk
        )
        # The parent computes whether this chunk produces output. Retain that
        # decision on this step because a later schedule changes the request.
        setattr(scheduler_output, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, output_frame_req_ids)
        pending = getattr(self, "_pending_async_output_frames", None)
        if pending is None:
            pending = self._pending_async_output_frames = {}
        for req_id in output_frame_req_ids:
            pending[req_id] = pending.get(req_id, 0) + 1
        if self.num_spec_tokens and not self.scheduler_config.async_scheduling:
            # ``AsyncScheduler`` leaves every scheduled request holding
            # ``[-1] * num_spec_tokens``, which upstream's GPU runner
            # overwrites from its own state in ``_prepare_input_ids``. On a
            # synchronous launch the TT runner has no such step: it verifies
            # whatever the scheduler delivers, so a placeholder surviving here
            # becomes a draft the accept walk compares against, matches (the
            # model is handed the same placeholder), and commits as an output
            # token.
            #
            # Cleared rather than restored to the ids just scheduled: a
            # proposal is handed over once, so a request whose row proposed
            # nothing this step must speculate on nothing next step rather
            # than replay a spent proposal. Drafts reach a request only
            # through ``update_draft_token_ids``.
            #
            # Left standing on an asynchronous launch, because there they are
            # the only lookahead reservation the request gets: upstream stops
            # routing drafts through the scheduler (``EngineCore.post_step``
            # skips ``take_draft_token_ids``), the next schedule budgets
            # ``1 + len(spec_token_ids)`` positions for the request, and
            # ``TTModelRunner._drafts_to_verify`` reads the placeholders as
            # that reservation and verifies the proposal the runner holds.
            for req_id in scheduler_output.num_scheduled_tokens:
                self.requests[req_id].spec_token_ids = []
        if not self._is_block_output_model:
            return
        extra_placeholders = (
            self._output_tokens_per_step - self.num_sampled_tokens_per_step
        )
        # Adaptive: the model emits its block ONLY on a SOLO DECODE step; every
        # other step (batched decodes, and the prefill that host-samples the
        # anchor) commits exactly one plain token and reserves one placeholder.
        # This MUST match the model's own gate, which runs the spec block in
        # decode_forward when batch == 1 and a plain token otherwise:
        #   - solo == batch 1 (exactly one request scheduled this step);
        #   - decode == the prompt was fully computed BEFORE this step (see the
        #     decode test below, which subtracts this step's scheduled tokens).
        # The decision is attached to THIS step's SchedulerOutput (not a Request
        # slot): under async the next step's schedule() overwrites Request state
        # before this step's output commits, but the engine core hands the
        # matching SchedulerOutput back to update_from_output, so the map always
        # pairs with the output being committed.
        solo = len(scheduler_output.num_scheduled_tokens) == 1
        if self._is_adaptive_block:
            self._mirror_spec_session(scheduler_output, solo)
        decisions: dict[str, bool] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            if self._is_adaptive_block:
                # num_computed_tokens is already advanced by THIS step's
                # scheduled tokens here, so subtract them back out: a step is a
                # decode iff the prompt was fully computed BEFORE it. This is a
                # pure scheduling-side quantity -- output-commit timing (which
                # differs between sync tests and the pipelined engine loop)
                # cannot skew it. A resumed replay scheduling prompt+output
                # tokens lands back below the prompt boundary and correctly
                # stays a non-block step.
                scheduled = scheduler_output.num_scheduled_tokens[req_id]
                is_decode = (
                    request.num_computed_tokens - scheduled >= request.num_prompt_tokens
                )
                # Owning the model's single spec session is the whole gate:
                # the model serves a solo decode as plain baseline whenever it
                # has no session for THAT request, so reserving a block for a
                # non-owner would reserve a width the model cannot emit (see
                # _mirror_spec_session). The capture frontier is deliberately
                # NOT re-derived here -- it is a property of the PREFILL that
                # armed the session, already recorded in the ownership mirror,
                # and re-deriving it from the request's CURRENT length would
                # flip a live session's reservation to width 1 mid-generation
                # while the model keeps emitting blocks.
                block_step = solo and is_decode and self._spec_session_owner == req_id
            else:
                block_step = True
            if block_step:
                request.num_output_placeholders += extra_placeholders
            decisions[req_id] = block_step
        set_tt_block_step_decisions(scheduler_output, decisions)

    def _spec_frontier_ok(self, measured_len: int) -> bool:
        """Whether the model arms a session for a prefill of ``measured_len``.

        The model measures its capture frontier against the ``prompt_lens`` the
        runner hands it, which is ``input_positions + chunk_lens`` -- the tokens
        computed INCLUDING this step. On a replay resumed from preemption that
        spans the generated output too, so it is NOT ``num_prompt_tokens``:
        measuring the prompt alone let a resumed request cross the frontier on
        the model side only, which drops the session while this scheduler still
        reserved a block, and the width check then kills the engine core
        (vllm-tt-plugin#118.2). The caller passes ``num_computed_tokens``, which
        ``_update_after_schedule`` has already advanced by this step.
        """
        return (
            self._adaptive_block_max_prompt == 0
            or measured_len <= self._adaptive_block_max_prompt
        )

    def _mirror_spec_session(self, scheduler_output, solo: bool) -> None:
        """Track which request owns the adaptive model's SINGLE spec session.

        The model keeps one global session: drafter taps are captured during a
        request's own prefill, and only the request that owns them can emit a
        multi-token block. Every transition of that session is driven by the
        shape of a step, so the scheduler can mirror it exactly without asking
        the model -- and that is what keeps the width this scheduler RESERVES
        equal to the width the model EMITS:

        * a PREFILL step re-seats the session. The model captures taps only for
          a solo, spec-eligible prefill and drops the session on any other
          prefill (batched, or a prompt over the capture frontier).
        * a BATCHED decode step destroys it. The model releases the decoder and
          serves plain baseline from then on, and it never re-arms, because
          taps only ever come from a prefill. This is the case that made a
          benchmark sweep fail: as concurrency drains back to one request, that
          request is solo and eligible but no longer owns a session.
        * a SOLO decode by the OWNER leaves ownership alone -- it bootstraps
          its pending taps and keeps the session for the rest of its life.
        * a SOLO decode by ANY OTHER request destroys it. Async scheduling can
          skip the owner once it has reached max_tokens (upstream guards that
          skip on num_output_placeholders), leaving a different request alone
          on the next step while the session is still armed. The model releases
          the session on that step for the same reason -- it would otherwise
          speculate from the owner's residual taps -- and this mirrors that
          release, so the reserved width stays 1 for the non-owner
          (vllm-tt-plugin#118.1).

        A block-output step is never mixed prefill+decode (docs/SCHEDULING.md;
        mixed steps stay off for block-output models), so the first scheduled
        request classifies the whole step.
        """
        owner = self._spec_session_owner
        # A finished or aborted owner no longer holds the session: the model
        # clears it in release_request, and the id is gone from self.requests.
        if owner is not None and owner not in self.requests:
            owner = None
        scheduled_tokens = scheduler_output.num_scheduled_tokens
        first_id = next(iter(scheduled_tokens), None)
        if first_id is not None:
            request = self.requests[first_id]
            step_is_decode = (
                request.num_computed_tokens - scheduled_tokens[first_id]
                >= request.num_prompt_tokens
            )
            if not step_is_decode:
                # num_computed_tokens is advanced by this step already, so it is
                # exactly the prompt_lens the model measures its frontier on.
                owner = (
                    first_id
                    if (solo and self._spec_frontier_ok(request.num_computed_tokens))
                    else None
                )
            elif not solo or first_id != owner:
                owner = None
        self._spec_session_owner = owner

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Bind this step's block-step decisions before the base loop commits
        its outputs, so ``_update_request_with_output`` reads the decision that
        produced THIS output rather than a later schedule's overwrite, and
        consume only the output frames this scheduled step reserved."""
        if self._is_block_output_model:
            self._committing_block_step_decisions = get_tt_block_step_decisions(
                scheduler_output
            )
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            if self._request_timeout_s:
                self._expire_timed_out_requests(outputs)
            self._emit_refused(outputs)
            self._publish_admit_headroom_if_drained()  # v3b6: the step that empties the engine
            return outputs
        finally:
            self._committing_block_step_decisions = {}
            pending = self._pending_async_output_frames
            for req_id in getattr(scheduler_output, _TT_OUTPUT_FRAME_REQ_IDS_ATTR, ()):
                count = pending.get(req_id, 0)
                if count <= 1:
                    pending.pop(req_id, None)
                else:
                    pending[req_id] = count - 1

    def _refuse_unplannable(self) -> None:
        """program 2026-10-26T17:09:24Z (a k3 MUST): a RUNNING request whose remaining prefill cannot PLAN on the
        model (vllm_tt_plugin.plan_check: TPPrefill.plan / clamp_lengths, host-only -- 'chunk [s, s+c) past the cache')
        is finished ABORTED here, before this step schedules it, instead of raising inside the runner step (where
        _latch_device_faults takes the engine down). Its client hears finish_reason abort with stop_reason
        'plan: <error>' at the next update_from_output; every other request keeps stepping. A prefix hit's start is
        planned by the connector (it steps a hit down to a plannable checkpoint); a fresh prompt starts on the chunk
        grid at 0."""
        from vllm_tt_plugin import plan_check, row_check  # noqa: PLC0415

        refused = row_check.take()  # (the runner's short page-row refusals, vllm_tt_plugin.row_check)
        if refused:
            live = {r.request_id for r in list(self.running) + list(self.waiting) if not r.is_finished()}
            ids = [r for r in refused if r in live]
            finished = self.finish_requests(ids, RequestStatus.FINISHED_ABORTED) if ids else []
            pending = self.__dict__.setdefault("_refused_outputs", [])
            for request in finished:
                request.stop_reason = refused.get(request.request_id, "page row")[:200]
                logger.warning("TT request %s refused by the runner (%s); finished with finish_reason abort, the engine "
                               "keeps stepping", request.request_id, request.stop_reason)  # fmt: skip
                pending.append(request)
        if not plan_check._CHECK:
            return
        bad = {}
        for request in list(self.running) + list(self.waiting):  # (a waiting request resumes at its computed tokens)
            if request.is_finished():
                continue
            done = int(request.num_computed_tokens)
            left = int(request.num_prompt_tokens) - done
            if left <= 0:
                continue
            err = plan_check.plan_error(done, left)
            if err is not None:
                bad[request.request_id] = err
        if not bad:
            return
        finished = self.finish_requests(list(bad), RequestStatus.FINISHED_ABORTED)
        pending = self.__dict__.setdefault("_refused_outputs", [])
        for request in finished:
            request.stop_reason = f"plan: {bad.get(request.request_id, '?')}"[:200]
            logger.warning("TT request %s refused: its remaining prefill does not plan on the model (%s); finished with "
                           "finish_reason abort, the engine keeps stepping", request.request_id, bad.get(request.request_id))  # fmt: skip
            pending.append(request)

    def _emit_refused(self, outputs: dict[int, EngineCoreOutputs]) -> None:
        """the client outputs of the requests _refuse_unplannable finished (as _expire_timed_out_requests builds them)"""
        pending = self.__dict__.get("_refused_outputs")
        if not pending:
            return
        self._refused_outputs = []
        for request in pending:
            client_outputs = outputs.get(request.client_index)
            if client_outputs is None:
                client_outputs = EngineCoreOutputs()
                outputs[request.client_index] = client_outputs
            client_outputs.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )

    def _expire_timed_out_requests(self, outputs: dict[int, EngineCoreOutputs]) -> None:
        """Finish every request older than ``tt.request_timeout_s`` (arrival to
        now, queueing included) and tell its client in THIS step's outputs.

        Done here, at the step boundary after the base scheduler committed the
        step's tokens, so a timed-out request never gets a later token and the
        next ``_update_states`` sees it in ``finished_req_ids`` exactly like a
        client abort: the runner releases its device state slot
        (``_release_model_request``) and vLLM freed its KV blocks in
        ``finish_requests``. Co-scheduled requests are untouched: only the
        expired request's own output row is added. The notification is built
        here rather than through ``EngineCoreProc._send_abort_outputs`` so an
        in-process engine (``LLMEngine``, host tests) sees the finish too.
        """
        now = self._tt_clock()
        budget = self._request_timeout_s
        expired = [
            request
            for request in itertools.chain(self.running, self.waiting)
            if not request.is_finished() and now - request.arrival_time > budget
        ]
        if not expired:
            return
        finished = self.finish_requests(
            [request.request_id for request in expired],
            RequestStatus.FINISHED_ABORTED,
        )
        for request in finished:
            request.stop_reason = "timeout"
            logger.warning(
                "TT request %s exceeded request_timeout_s=%.1f (age %.1f s, %d "
                "output token(s)); finished with finish_reason abort",
                request.request_id,
                budget,
                now - request.arrival_time,
                request.num_output_tokens,
            )
            client_outputs = outputs.get(request.client_index)
            if client_outputs is None:
                client_outputs = EngineCoreOutputs()
                outputs[request.client_index] = client_outputs
            client_outputs.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """Commit one block and reconcile its full physical reservation."""
        if not self._is_block_output_model:
            return super()._update_request_with_output(request, new_token_ids)
        # Adaptive: a request that decoded batched (or over-frontier) this step
        # committed a single baseline token (no block was reserved) -> plain
        # reconciliation. The decision is read from THIS step's SchedulerOutput
        # (set in _update_after_schedule), which the engine core pairs with this
        # output even when async scheduling has already run a later schedule().
        if self._is_adaptive_block:
            block_step = self._committing_block_step_decisions.get(request.request_id)
            if block_step is None:
                raise RuntimeError(
                    "adaptive block-output request committed output without a "
                    f"scheduling decision: req_id={request.request_id!r}"
                )
            if not block_step:
                # The scheduler stamped this step width 1, so the model owes
                # exactly one token. Check before delegating: super() appends
                # whatever it is handed, so a model that returned a BLOCK here
                # would commit extra tokens against a single reserved
                # placeholder and the mismatch would surface later as a
                # placeholder leak or a corrupted continuation, far from its
                # cause. This is the mirror of the block-width check below.
                if len(new_token_ids) != self.num_sampled_tokens_per_step:
                    raise ValueError(
                        "Model output width violates the scheduled baseline "
                        f"width: got {len(new_token_ids)}, expected "
                        f"{self.num_sampled_tokens_per_step} "
                        f"(req_id={request.request_id!r}); the scheduler "
                        "stamped this step width 1 (batched, or a prompt over "
                        "the spec frontier), so the model must return a single "
                        "baseline token -- the scheduler and model block gates "
                        "disagree"
                    )
                return super()._update_request_with_output(request, new_token_ids)
        if request.async_tokens_to_discard:
            # A block step reserved K placeholders; the AsyncScheduler discard
            # path drains only one per stale frame, so it cannot balance a
            # dropped block. Block requests are never reset-preempted
            # (reset_prefix_cache raises while one runs) and solo spec steps are
            # not KV-preempted, so this must not happen -- fail loudly rather
            # than silently leak placeholders.
            raise RuntimeError(
                "A stale async output reached block serving for a block step; "
                "block-output frames cannot be discarded (reset/preempt of a "
                "running block request is unsupported)"
            )
        if len(new_token_ids) != self._output_tokens_per_step:
            raise ValueError(
                "Model output width violates output_tokens_per_step: "
                f"{len(new_token_ids)} != {self._output_tokens_per_step} "
                f"(req_id={request.request_id!r}); the scheduler reserved a "
                "block but the model returned a different width -- the "
                "scheduler and model block gates disagree"
            )

        # Scheduler appends token-by-token and trims at EOS, stop tokens,
        # max_tokens, or max_model_len. The reservation is physical, so consume
        # all K placeholders even when the client-visible block is trimmed.
        # Calling Scheduler directly intentionally skips AsyncScheduler's
        # cache_blocks hook. Platform validation disables prefix caching for
        # block-output models, and __init__ asserts that invariant.
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids
        )
        request.num_output_placeholders -= self._output_tokens_per_step
        if request.num_output_placeholders < 0:
            raise RuntimeError(
                "Output placeholders underflowed after block-output reconciliation"
            )
        return new_token_ids, stopped
