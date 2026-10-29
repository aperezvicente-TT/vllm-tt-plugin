# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Config-time admission policy for speculative decoding.

Separate from ``spec_decode``, which is the value contract a tt-metal model
class imports to build a ``SpecPlan``. Policy lives here so that contract module
stays a leaf with no vLLM dependency: only this module reads
``model_capabilities`` keys, resolves a vLLM speculative method name, and
decides whether a launch may proceed.

Every refusal raises. Speculation is never disabled silently, because a server
that quietly serves without it reports speedups it did not achieve.

The model-side contract this admits against, which a tt-metal model class must
implement, is documented in ``docs/SPEC_DECODE_CONTRACT.md``.
"""

import os
from typing import TYPE_CHECKING, get_args

from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    DRAFTER_STATE_PAGED,
    HIDDEN_HANDOFFS,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE,
    SPEC_REQUIREMENTS,
    SpecPlan,
    SpecReject,
    normalize_declared_values,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Accept modes the runner can drive today. A plan offering none of these is
# refused; a plan offering more keeps its extra modes and is still admitted,
# so declaring a real capability never makes a model less admissible.
#
# ``logits`` is a legal mode and a model may serve it, but the runner asks for
# ``argmax_ids`` on every step and refuses any other answer, so admitting a
# logits-only plan would pass a launch that fails on its first decode. The
# sampled accept walk is what adds it back.
_RUNNABLE_ACCEPT_MODES = (ACCEPT_MODE_ARGMAX_IDS,)

# Upstream custom_class is a user-provided proposer extension that needs no
# separate draft checkpoint configuration. The TT runner uses that category
# to call the loaded model's propose_draft_tokens instead of importing and
# constructing an upstream proposer. The model value is therefore a fixed TT
# dispatch marker, not an executable class path or a weights location.
MODEL_OWNED_DRAFT_METHOD = "custom_class"
MODEL_OWNED_DRAFT_SENTINEL = "vllm_tt_plugin.model_owned_drafter"

# vLLM's name for a model drafting with its own Multi-Token Prediction head. On
# TT that head is part of the loaded target, so the method runs through the
# same model-owned drafter contract as ``custom_class``: the runner calls
# ``propose_draft_tokens``, and no upstream proposer or draft checkpoint is
# built (``platform._install_tt_mtp_speculative_config_patch`` keeps vLLM's
# ``SpeculativeConfig`` from building a draft model config). It keeps vLLM's
# EAGLE/MTP requirements below, because an MTP head reads the target's hidden
# state by construction.
MTP_DRAFT_METHOD = "mtp"

# vLLM's name for the DFlash family of block drafters (a separate checkpoint
# drafting a block of K tokens from the target's hidden-state taps). On TT the
# loaded GLM-5.3-Flash backend owns that drafter too: the Apache-2.0 DFlash2
# drafter runs INSIDE the model's speculative decode step (the backend is built
# with GLM53_SPEC_KIND=dflash2, its weights read from GLM53_DF2_DIR), so the
# runner calls ``propose_draft_tokens`` exactly as for ``mtp`` and vLLM builds
# no draft model (``platform._install_tt_mtp_speculative_config_patch``). The
# only drafter checkpoint admitted is the Apache-2.0 one.
DFLASH_DRAFT_METHOD = "dflash"
DFLASH_DRAFTER_ID = "canada-quant/GLM-5.3-Flash-DFlash2-G"
DFLASH_DRAFTER_DIR_ENV = "GLM53_DF2_DIR"

# Methods whose drafts come from the loaded model's ``propose_draft_tokens``.
MODEL_OWNED_DRAFT_METHODS = (
    MODEL_OWNED_DRAFT_METHOD,
    MTP_DRAFT_METHOD,
    DFLASH_DRAFT_METHOD,
)

# Methods the runner can actually propose drafts for. The requirements table
# below says what a method needs *of the model*; this says what the plugin has
# implemented. Admitting a method with no proposer would start a server that
# takes the speculative flags, drafts nothing, and serves plain decoding while
# reporting a speedup it never achieved.
_PROPOSABLE_METHODS = ("ngram", *MODEL_OWNED_DRAFT_METHODS)

_DEVICE_DRAFTER = (SPEC_REQUIREMENT_DEVICE_PROPOSE, SPEC_REQUIREMENT_HIDDEN_FEED)


def _build_method_requirements() -> dict[str, tuple[str, ...]]:
    """Map recognized vLLM method names to model capability requirements.

    Built once from vLLM's own literals rather than hand-copied, so an upstream
    rename drops a name out of this table instead of leaving the plugin mapping
    a name vLLM no longer knows. Execution also requires a proposer implemented
    by the runner and admitted through ``_PROPOSABLE_METHODS``.
    """
    # Imported lazily: vllm.config pulls in a module that resolves
    # current_platform at import time, which loads this plugin, so importing it
    # at module scope from a module on the plugin's own bootstrap path is a
    # cycle. The same reasoning is spelled out at platform.py's vllm.config
    # import.
    from vllm.config.speculative import EagleModelTypes

    table: dict[str, tuple[str, ...]] = {
        # Host drafters ask nothing of the model. The model still needs a
        # spec_plan, because the plugin needs its feasible (max_num_seqs, K)
        # points either way.
        "ngram": (),
        "suffix": (),
        # A separate draft model carries its own growing KV cache, which the
        # scheduler must own.
        "draft_model": (
            SPEC_REQUIREMENT_DEVICE_PROPOSE,
            SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE,
        ),
        # Device drafters that read the target's hidden state but are not in
        # vLLM's EagleModelTypes grouping.
        "medusa": _DEVICE_DRAFTER,
        "mlp_speculator": _DEVICE_DRAFTER,
        # The model's own drafter, proposing on device through
        # ``propose_draft_tokens``. It proposes, and that is all this method
        # can demand: what its drafter reads is the model's own business. An
        # MTP head reads the target hidden state and declares
        # ``hidden_feed`` for it, while a drafter continuing from the committed
        # block alone declares nothing extra, and requiring the declaration
        # here would have forced that model to claim a feed it never uses. The
        # named upstream methods below are different: each one is a drafter
        # architecture that reads the hidden state by construction, which is
        # why ``mtp`` keeps that requirement even though TT serves it through
        # this same contract.
        MODEL_OWNED_DRAFT_METHOD: (SPEC_REQUIREMENT_DEVICE_PROPOSE,),
    }
    # EagleModelTypes flattens to EAGLE, every MTP variant and dFlash. All of
    # them draft on device from the target's hidden state.
    for name in get_args(EagleModelTypes):
        table.setdefault(name, _DEVICE_DRAFTER)
    return table


def method_requirements(method: str) -> tuple[str, ...]:
    """What a vLLM speculative method requires of the model.

    The plugin owns this mapping, so a new upstream method name is a plugin
    change rather than a model release, and a model declares what it can serve
    without ever enumerating method names.
    """
    table = _build_method_requirements()
    if method in table:
        return table[method]
    raise ValueError(
        f"speculative method {method!r} is not supported by the TT backend. "
        f"Set --spec-method, or the 'method' key inside --speculative-config, "
        f"to one of {sorted(table)}"
    )


def resolve_speculative_plan(
    vllm_config: "VllmConfig",
    model_class: type,
    model_capabilities: dict | None,
    max_num_seqs: int,
) -> SpecPlan | None:
    """Resolve a speculative configuration at config time, or refuse it.

    Returns the model's plan when speculation is admitted, and ``None`` when
    the configuration asks for no speculation. Every refusal raises with the
    offending values.

    ``max_num_seqs`` is the concurrency the plan is dimensioned against, passed
    in rather than read off ``vllm_config`` because the platform rewrites
    ``scheduler_config.max_num_seqs`` when it folds data parallelism into lanes.
    Pass the value that is final for the launch.

    The model's ``spec_plan`` classmethod is called with a ``vllm_config`` on
    which no Tenstorrent platform state is guaranteed to be stored, so a
    ``spec_plan`` implementation must not read ``get_tt_*`` helpers off it.
    """
    speculative_config = vllm_config.speculative_config
    if not speculative_config:
        return None

    capabilities = model_capabilities or {}
    if not capabilities.get("supports_spec_decode", False):
        raise ValueError(
            f"{model_class.__name__} does not declare "
            "model_capabilities['supports_spec_decode'], so it cannot serve a "
            "speculative_config. Drop the speculative flags"
        )

    method = str(speculative_config.method)
    required = method_requirements(method)
    declared = normalize_declared_values(
        capabilities.get("spec_requirements"),
        SPEC_REQUIREMENTS,
        f"{model_class.__name__} model_capabilities['spec_requirements']",
    )
    missing = [name for name in required if name not in declared]
    if missing:
        raise ValueError(
            f"speculative method {method!r} requires {list(required)}, but "
            f"{model_class.__name__} declares spec_requirements "
            f"{list(declared)}; missing {missing}. Change --spec-method, or "
            "drop the speculative flags"
        )

    # Validated when the hidden state is fed at all, whether the method
    # demands it or the model volunteers it. A model declaring the feed and no
    # handoff has not said how the state reaches its drafter, and the runner
    # reads the handoff to decide whether a step that produces no hidden
    # handle can still ask that drafter to propose. A typo in a declaration no
    # launch uses stays unvalidated, so it cannot refuse an ngram launch.
    if (
        SPEC_REQUIREMENT_HIDDEN_FEED in required
        or SPEC_REQUIREMENT_HIDDEN_FEED in declared
    ):
        handoff = normalize_declared_values(
            capabilities.get("spec_hidden_handoff"),
            HIDDEN_HANDOFFS,
            f"{model_class.__name__} model_capabilities['spec_hidden_handoff']",
        )
        if not handoff:
            raise ValueError(
                f"speculative method {method!r} feeds the target hidden state "
                f"to its drafter, but {model_class.__name__} declares no "
                f"spec_hidden_handoff; expected one of "
                f"{sorted(HIDDEN_HANDOFFS)}"
            )

    # vLLM's custom-class method carries a dotted proposer path it loads in its
    # own runner. The TT runner loads nothing: the drafter is the model. So the
    # path is pinned to one documented value, because any other one names a
    # proposer that will never be imported and would read as the thing doing
    # the drafting.
    if method == MODEL_OWNED_DRAFT_METHOD:
        declared_model = getattr(speculative_config, "model", None)
        if declared_model != MODEL_OWNED_DRAFT_SENTINEL:
            raise ValueError(
                f"speculative method {method!r} means the model's own drafter, "
                "so its 'model' key must be exactly "
                f"{MODEL_OWNED_DRAFT_SENTINEL!r}, not {declared_model!r}. vLLM "
                "requires a dotted path there and nothing imports it: the "
                f"drafter is {model_class.__name__}.propose_draft_tokens"
            )

    # vLLM's mtp method reads 'model' as the checkpoint holding the MTP head and
    # defaults it to the target's own. On TT the drafter is the loaded target's
    # head, so 'model' may name only the target: any other checkpoint is never
    # loaded, and naming it would read as the thing doing the drafting.
    if method == MTP_DRAFT_METHOD:
        declared_model = getattr(speculative_config, "model", None)
        target_config = getattr(vllm_config, "model_config", None)
        target_model = getattr(target_config, "model", None)
        if declared_model is not None and declared_model != target_model:
            raise ValueError(
                f"speculative method {method!r} means the MTP head of "
                f"{model_class.__name__} itself, drafting on device through "
                "propose_draft_tokens, so its 'model' key must be unset or "
                f"name the target checkpoint {target_model!r}, not "
                f"{declared_model!r}. The TT backend loads no separate draft "
                "checkpoint"
            )

    # vLLM's dflash method reads 'model' as the drafter checkpoint and builds a
    # draft model from it. On TT the loaded target runs the drafter itself and
    # reads its weights from GLM53_DF2_DIR, so 'model' may only name the
    # Apache-2.0 drafter (its Hub id, or the local directory the backend reads):
    # any other checkpoint is never loaded and would read as the drafter.
    if method == DFLASH_DRAFT_METHOD:
        declared_model = getattr(speculative_config, "model", None)
        local_dir = os.environ.get(DFLASH_DRAFTER_DIR_ENV)
        is_dir = isinstance(declared_model, str) and os.path.isdir(declared_model)
        if not (
            declared_model in (None, DFLASH_DRAFTER_ID)
            or (local_dir is not None and declared_model == local_dir)
            or (local_dir is None and is_dir)
        ):
            raise ValueError(
                f"speculative method {method!r} means the Apache-2.0 DFlash2 "
                f"drafter {DFLASH_DRAFTER_ID!r} run by {model_class.__name__} "
                "itself (propose_draft_tokens), so its 'model' key must be "
                f"unset, that id, or the local drafter directory "
                f"{DFLASH_DRAFTER_DIR_ENV} names ({local_dir!r}), not "
                f"{declared_model!r}. The TT backend loads no separate draft "
                "checkpoint through vLLM"
            )
        if local_dir is None and is_dir:
            # the backend reads the drafter from this directory (the engine
            # core inherits the environment of this process)
            os.environ[DFLASH_DRAFTER_DIR_ENV] = declared_model

    # Refused on the requirement, not only on the returned plan, so a model
    # whose two declarations contradict each other is reported rather than
    # admitted through the gap between them.
    if SPEC_REQUIREMENT_PAGED_DRAFTER_CACHE in required:
        raise ValueError(
            f"speculative method {method!r} needs a scheduler-owned drafter "
            "cache, which the TT backend does not yet allocate. Change "
            "--spec-method, or drop the speculative flags"
        )

    if method not in _PROPOSABLE_METHODS:
        raise ValueError(
            f"speculative method {method!r} is known to the TT backend but no "
            f"proposer drives it yet; the runner proposes for "
            f"{list(_PROPOSABLE_METHODS)}. A host method needs its proposer "
            "wired into the runner, and a device method needs that plus a call "
            "to the model's propose_draft_tokens and the hidden-state handoff. "
            "Set --spec-method, or the 'method' key inside "
            "--speculative-config, or drop the speculative flags"
        )

    spec_plan = getattr(model_class, "spec_plan", None)
    if not callable(spec_plan):
        raise ValueError(
            f"{model_class.__name__} declares supports_spec_decode but its "
            f"spec_plan is {spec_plan!r}, not a callable classmethod, so its "
            "feasible (max_num_seqs, K) points cannot be resolved"
        )

    requested_k = int(speculative_config.num_speculative_tokens)
    outcome = spec_plan(vllm_config, max_num_seqs, requested_k)

    if isinstance(outcome, SpecReject):
        supported = (
            f"; supported draft lengths at max_num_seqs={max_num_seqs}: "
            f"{list(outcome.supported_k)}"
            if outcome.supported_k
            else f"; no draft length is supported at max_num_seqs={max_num_seqs}"
        )
        raise ValueError(
            f"{model_class.__name__} refused speculation at "
            f"max_num_seqs={max_num_seqs}, num_speculative_tokens="
            f"{requested_k}: {outcome.reason}{supported}. Change --spec-tokens "
            "or --max-num-seqs, or drop the speculative flags"
        )
    if not isinstance(outcome, SpecPlan):
        raise ValueError(
            f"{model_class.__name__}.spec_plan must return SpecPlan or "
            f"SpecReject, got {type(outcome).__name__}"
        )
    if outcome.effective_k > requested_k:
        raise ValueError(
            f"{model_class.__name__}.spec_plan returned effective_k="
            f"{outcome.effective_k}, above the requested "
            f"num_speculative_tokens={requested_k}"
        )
    if outcome.drafter_state == DRAFTER_STATE_PAGED:
        raise ValueError(
            f"{model_class.__name__}.spec_plan declares drafter_state "
            f"{DRAFTER_STATE_PAGED!r}, which needs a scheduler-owned drafter "
            "cache the TT backend does not yet allocate"
        )
    if not any(mode in outcome.accept_modes for mode in _RUNNABLE_ACCEPT_MODES):
        raise ValueError(
            f"{model_class.__name__}.spec_plan offers accept modes "
            f"{list(outcome.accept_modes)}, none of which the runner can "
            f"drive; it drives {list(_RUNNABLE_ACCEPT_MODES)}"
        )

    return outcome


__all__ = [
    "method_requirements",
    "resolve_speculative_plan",
]
