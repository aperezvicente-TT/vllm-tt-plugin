#!/usr/bin/env bash
# serve_cmd.host-8chip.sh — the server command the supervisor runs on host-8chip (one line in
# supervisor.host-8chip.json: "server_cmd"). It is serve_window.sh's launch without the window / workload /
# scheduler logic: the installed resident target (work-s9/resident_target.sh: tree, venv, plugin, image pin,
# 14b's served env), Stage 9's env.sh, the pinned server args, `tt-lock run all` (the lock is held for the
# server's lifetime = a standing server), the API-server pid written for the supervisor's SIGINT.
# The supervisor supplies: GLM53_STEP_STATUS (plugin step meter), GLM53_SUPERVISOR_PIDFILE,
# GLM53_SUPERVISOR_RUN_DIR, PROMETHEUS_MULTIPROC_DIR, GLM53_STARTUP_TIMELINE, GLM53_TIMELINE_PATH.
# Everything the operator may want to change is an environment variable with the same name and default as
# serve_window.sh / launch_resident.sh; the target file wins over the defaults below.
set -u
W=/opt/glm53/work-s9
S=/opt/glm53/kit/doc/ttnn/09-vllm/scripts
TL=/opt/glm53/kit/.agents/scripts/tt-lock
RUN=${GLM53_SUPERVISOR_RUN_DIR:-$W/supervisor/runs/manual}; mkdir -p $RUN
PIDFILE=${GLM53_SUPERVISOR_PIDFILE:-$RUN/api_server.pid}

# 1. the installed resident target (install_on_gate.sh writes it; launch_resident.sh's defaults otherwise)
TARGET=${GLM53_TARGET_FILE:-$W/resident_target.sh}
if [ -f "$TARGET" ]; then eval "$(grep -E '^export GLM53_' "$TARGET" | sed 's/ GLM53_RUN=.*//')"; fi
export GLM53_S9_TREE=${GLM53_S9_TREE:-/opt/glm53/tt-metal-s9i8}
export GLM53_VENV=${GLM53_VENV:-/opt/glm53/venv-lmcache}
export GLM53_PLUGIN_SRC=${GLM53_PLUGIN_SRC:-/opt/glm53/vllm-tt-plugin-s9/src}
export GLM53_BACKEND_MODULE=${GLM53_BACKEND_MODULE:-models.autoports.zai_org_glm_5_3_flash.tt.stack_model}
export GLM53_BACKEND_KW=${GLM53_BACKEND_KW:-'{"native_t": true, "weight_cache": {}}'}
: "${GLM53_STACK_KNOBS:=s7a283}" "${GLM53_TP_XSL:=1}" "${GLM53_TP_XSL_SLICED:=1}" "${GLM53_STACK_WIDTHS:=none}"
: "${GLM53_STACK_PREFILL_TRACES:=0}" "${GLM53_STACK_SHARE_FABRIC:=1}" "${GLM53_STACK_KNOB_SET:=s:GLM53_MHC_PRE_SPLIT=1}"
: "${GLM53_STACK_NATIVE_WIDTHS:=8,16}" "${GLM53_EXPECT_CHUNK:=32}"
: "${GLM53_TP_IMAGE_ROOT:=/opt/glm53/data/dram_images/tp14b}"
export GLM53_STACK_KNOBS GLM53_TP_XSL GLM53_TP_XSL_SLICED GLM53_STACK_WIDTHS GLM53_STACK_PREFILL_TRACES
export GLM53_STACK_SHARE_FABRIC GLM53_STACK_KNOB_SET GLM53_STACK_NATIVE_WIDTHS GLM53_EXPECT_CHUNK GLM53_TP_IMAGE_ROOT
export GLM53_IMAGE_PIN=${GLM53_IMAGE_PIN:-fde4697e51b2-tp-ds-v1-2x4-1530493535ed}
if [ "${GLM53_LAUNCH_WEIGHTS:-image}" = tensor ]; then
  unset GLM53_WEIGHT_SOURCE GLM53_IMAGE_REQUIRED GLM53_EXPECT_WEIGHT_SOURCE
else
  export GLM53_WEIGHT_SOURCE=image GLM53_IMAGE_REQUIRED=1 GLM53_IMAGE_BUILD=0 GLM53_EXPECT_WEIGHT_SOURCE=image
fi
export TMPDIR=/opt/glm53/tz

# 2. Stage 9's environment (tree, venv, plugin path, vLLM timeouts, mesh descriptor, ccache)
source $S/env.sh
cd $W  # neutral cwd: tt-metal resolves kernel sources cwd-first

# 3. the pinned server (serve_window.sh's SERVER_ARGS; GLM53_EXTRA_SERVER_ARGS / GLM53_SPEC_CONFIG as there)
PORT=${GLM53_PORT:-8000}; SEQS=${GLM53_MAX_SEQS:-32}; MML=${GLM53_MAX_MODEL_LEN:-2051}
: "${GLM53_TT_EXTRA_CFG:='"worker_l1_size": 1264512'}"   # launch_resident.sh's value
TT_CFG=$(printf '{"sample_on_device_mode": "all", "trace_mode": "all", "enable_model_warmup": true, "trace_region_size": %d, "prefill_chunk_alignment": 128%s}' \
  ${GLM53_TRACE_REGION:-536870912} "${GLM53_TT_EXTRA_CFG:+, $GLM53_TT_EXTRA_CFG}")
SERVER_ARGS=(--model zai-org/GLM-5.3-Flash --block-size 64 --max-num-seqs $SEQS --max-model-len $MML
  --max-num-batched-tokens ${GLM53_MAX_BATCHED_TOKENS:-2048} --no-enable-prefix-caching --override-generation-config '{"top_k": 32}'
  --max-logprobs -1 --tool-call-parser glm47 --enable-auto-tool-choice --host 127.0.0.1 --port $PORT
  --shutdown-timeout 900 --additional-config "{\"tt\": $TT_CFG}" ${GLM53_EXTRA_SERVER_ARGS:-})
[ -n "${GLM53_SPEC_CONFIG:-}" ] && SERVER_ARGS+=(--speculative-config "$GLM53_SPEC_CONFIG" --no-async-scheduling)
TRK=(); [ "${GLM53_TRACKING:-1}" = 1 ] && TRK=(TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0)
export GLM53_VLLM_EVENTS=$RUN/events.jsonl
{ echo "code=$GLM53_CODE_SHA seqs=$SEQS max_model_len=$MML image=$GLM53_IMAGE_PIN target=${GLM53_TARGET:-default}";
  env | grep -E "^(TT_|VLLM_|MESH_|GLM53_|HF_|PYTHONPATH|LD_LIBRARY_PATH|PROMETHEUS)" | sort;
  printf '%q ' env "${TRK[@]}" python -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}"; echo; } > $RUN/server_cmd.txt
$S/dirty_check.sh table > $RUN/tt_chips.txt 2>/dev/null || true   # the chip table for a post-mortem dirty_check scan

# 4. the standing server under the mesh lock; its pid for the supervisor's SIGINT (serve_window.sh's pattern)
exec $TL run all -- env "${TRK[@]}" bash -c 'echo $$ > "$0"; exec python -m vllm.entrypoints.openai.api_server "$@"' \
  "$PIDFILE" "${SERVER_ARGS[@]}"
