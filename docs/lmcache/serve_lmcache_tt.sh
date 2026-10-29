#!/usr/bin/env bash
# Launch recipe (A2): vLLM 0.26.0 + vllm-tt-plugin + LMCache (host RAM + NVMe tiers) for GLM-5.3-Flash.
#
# STATUS 2026-10-20: a recipe, not a validated launch. It needs (1) A1's plugin-side connector host (KV spec per DSA
# layer, register_kv_caches, set_host_xfer_buffer_ops(CopyBlocksOp), start_load_kv / wait_for_layer_load /
# save_kv_layer / wait_for_save in the model runner) and (2) a device window under the mesh scheduler + tt-lock.
# Host-validated today: the config loads, LMCache selects the TT connector on a CPU-only torch, and the connector
# round-trips pages through the CPU and disk tiers (tests/lmcache/test_tt_*.py in /opt/glm53/venv-lmcache).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE:-$HERE/lmcache_tt.yaml}
mkdir -p /opt/glm53/lmcache-disk
# Device environment: the Stage 9 serving env (tree, mesh, timeouts) with venv-lmcache instead of venv-vllm.
# shellcheck disable=SC1091
source /opt/glm53/kit/doc/ttnn/09-vllm/scripts/env.sh
export VIRTUAL_ENV=/opt/glm53/venv-lmcache PATH=/opt/glm53/venv-lmcache/bin:$PATH
export VLLM_TT_PLUGIN_ROOT=/opt/glm53/vllm-tt-plugin-lmcache-a2
SNAP=${GLM53_HF_SNAPSHOT:-/opt/glm53/hf/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a}
exec vllm serve "$SNAP" --served-model-name zai-org/GLM-5.3-Flash --host 127.0.0.1 --port "${GLM53_PORT:-18000}" \
  --block-size 64 --max-num-seqs "${GLM53_MAX_SEQS:-32}" --max-model-len "${GLM53_MAX_MODEL_LEN:-2051}" \
  --max-num-batched-tokens "${GLM53_MAX_BATCHED_TOKENS:-2048}" --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_mode": "all", "enable_model_warmup": true, "prefill_chunk_alignment": 128}}' \
  "$@"
