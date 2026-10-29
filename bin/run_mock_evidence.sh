#!/usr/bin/env bash
# run_mock_evidence.sh — the host-only evidence of the supervisor: unit tests (policy state machine, Prometheus
# parser, plugin step meter) + the end-to-end mock harness (fake server driven through every recovery path).
# Output: logs/mock_evidence_<UTC>.log (pytest -v) and logs/mock_evidence_latest.log. No device, no vLLM needed.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
ts=$(date -u +%Y%m%dT%H%M%SZ); out=logs/mock_evidence_$ts.log
{ echo "supervisor mock evidence $ts host=$(hostname) python=$(python3 --version 2>&1) branch=$(git rev-parse --abbrev-ref HEAD) rev=$(git rev-parse --short HEAD)$( [ -n "$(git status --porcelain -- bin src)" ] && echo -dirty)"
  python3 -m pytest -v -p no:cacheprovider bin/tests 2>&1; echo "pytest rc=${PIPESTATUS[0]}"; } | tee $out
ln -sf $(basename $out) logs/mock_evidence_latest.log
tail -1 $out
