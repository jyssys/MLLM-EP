#!/usr/bin/env bash
set -euo pipefail

readonly POC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cat >&2 <<'EOF'
FlashVEP Phase 1 profiling is blocked before model loading.

The required seven-GPU TP/EP baseline is invalid in vLLM 0.20.0 because the
Qwen3-VL model has 32 attention heads and TP=7 requires divisibility by 7.
No profiling flag, CUDA event, NVTX range, or timing output will be activated.
Run poc_flashvep/baseline_command.sh only to reproduce the configuration error.
EOF

echo "Evidence: ${POC_ROOT}/poc_flashvep/reports/phase1_profile.md" >&2
exit 3
