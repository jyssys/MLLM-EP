#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/external/lmms-eval:${PYTHONPATH:-}"
export HF_HOME="${ROOT_DIR}/data/hf_cache"

python3 -m lmms_eval eval \
  --model dummy \
  --tasks mmmu_val \
  --limit 1 \
  --batch_size 1 \
  --device cpu \
  --predict_only \
  --log_samples \
  --output_path "${ROOT_DIR}/outputs/lmms_dry_run_mmmu" \
  --verbosity ERROR

