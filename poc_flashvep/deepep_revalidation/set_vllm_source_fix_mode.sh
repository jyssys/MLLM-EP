#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: set_vllm_source_fix_mode.sh none|attention|deepstack|both}
case "${MODE}" in
  none|attention|deepstack|both) ;;
  *) echo "invalid mode: ${MODE}" >&2; exit 2 ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PATCH_DIR="${REPO_ROOT}/poc_flashvep/deepep_revalidation/patches"
VLLM_SITE=${VLLM_SITE:-/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages}
ATTN_PATCH="${PATCH_DIR}/vllm_0_20_dbo_attention_cache_ubatch.patch"
DEEPSTACK_PATCH="${PATCH_DIR}/vllm_0_20_qwen3vl_deepstack_ubatch.patch"
RUNNER="${VLLM_SITE}/vllm/v1/worker/gpu_model_runner.py"
WRAPPER="${VLLM_SITE}/vllm/v1/worker/gpu_ubatch_wrapper.py"

if rg -q 'cache_key = \(ubid, kv_cache_spec, type\(builder\)\)' "${RUNNER}"; then
  patch --batch -R -p1 -d "${VLLM_SITE}" < "${ATTN_PATCH}"
fi
if rg -q '"ubatch_token_slice": ubatch_slice\.token_slice' "${WRAPPER}"; then
  patch --batch -R -p1 -d "${VLLM_SITE}" < "${DEEPSTACK_PATCH}"
fi

case "${MODE}" in
  attention)
    patch --batch -p1 -d "${VLLM_SITE}" < "${ATTN_PATCH}"
    ;;
  deepstack)
    patch --batch -p1 -d "${VLLM_SITE}" < "${DEEPSTACK_PATCH}"
    ;;
  both)
    patch --batch -p1 -d "${VLLM_SITE}" < "${ATTN_PATCH}"
    patch --batch -p1 -d "${VLLM_SITE}" < "${DEEPSTACK_PATCH}"
    ;;
esac

python3 -m py_compile \
  "${VLLM_SITE}/vllm/v1/worker/gpu_model_runner.py" \
  "${VLLM_SITE}/vllm/v1/worker/gpu_ubatch_wrapper.py" \
  "${VLLM_SITE}/vllm/model_executor/models/qwen3_vl.py"

printf 'source_fix_mode=%s\n' "${MODE}"
sha256sum \
  "${VLLM_SITE}/vllm/v1/worker/gpu_model_runner.py" \
  "${VLLM_SITE}/vllm/v1/worker/gpu_ubatch_wrapper.py" \
  "${VLLM_SITE}/vllm/model_executor/models/qwen3_vl.py"
