#!/usr/bin/env bash
# Run from the repository root inside an authorized and leased GPU environment.
set -euo pipefail

study=poc_flashvep.pd_resource_complementary_parallelism
root="${PD_RUN_ROOT:?Set PD_RUN_ROOT to the artifact and frozen trace directory}"
py="${PD_PYTHON:-python}"
model="${PD_MODEL:-$root/models/qwen3-8b}"
trace="$root/frozen"
case_for() {
    local name="$1" geometry="$2" cohort="$3" qps="$4"
    printf '%s' "$name:$trace/$geometry-n$cohort.json:$qps"
}
near8=(--case "$(case_for 8k_near 8192x16 64 4.5)")
near16=(--case "$(case_for 16k_near 16384x16 64 1.9)")
over16=(--case "$(case_for 16k_over 16384x16 64 2.6)")
common=("${near8[@]}" "${near16[@]}" "${over16[@]}")
run() {
    local out="$1"
    shift
    "$py" -m "$study.native" --model "$model" --out "$root/$out" "$@"
}

case "${1:?Supply the study arm}" in
  p-reference)
    budget="${2:-8192}"
    run "p-tuned/pb$budget" --mode prefill --batch-tokens "$budget" \
      --case "$(case_for 8k_high 8192x16 24 12)" \
      --case "$(case_for 16k_high 16384x16 24 12)" \
      --case "$(case_for 32k_high 32768x16 24 8)"
    ;;
  native)
    run native-pd/online \
      --case "$(case_for 8k_low 8192x16 24 2)" "${near8[@]}" \
      --case "$(case_for 8k_over 8192x16 64 6)" \
      --case "$(case_for 16k_low 16384x16 24 .8)" "${near16[@]}" "${over16[@]}" \
      --case "$(case_for 32k_low 32768x16 24 .45)" \
      --case "$(case_for 32k_near 32768x16 24 .8)" \
      --case "$(case_for 32k_over 32768x16 24 1.15)" \
      --case "$(case_for 4k_o64_mix 4096x64 24 4)" \
      --case "$(case_for 2k_o256_decode 2048x256 24 4)" \
      --case "$(case_for 16k_o64 16384x64 24 1.9)" \
      --case "$(case_for 16k_trace 16384x16 24 2.6)" --profile-case 16k_trace
    ;;
  p-light)
    run dense-controls/p-light --activity-library "${PD_ACTIVITY_LIBRARY:?Set the compiled CUPTI adapter path}" \
      --profile-case 16k_light --shape-case 8k_near \
      --case "$(case_for 16k_light 16384x16 24 2.6)" "${common[@]}"
    ;;
  steady)
    run native-steady/online --activity-library "${PD_ACTIVITY_LIBRARY:?Set the compiled CUPTI adapter path}" \
      --batch-tokens "${2:-8192}" --profile-case 16k_steady \
      --case "$(case_for 16k_steady 16384x16 64 2.6)" "${near16[@]}" \
      --case "$(case_for mixed_heavy mixed-heavy 64 1.45)" \
      --case "$(case_for mixed mixed 24 1.25)"
    ;;
  steady-p-only)
    run native-p-only/online --activity-library "${PD_ACTIVITY_LIBRARY:?Set the compiled CUPTI adapter path}" \
      --profile-case 16k_p_only --profile-prefill-only \
      --case "$(case_for 16k_p_only 16384x16 64 2.6)"
    ;;
  dp2)
    run dense-controls/dp2 --mode dp --decode-workers 2 "${common[@]}" \
      --case "$(case_for 32k_near 32768x16 24 .8)"
    ;;
  dp2-small-chunks)
    budget="${2:-2048}"
    run "dense-controls/dp2-chunk$budget" --mode dp --decode-workers 2 \
      --batch-tokens "$budget" "${common[@]}"
    ;;
  dp2-no-chunk)
    run dense-controls/dp2-no-chunk --mode dp --decode-workers 2 --batch-tokens 33024 --no-chunk "${common[@]}"
    ;;
  pd-cross)
    run dense-controls/pd-cross --cross-layer-blocks "${common[@]}"
    ;;
  pd-layout-probe)
    VLLM_KV_CACHE_LAYOUT=HND run dense-controls/pd-layout-probe --cross-layer-blocks \
      --case "$(case_for 8k_low 8192x16 24 2)"
    ;;
  pd-deflect)
    run dense-controls/pd-deflect --cross-layer-blocks --deflect-at 2 "${common[@]}"
    ;;
  p_tp2_d1)
    run native-static-3gpu/p_tp2_d1 --p-tp 2 "${common[@]}"
    ;;
  p2_d1)
    run native-static-3gpu/p2_d1 --prefill-workers 2 "${common[@]}"
    ;;
  p1_d2)
    run native-static-3gpu/p1_d2 --decode-workers 2 "${common[@]}"
    ;;
  *)
    printf '%s\n' "Unknown study arm $1" >&2
    exit 2
    ;;
esac
