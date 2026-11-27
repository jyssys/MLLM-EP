#!/usr/bin/env bash
set -euo pipefail

eada_runtime=$(realpath "$1")
eada_runs=$(realpath "$2")
eada_section=$3
shift 3
eada_suffix=${1:-}
eada_attention=${EADA_ATTENTION_ENGINE:-eager}
eada_vision=()
if [[ $eada_attention == graph || ${EADA_ENCODER_PIPELINE:-0} == 1 ]]; then
  eada_vision+=(--encoder-pipeline)
fi
if [[ ${EADA_PACKED_VISION:-0} == 1 ]]; then
  eada_vision+=(--vision-backend packed-fa3)
fi
eada_module=poc_flashvep.encoder_assisted_decode_attention
eada_language="$eada_runs/models/qwen25_7b"
eada_images=../mllm_epd_autosetup/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
eada_manifest=../h200_discovery_runs/dataset/grouped_v2/discovery200.json
eada_load1="$eada_runs/native_scan_retry/configs/native_strong_free_1_arrival.json"
eada_load25="$eada_runs/native_profile_q2p5/configs/native_strong_free_2p5_sustained_r2_clean_before_arrival.json"
eada_natural_ratio=.25
export PATH="$eada_runtime/bin:$PATH"
export CUDA_HOME="$eada_runtime/targets/x86_64-linux"
export CC="$eada_runtime/bin/x86_64-conda-linux-gnu-cc" CXX="$eada_runtime/bin/x86_64-conda-linux-gnu-c++"
export FLASHINFER_NVCC="$eada_runtime/bin/nvcc"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 VLLM_USE_FLASHINFER_SAMPLER=0

run_image() {
  local label=$1
  shift
  "$eada_runtime/bin/python" -u -m "$eada_module.online" --model "$eada_images" --manifest "$eada_manifest" \
    --out "$eada_runs/$label$eada_suffix" --attention-engine "$eada_attention" "${eada_vision[@]}" "$@"
}
run_text() {
  local label=$1
  local manifest=$2
  local workload=$3
  shift 3
  "$eada_runtime/bin/python" -u -m "$eada_module.online" --model "$eada_language" --family qwen-text \
    --manifest "$manifest" --workload "$workload" --out "$eada_runs/$label$eada_suffix" \
    --attention-engine "$eada_attention" "$@"
}
case "$eada_section" in
  smoke)
    run_image smoke_local --requests 6 --qps 3 --input-cap 16000 --output 16 --instrument
    run_image smoke_encoder --requests 6 --qps 3 --input-cap 16000 --output 16 --helper encoder --instrument
    "$eada_runtime/bin/python" -u -m "$eada_module.witness" --model "$eada_images" \
      --manifest "$eada_manifest" --out "$eada_runs/model_witness$eada_suffix.json"
    ;;
  eager-pair)
    run_image eager_local_q1 --requests 100 --qps 1 --arrival "$eada_load1"
    run_image eager_encoder_q1 --requests 100 --qps 1 --arrival "$eada_load1" --helper encoder --ratio "$eada_natural_ratio"
    ;;
  graph-witness)
    "$eada_runtime/bin/python" -u -m "$eada_module.paged_graph" --decode-device 2 --helper-device 0 \
      --smoke-output "$eada_runs/paged_graph_witness$eada_suffix.json"
    "$eada_runtime/bin/python" -u -m "$eada_module.witness" --model "$eada_images" \
      --manifest "$eada_manifest" --out "$eada_runs/model_witness_graph$eada_suffix.json" \
      --graph --ratio "$eada_natural_ratio"
    ;;
  packed-witness)
    "$eada_runtime/bin/python" -u -m "$eada_module.witness" --model "$eada_images" \
      --manifest "$eada_manifest" --out "$eada_runs/model_witness_packed$eada_suffix.json" --packed-vision
    ;;
  online)
    run_image img_local_q1 --requests 100 --qps 1 --arrival "$eada_load1"
    run_image img_encoder_q1 --requests 100 --qps 1 --arrival "$eada_load1" --helper encoder --ratio "$eada_natural_ratio"
    run_image img_prefill_q1 --requests 100 --qps 1 --arrival "$eada_load1" --helper prefill --ratio "$eada_natural_ratio"
    run_image img_visual_q1 --requests 100 --qps 1 --arrival "$eada_load1" --helper encoder --ratio "$eada_natural_ratio" --partition visual_first
    run_image img_adaptive_q1 --requests 100 --qps 1 --arrival "$eada_load1" --helper idle-encoder --ratio "$eada_natural_ratio"
    run_image img_ve_2d_q1 --requests 100 --qps 1 --arrival "$eada_load1" --topology dual-decode
    run_image img_local_q25 --requests 200 --qps 2.5 --arrival "$eada_load25"
    run_image img_encoder_q25 --requests 200 --qps 2.5 --arrival "$eada_load25" --helper encoder --ratio "$eada_natural_ratio"
    ;;
  repeat-medium-reverse)
    run_image img_encoder_q25 --requests 200 --qps 2.5 --arrival "$eada_load25" --helper encoder --ratio "$eada_natural_ratio"
    run_image img_local_q25 --requests 200 --qps 2.5 --arrival "$eada_load25"
    ;;
  repeat-medium-forward)
    run_image img_local_q25 --requests 200 --qps 2.5 --arrival "$eada_load25"
    run_image img_encoder_q25 --requests 200 --qps 2.5 --arrival "$eada_load25" --helper encoder --ratio "$eada_natural_ratio"
    ;;
  profile-stages)
    "$eada_runtime/bin/python" -u -m "$eada_module.witness" --model "$eada_images" \
      --manifest "$eada_manifest" --out "$eada_runs/model_witness_profile$eada_suffix.json" \
      --graph --ratio "$eada_natural_ratio" --natural-details --profile-stages
    ;;
  prefill-kv-witness)
    "$eada_runtime/bin/python" -u -m "$eada_module.witness" --model "$eada_images" \
      --manifest "$eada_manifest" --out "$eada_runs/model_witness_prefill$eada_suffix.json" \
      --graph --helper-rank 1 --ratio "$eada_natural_ratio" --natural-details
    ;;
  role-confirm)
    run_image img_local_q1 --requests 100 --qps 1 --arrival "$eada_load1"
    run_image img_encoder_q1 --requests 100 --qps 1 --arrival "$eada_load1" --helper encoder --ratio "$eada_natural_ratio"
    run_image img_prefill_q1 --requests 100 --qps 1 --arrival "$eada_load1" --helper prefill --ratio "$eada_natural_ratio"
    ;;
  fast-vision-replay)
    if [[ ${EADA_PACKED_VISION:-0} != 1 ]]; then
      printf 'Use EADA_PACKED_VISION=1 for the alternate Vision diagnostic.\n' >&2
      exit 2
    fi
    run_image img_packed_local_q1 --requests 100 --qps 1 --arrival "$eada_load1" --output 64
    run_image img_packed_encoder_q1 --requests 100 --qps 1 --arrival "$eada_load1" --output 64 \
      --helper encoder --ratio "$eada_natural_ratio"
    ;;
  text)
    run_text txt_matched_local "$eada_manifest" controlled-text \
      --topology dual-prefill --requests 100 --qps 1 --arrival "$eada_load1" --output 64
    run_text txt_matched_prefill "$eada_manifest" controlled-text \
      --topology dual-prefill --requests 100 --qps 1 --arrival "$eada_load1" --output 64 --helper prefill
    run_text txt_matched_2d "$eada_manifest" controlled-text \
      --topology dual-decode --requests 100 --qps 1 --arrival "$eada_load1" --output 64
    run_image mm_text_local --workload controlled-text --requests 100 --qps 1 --arrival "$eada_load1" --output 64
    run_image mm_text_encoder --workload controlled-text --requests 100 --qps 1 --arrival "$eada_load1" \
      --output 64 --helper encoder
    run_image mm_image_replay_local --requests 100 --qps 1 --arrival "$eada_load1" --output 64
    run_image mm_image_replay_encoder --requests 100 --qps 1 --arrival "$eada_load1" --output 64 --helper encoder
    run_text txt_real_local "$eada_runs/longbench/qwen_128.json" real-text --topology dual-prefill --requests 128 --qps 1
    run_text txt_real_prefill "$eada_runs/longbench/qwen_128.json" real-text --topology dual-prefill --requests 128 \
      --qps 1 --helper prefill
    ;;
  capacity)
    run_image pressure_local --requests 128 --qps 6 --output 128 --images many --d-gib 4
    run_image pressure_encoder --requests 128 --qps 6 --output 128 --images many --d-gib 4 --helper encoder
    run_image decode256_local --requests 32 --qps 6 --output 256 --images many --d-gib 24 --max-batch 16
    run_image decode256_encoder --requests 32 --qps 6 --output 256 --images many --d-gib 24 --max-batch 16 \
      --helper-gib 24 --helper encoder
    ;;
  tight-capacity)
    run_image pressure2_local --requests 128 --qps 6 --output 128 --images many --d-gib 2 --helper-gib 4
    run_image pressure2_encoder --requests 128 --qps 6 --output 128 --images many --d-gib 2 --helper-gib 4 \
      --helper encoder
    ;;
  stock-dp)
    "$eada_runtime/bin/python" -u -m "$eada_module.stock_dp" --model "$eada_images" --manifest "$eada_manifest" \
      --arrival "$eada_load1" --out "$eada_runs/stock_dp3_q1$eada_suffix" --requests 100
    ;;
  *)
    printf 'Unknown study section: %s\n' "$eada_section" >&2
    exit 2
    ;;
esac
