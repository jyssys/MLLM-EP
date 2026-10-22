#!/usr/bin/env bash
set -euo pipefail
restore_burner() {
  cd /home/esjung/MLLM-EP-ouro-request-pp
  /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_8arm.lease restore --gpus 4
}
trap restore_burner EXIT
cd /home/esjung/MLLM-EP-github
export CUDA_VISIBLE_DEVICES=4
export OURO_PREFILL_BACKEND=flash_attn_4
export PYTHONPATH=/home/esjung/external/ouro-chunk-loop-prefill-rlt
POC_PYTHON=/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python
for start in 0 1 2; do
  case "$start" in
    0) trials=2,3,4,5 ;;
    1) trials=6,7,8 ;;
    2) trials=9,10,11 ;;
  esac
  "$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.bench \
    --label "fa4_confirm_warm_s${start}" --trials "$trials" \
    --cells atomic:1024,split:1024,mixed:128,mixed:1024,grouped:1024 \
    --shapes 2048:4,8192:8 --reuse-engine --warmup-trial 0
done
"$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.bench \
  --label fa4_trace_v1 --trials 2 --cells atomic:1024,mixed:1024,grouped:1024,staged4:1024 \
  --shapes 2048:4 --trace --reuse-engine --warmup-trial 0
"$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.bench \
  --label fa4_observer_off_v1 --trials 2 --cells atomic:1024,mixed:1024,grouped:1024,staged4:1024 \
  --shapes 2048:4 --no-record-quanta --reuse-engine --warmup-trial 0
"$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.bench \
  --label fa4_graph_off_v1 --trials 2 --cells atomic:1024,grouped:1024 \
  --shapes 2048:4 --no-graphs --reuse-engine --warmup-trial 0
"$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.bench \
  --label fa4_prefill_only_v1 --trials 2 --cells atomic:1024,split:1024,grouped:1024 \
  --shapes 2048:1 --reuse-engine --warmup-trial 0
"$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.decode_control
if nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --force-overwrite=false -o /home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_chunk_loop_prefill_joint_discovery/raw/nsys_grouped \
  "$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.bench \
  --label nsys_grouped_v1 --trials 2 --cells grouped:1024 --shapes 2048:4 --trace; then
  printf '%s\n' 'NSYS_CAPTURE_COMPLETED'
else
  printf '%s\n' 'NSYS_CAPTURE_FAILED: preserve logs and continue correctness probe' >&2
fi
"$POC_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_prefill.smoke \
  --label fa4_8k_correctness_v1 --lengths 8192 --chunk 1024 --modes atomic,grouped
cd /home/esjung/MLLM-EP-ouro-request-pp
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_8arm.lease restore --gpus 4
