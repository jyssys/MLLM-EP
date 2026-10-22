#!/usr/bin/env bash
set -euo pipefail
restore_all() {
  cd /home/esjung/MLLM-EP-ouro-request-pp
  /home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_8arm.lease restore --gpus 4,5,6,7
}
trap restore_all EXIT
cd /home/esjung/MLLM-EP-ouro-request-pp
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_8arm.lease stop --gpus 4,5,6,7
cd /home/esjung/MLLM-EP-github
OURO_SLO_QUEUE=$(CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_chunk_loop_slo.resolve_manifest --manifest "$1")
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -u -m poc_flashvep.ouro_chunk_loop_slo.dispatch --manifest "$OURO_SLO_QUEUE"
case "$OURO_SLO_QUEUE" in
  *separate_profiling_manifest*)
    /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -u -m poc_flashvep.ouro_chunk_loop_slo.dispatch --gpus 4 --manifest poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/solo_policy_verification_manifest.json
    restore_all
    trap - EXIT
    cd /home/esjung/MLLM-EP-github
    CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -u -m poc_flashvep.ouro_chunk_loop_slo.finish_analysis
    ;;
esac
