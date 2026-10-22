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
TASK_PYTHON=/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python
"$TASK_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_slo.dispatch \
  --gpus 4 --manifest poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/solo_manifest_v2.json
"$TASK_PYTHON" -u -m poc_flashvep.ouro_chunk_loop_slo.dispatch \
  --gpus 4,5,6,7 --manifest poc_flashvep/reports/ouro_chunk_loop_slo_sensitivity_discovery/parallel_calibration_manifest_v2.json
