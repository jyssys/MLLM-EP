"""Keep this experiment separate from the frozen request-PP evidence."""
import json
import os
from pathlib import Path
from poc_flashvep.ouro_request_pp.contract import MODEL,REVISION,GPUS,UUIDS,sha,write

PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'poc_flashvep/reports/ouro_tp4_parallelism_discovery'
OLD=PROJECT/'poc_flashvep/reports/ouro_request_pp_discovery'
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_tp4_parallelism_discovery_spec.md')
PYTHON='/home/esjung/.venvs/llada2-mini-reference/bin/python'
VLLM_PYTHON='/home/esjung/.venvs/cp-composition-vllm026/bin/python'
# Frozen before the first TP probe. No post-failure tolerance increase.
TOL=dict(logit_max_abs=.75,logit_mean_abs=.08,hidden_relative_L2=.03,KV_relative_L2=.03,
         operator_relative_L2=.03,stable_native_margin=.5,stable_argmax_mismatches=0)


def mask():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4,5,6,7'


def cases():return json.loads((ROOT/'workload.json').read_text())['cases']


def rows(case):return case['requests']
