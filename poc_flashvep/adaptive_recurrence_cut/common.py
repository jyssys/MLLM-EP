"""Pins and an independent experiment namespace."""
import json
from pathlib import Path
from ..looped_pd_abc.common import REPO,RLT,PYTHON,MODELS,MODEL_PINS,UUIDS,environment,write,digest

ROOT=REPO/'poc_flashvep/reports/adaptive_recurrence_cut'
PRIOR=REPO/'poc_flashvep/reports/rcc_pd'
NUMERICAL=REPO/'poc_flashvep/reports/looped_pd_b_numerical_diagnosis'
BASE='b53a8fed36a1706ac74754b53753f20edb8b913e'
CORRECTNESS='3b7a722898dab5f6dd797b474c1ff461659d8fa1'
CUTS={'ouro':[0,1,2,3,4],'huginn':[0,4,8,16,24,28,32]}
VARIANTS={'ouro':'down_norm','huginn':'down_norm_adapter'}
OWNED=('poc_flashvep/adaptive_recurrence_cut/', 'poc_flashvep/reports/adaptive_recurrence_cut/',
       'poc_flashvep/reports/adaptive_recurrence_cut_vs_token_layer_split_poc_spec.md',
       'poc_flashvep/reports/adaptive_recurrence_cut_static_map.md',
       'poc_flashvep/reports/adaptive_recurrence_cut_oracle_headroom.md',
       'poc_flashvep/reports/adaptive_recurrence_cut_token_loop_layer_comparison.md',
       'poc_flashvep/reports/adaptive_recurrence_cut_online_policy_results.md',
       'poc_flashvep/reports/adaptive_recurrence_cut_final_summary.md')


def setup():
    for name in ('raw','data','configs','logs','state','figures'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)


def append(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:stream.write(json.dumps(value,default=str)+'\n')


def load_config():
    import os
    return json.loads(Path(os.environ['ARC_CONFIG']).read_text())
