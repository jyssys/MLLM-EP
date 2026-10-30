"""Immutable correctness release and a separate RCC evidence namespace."""
from pathlib import Path
from ..looped_pd_abc.common import MODELS,MODEL_PINS,RLT,PYTHON,UUIDS,write,digest,environment

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'poc_flashvep/reports/rcc_pd'
CF_ROOT=REPO/'poc_flashvep/reports/looped_pd_b_numerical_diagnosis'
CF_COMMIT='3b7a722898dab5f6dd797b474c1ff461659d8fa1'
CUTS={'ouro':2,'huginn':16}
VARIANTS={'ouro':'down_norm','huginn':'down_norm_adapter'}
OWNED=('poc_flashvep/rcc_pd_poc/', 'poc_flashvep/reports/rcc_pd/',
       'poc_flashvep/reports/rcc_pd_existing_a_scheduler_audit.md',
       'poc_flashvep/reports/rcc_pd_fixed_cut_cobatching_results.md',
       'poc_flashvep/reports/rcc_pd_slo_admission_results.md',
       'poc_flashvep/reports/rcc_pd_poc_final_summary.md')


def setup():
    for name in ('raw','configs','logs','state','data','figures'):(ROOT/name).mkdir(parents=True,exist_ok=True)


def append(path,value):
    import json
    with path.open('a') as stream:stream.write(json.dumps(value,default=str)+'\n')
