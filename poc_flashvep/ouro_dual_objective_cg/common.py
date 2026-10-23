"""Campaign paths and strict physical-device scope."""
import os
from pathlib import Path
from ..ouro_budget_joint_cg.common import REPO, MODEL, PYTHON, FORK, UUIDS, digest, write, append

PARENT = '821b11653695d0a470bd88bfcc997ae1bbcf5bc9'
ROOT = REPO / 'poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_discovery'
PRIOR = REPO / 'poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery'
SPEC = REPO / 'poc_flashvep/reports/ouro_dual_objective_cg_slo_workload_poc_spec.md'
GPUS = (4, 5, 6, 7)

def guard():
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert value in tuple(map(str, GPUS)), ('Unauthorized device mask', value)
    return int(value)
