"""New campaign namespace. Shared helpers do not change historical artifacts."""
from pathlib import Path
from ..ouro_adaptive_chunk_loop.common import REPO,MODEL,PYTHON,UUIDS,UPSTREAM,digest,write,append,guard,config

ROOT=REPO/'poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_discovery'
PRIOR=REPO/'poc_flashvep/reports/ouro_8gpu_deadline_aware_joint_chunk_loop_discovery'
FORK=Path('/home/esjung/external/ouro-budget-joint-rlt')
SPEC=REPO/'poc_flashvep/reports/ouro_8gpu_compute_budget_joint_cg_optimization_poc_spec.md'

CHUNKS=(64,128,256,512,768,1024,1536,2048)
GROUPS=(1,2,3,4)
BUDGETS=(10,20,40,60,80,120,160,240)
