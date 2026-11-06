"""Keep this campaign separate from frozen evidence and the real Git index."""
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from ..d_local_cross_depth_batching import common as prior

REPO=prior.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_gated_loop_continuation'
PREVIOUS=REPO/'poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching'
DEEP=REPO/'poc_flashvep/reports/ouro_huginn_loop_specific_native_split_deep_validation'
BASE='3b3633dbc64bcd916c2ba1ff513deb4226cc647d'
BRANCH='flashvep/adaptive-idle-gated-loop-continuation-poc'
START=datetime(2026,11,6,13,5,11,tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE=START+8*3600
GPU_DEADLINE=START+7*3600+10*60
LONG_START_CUTOFF=GPU_DEADLINE-20*60
write,digest=prior.write,prior.digest


def configure():
    for key in ('ROOT','SOURCE','BASE','BRANCH','START','DEADLINE','GPU_DEADLINE','LONG_START_CUTOFF'):
        setattr(prior,key,globals()[key])
    prior.PREVIOUS=PREVIOUS
    return prior.configure()


def initialize():
    configure();prior.initialize()
    write(ROOT/'state/budget.json',dict(start_epoch=START,hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE,long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=8,timezone='Asia/Seoul',
        scope='New authorized campaign. Audit, GPU tests, analysis, push, and burner restoration share the fixed clock.'))


def save(cfg):
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg,'Immutable configuration differs'
    else:write(path,cfg)


if __name__=='__main__':
    initialize();print(json.dumps(dict(root=str(ROOT),branch=BRANCH,start=START,deadline=DEADLINE)))
