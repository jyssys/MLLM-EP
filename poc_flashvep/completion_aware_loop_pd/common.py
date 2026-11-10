"""Isolate the campaign and reuse the qualified GPU owner."""
import json
import os
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from ..idle_intervention_dynamic_loop import common as prior

REPO=prior.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation'
OLD=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_intervention_dynamic_loop'
BASE='0adb7491c866d6d66cf71ea30dca63f82c46d135'
BRANCH='flashvep/completion-aware-adaptive-loop-pd'
START=datetime(2026,11,10,14,51,19,tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE=START+8*3600
GPU_DEADLINE=DEADLINE-45*60
LONG_START_CUTOFF=GPU_DEADLINE-20*60
write,digest=prior.write,prior.digest


def configure():
    for key in ('ROOT','SOURCE','BASE','BRANCH','START','DEADLINE','GPU_DEADLINE','LONG_START_CUTOFF'):
        setattr(prior,key,globals()[key])
    result=prior.configure()
    from ..looped_pd_autonomous_headroom.common import resources
    resources().APPROVED_LOGS.add(OLD/'burner.log')
    return result


def initialize():
    configure();prior.initialize()
    write(ROOT/'state/budget.json',dict(start_epoch=START,hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE,long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=8,scope='Operational campaign bound. The current user request specifies no new numerical time limit. Publication and restoration remain inside this bound.'))


def save(cfg):
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg,'Immutable configuration changed'
    else:write(path,cfg)


if __name__=='__main__':
    initialize();print(json.dumps(dict(root=str(ROOT),branch=BRANCH)))
