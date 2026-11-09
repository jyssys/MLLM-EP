"""Keep the new clock and evidence separate from frozen campaigns."""
import json
import os
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from ..adaptive_idle_loop_continuation import common as prior

REPO=prior.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_intervention_dynamic_loop'
PREVIOUS=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_gated_loop_continuation'
DEEP=REPO/'poc_flashvep/reports/ouro_huginn_loop_specific_native_split_deep_validation'
BASE='ddbf49ae3ef610fefd78248d765a27641b21b98f'
BRANCH='flashvep/adaptive-idle-intervention-dynamic-loop-poc'
START=datetime(2026,11,8,18,7,15,tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
DEADLINE=START+14*3600
GPU_DEADLINE=START+12.5*3600
LONG_START_CUTOFF=GPU_DEADLINE-30*60
RESUMPTION=None
COMPLETION_WINDOW=None
if os.environ.get('IDLE_LOOP_RESUMPTION'):
    resume_path=Path(os.environ['IDLE_LOOP_RESUMPTION']).resolve()
    assert resume_path.parent==ROOT/'state', 'Resumption record must belong to this campaign'
    RESUMPTION=json.loads(resume_path.read_text())
    assert RESUMPTION['authorization']=='User explicitly requested completion of the unfinished PoC on 2026-11-09'
    START=datetime.fromisoformat(RESUMPTION['start']).timestamp()
    DEADLINE=datetime.fromisoformat(RESUMPTION['final_deadline']).timestamp()
    GPU_DEADLINE=datetime.fromisoformat(RESUMPTION['GPU_deadline']).timestamp()
    LONG_START_CUTOFF=GPU_DEADLINE-20*60
    assert START<LONG_START_CUTOFF<GPU_DEADLINE<DEADLINE<=START+4*3600
if os.environ.get('IDLE_LOOP_COMPLETION_WINDOW'):
    window_path=Path(os.environ['IDLE_LOOP_COMPLETION_WINDOW']).resolve()
    assert RESUMPTION is not None and window_path.parent==ROOT/'state'
    COMPLETION_WINDOW=json.loads(window_path.read_text())
    assert COMPLETION_WINDOW['GPU_deadline']==GPU_DEADLINE and COMPLETION_WINDOW['final_deadline']==DEADLINE
    LONG_START_CUTOFF=COMPLETION_WINDOW['maximum_controller_cutoff']
    assert LONG_START_CUTOFF==GPU_DEADLINE-180
OPERATIONAL=Path('/home/esjung/vllm-ep/burner_runs/restart_20261108_175646')
write,digest=prior.write,prior.digest


def configure():
    for key in ('ROOT','SOURCE','BASE','BRANCH','START','DEADLINE','GPU_DEADLINE','LONG_START_CUTOFF'):
        setattr(prior,key,globals()[key])
    prior.PREVIOUS=PREVIOUS
    result=prior.configure()
    from ..looped_pd_autonomous_headroom.common import resources
    resources().APPROVED_LOGS.add(OPERATIONAL/'burner.log')
    return result


def initialize():
    frozen_budget=ROOT/'state/budget.json'
    original_budget=json.loads(frozen_budget.read_text()) if RESUMPTION and frozen_budget.exists() else None
    configure();prior.initialize()
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources();verified=owner.burners()
    for file in (OPERATIONAL/'state').glob('burner_*.json'):
        value=json.loads(file.read_text());pid=value['supervisor']
        try:
            proc=Path('/proc',str(pid));words=(proc/'cmdline').read_bytes().decode().split('\0')
            if (set(value['GPUs'])<=set(verified) and proc.stat().st_uid==os.getuid()
                    and os.getsid(pid)==pid and owner.SCRIPT in words and '--worker' not in words):
                target=ROOT/'state'/file.name
                if not target.exists():write(target,dict(value,adopted_from=str(file)))
        except (FileNotFoundError,ProcessLookupError):pass
    budget=dict(start_epoch=START,hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE,long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=4 if RESUMPTION else 14,timezone='Asia/Seoul',
        bounded_completion_window=COMPLETION_WINDOW,
        scope='Explicit bounded resumption. The original expired clock remains frozen.' if RESUMPTION else
        'Actual intervention and online negotiation have priority. Perfect kernel tracing is not a prerequisite. Pauses do not reset this clock.')
    if RESUMPTION:
        if original_budget is not None:write(frozen_budget,original_budget)
        write(ROOT/'state/resumption_active_budget.json',budget)
    else:write(frozen_budget,budget)


def save(cfg):
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg,'Immutable configuration changed'
    else:write(path,cfg)


if __name__=='__main__':
    initialize();print(json.dumps(dict(root=str(ROOT),branch=BRANCH,start=START,deadline=DEADLINE)))
