"""Retry a source-build mismatch in new namespaces, with the guard intact."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .profile_controls import analyze
from .host_screen import screen


def prepare():
    old=json.loads((ROOT/'state/huginn_physical_confirmation_plan.json').read_text())
    state='confirm_huginn_L3072_observer_state8_v3';cfg=json.loads((ROOT/'configs'/(old['jobs'][0]+'.json')).read_text())
    cfg.update(name=state,activity_full_device_sync_after_outputs=True,
        phase='CURRENT_BUILD_AND_ALL_STREAM_SYNC_EXACT_STATE_RECHECK')
    save(cfg);plans=[]
    for path in old['plans']:
        p=json.loads(Path(path).read_text());jobs=[]
        for parent in p['jobs']:
            cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('confirm_','confirm_v3_')
            cfg.update(name=name,activity_full_device_sync_after_outputs=True)
            if cfg.get('discovery_activity'):cfg['required_qualification']=state
            save(cfg);jobs.append(name)
        path=Path(path).with_name(Path(path).name.replace('confirm_','confirm_v3_'))
        write(path,dict(clean_reference=p['clean_reference'],jobs=jobs));plans.append(str(path))
    write(ROOT/'state/huginn_physical_resume_plan.json',dict(state8=state,plans=plans,
        reason='The prior attempt stopped before launching a GPU worker. A concurrent source edit and collector build triggered the source/library hash guard. The guard remains enabled. Configurations and failed source snapshots remain available.'))
    return state,plans


def main():
    initialize();state,plans=prepare()
    while not (ROOT/'state/graph_sync_18k_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    if sequence([state],'huginn_resume_observer_state'):
        for i,path in enumerate(plans):
            cfg=json.loads(Path(path).read_text())
            if not sequence(cfg['jobs'],f'huginn_resumed_physical_confirmation_{i}'):break
            value=analyze(path)
            for row in value['observers']:
                if row['trace']:
                    t=row['trace']['prefill'];screen(row['name'],window=(t['left'],t['right']))
    write(ROOT/'state/huginn_resumed_physical_confirmation_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
