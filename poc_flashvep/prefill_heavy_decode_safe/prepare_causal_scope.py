"""Retain every missing timestamp and require causal exclusion proof."""
import json
from .common import *


def main():
    initialize();old=json.loads((ROOT/'state/graph_before_model_18k_plan.json').read_text());jobs=[]
    for parent in old['all_jobs']:
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('graph_before_model_','graph_causal_scope_')
        cfg.update(name=name,activity_measurement_start_device_fence=True,activity_causal_record_scope=True,
            phase='GRAPH_DRIVER_CORRELATION_AND_DEVICE_FENCE_SCOPE_CONTROL')
        if parent==old['state8']:cfg['required_qualification']='graph_before_model_ouro_L18432_observer_state8'
        elif cfg.get('discovery_activity'):cfg['required_qualification']=old['state8'].replace('graph_before_model_','graph_causal_scope_')
        save(cfg);jobs.append(name)
    path=ROOT/'state/graph_causal_scope_18k_plan.json'
    write(path,dict(state8=jobs[0],all_jobs=jobs,clean_reference=jobs[1],jobs=jobs[2:],
        scope='Global missing records remain in the invalid CSV and counters. A matching CPU launch after capture, or a matching launch before a measured pre-capture device fence, can prove exclusion. Every unresolved missing timestamp remains fatal. All latency screens remain unchanged.'))
    old=json.loads((ROOT/'state/ouro_18k_confirmation_v4_plan.json').read_text());out=[];plans=[]
    for parent in old['jobs']:
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
        if cfg['state_probe']:out.append(parent);continue
        name=parent.replace('confirm_v4_','confirm_scope_')
        cfg.update(name=name,activity_measurement_start_device_fence=True,activity_causal_record_scope=True)
        if cfg.get('discovery_activity'):cfg['required_qualification']=jobs[0]
        save(cfg);out.append(name)
    for p in old['plans']:
        original=json.loads(Path(p).read_text());path=Path(p).with_name(Path(p).name.replace('confirm_v4_','confirm_scope_'))
        write(path,dict(clean_reference=original['clean_reference'].replace('confirm_v4_','confirm_scope_'),
            jobs=[n.replace('confirm_v4_','confirm_scope_') for n in original['jobs']]))
        plans.append(str(path))
    write(ROOT/'state/ouro_causal_confirmation_plan.json',dict(jobs=out,plans=plans))
    print(' '.join(jobs))


if __name__=='__main__':main()
