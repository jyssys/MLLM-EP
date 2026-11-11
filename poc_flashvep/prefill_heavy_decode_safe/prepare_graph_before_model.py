"""Enable graph tracing before any model constructor can instantiate a graph."""
import json
from .common import *


def main():
    initialize();old=json.loads((ROOT/'state/graph_sync_18k_plan.json').read_text());jobs=[]
    for parent in old['all_jobs']:
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('graph_sync_','graph_before_model_')
        cfg.update(name=name,activity_before_model=True,
            phase='PRE_MODEL_GRAPH_TRACE_SELECTION_MATCHED_NATIVE_CONTROL')
        if parent==old['state8']:cfg['required_qualification']='graph_sync_ouro_L18432_observer_state8'
        elif cfg.get('discovery_activity'):cfg['required_qualification']=old['state8'].replace('graph_sync_','graph_before_model_')
        save(cfg);jobs.append(name)
    path=ROOT/'state/graph_before_model_18k_plan.json'
    write(path,dict(state8=jobs[0],all_jobs=jobs,clean_reference=jobs[1],jobs=jobs[2:],
        scope='Graph mode now precedes the model constructor, not only StageEngine. Missing timestamps remain fatal. This tests a documented graph-instantiation ordering requirement, not an empty-graph assumption.'))
    write(ROOT/'state/current_ouro_observer_choice.json',dict(plan=str(path)))
    old=json.loads((ROOT/'state/ouro_18k_confirmation_plan.json').read_text());jobs=[];plans=[]
    for parent in old['jobs']:
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('confirm_','confirm_v4_')
        cfg.update(name=name,activity_before_model=True,activity_full_device_sync_after_outputs=True)
        if '_capture' in name:cfg['required_qualification']='graph_before_model_ouro_L18432_observer_state8'
        elif '_state32' not in name:cfg['required_qualification']=old['jobs'][0].replace('confirm_','confirm_v4_')
        save(cfg);jobs.append(name)
    for p in old['plans']:
        c=json.loads(Path(p).read_text());path=Path(p).with_name(Path(p).name.replace('confirm_','confirm_v4_'))
        write(path,dict(clean_reference=c['clean_reference'].replace('confirm_','confirm_v4_'),
            jobs=[n.replace('confirm_','confirm_v4_') for n in c['jobs']]))
        plans.append(str(path))
    write(ROOT/'state/ouro_18k_confirmation_v4_plan.json',dict(jobs=jobs,plans=plans))
    print(' '.join(jobs))


if __name__=='__main__':main()
