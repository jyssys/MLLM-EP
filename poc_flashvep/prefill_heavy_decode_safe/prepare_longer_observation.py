"""Freeze wider analysis windows before either Native experiment starts."""
import json,time
from .common import *


def replace_plan(path,seconds,keep_first=False):
    old=json.loads(path.read_text());audit=path.with_name(path.stem+'_before_longer_window.json')
    if old.get('prospective_observation_seconds')==seconds:return
    assert not audit.exists()
    all_jobs=old['all_jobs'] if 'all_jobs' in old else old['jobs']
    jobs=[]
    for index,parent in enumerate(all_jobs):
        assert not (ROOT/'raw'/parent/'command.json').exists(),'Do not change an active experiment plan'
        if keep_first and index==0:jobs.append(parent);continue
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent+f'_w{seconds}'
        cfg.update(name=name,activity_window_seconds=float(seconds),
            observation_selection_reason='Freeze a longer same-run interval before seeing its results. This reduces sensitivity of queue-half growth to a short Poisson burst. Actual collection already extends to shutdown, so this changes the retained interval rather than adding timed tracing.')
        assert cfg['activity_window_fraction']*cfg['arrival_offsets'][-1]+seconds<cfg['arrival_offsets'][-1], 'Retain the observation before the final offered arrival'
        save(cfg);jobs.append(name)
    write(audit,old)
    new=dict(old,prospective_observation_seconds=seconds,selection_epoch=time.time())
    if 'all_jobs' in old:
        new.update(all_jobs=jobs,clean_reference=jobs[1],jobs=jobs[2:])
    else:new['jobs']=jobs
    write(path,new)
    return old,new


def main():
    initialize();replace_plan(ROOT/'state/graph_before_model_18k_plan.json',24,True)
    path=ROOT/'state/ouro_18k_confirmation_v4_plan.json';old=json.loads(path.read_text())
    if old.get('prospective_observation_seconds')==32:return
    jobs=[old['jobs'][0]];plans=[]
    for p in old['plans']:
        before=json.loads(Path(p).read_text());names=[]
        for parent in [before['clean_reference'],*before['jobs']]:
            assert not (ROOT/'raw'/parent).exists()
            cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent+'_w32'
            cfg.update(name=name,activity_window_seconds=32.,
                observation_selection_reason='Prospective wider analysis interval. Frozen whole-run, local-tail, and ownership gates remain unchanged.')
            assert cfg['activity_window_fraction']*cfg['arrival_offsets'][-1]+32<cfg['arrival_offsets'][-1]
            save(cfg);names.append(name);jobs.append(name)
        selected=Path(p).with_name(Path(p).stem+'_w32.json')
        write(selected,dict(clean_reference=names[0],jobs=names[1:]));plans.append(str(selected))
    write(path.with_name(path.stem+'_before_longer_window.json'),old)
    write(path,dict(jobs=jobs,plans=plans,prospective_observation_seconds=32,selection_epoch=time.time()))
    print('Prepared wider windows before execution. All earlier configurations remain unchanged.')


if __name__=='__main__':main()
