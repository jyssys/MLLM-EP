"""Continue completed controls through capacity and native profile analysis."""
import copy
from .common import *
from .prepare import base,put
from .run import execute


def main():
    initialize();activate()
    from ..looped_pd_headroom_recovery.cuda_only_profile import analyze
    from ..looped_pd_headroom_recovery.gates import one
    from ..looped_pd_headroom_recovery import analyze as summary
    ledger=[]
    for model in ('ouro','huginn'):
        for label,group in (('baseline',0),('graph1024',1024)):
            c=base(model,f'capacity_extension_{model}_{label}',worker='capacity_worker',
                capacity_mode=True,phase='independent_capacity',projection_graph_rows=group,
                jobs=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4)],timeout=600)
            name=put(c,4)
            write(ROOT/'state/extension_progress.json',dict(stage='CAPACITY',name=name,time=time.time()))
            okay=execute(name)
            ledger.append(dict(name=name,complete=okay));write(ROOT/'state/extension_results.json',ledger)
        for label in ('baseline','graph1024'):
            clean=f'pilot_{model}_{label}_n24'
            c=copy.deepcopy(json.loads((ROOT/'configs'/(clean+'.json')).read_text()))
            c.update(name=f'profile_extension_{model}_{label}_both24',phase='bounded_stage_profiling',
                cuda_only_profile=True,capture_roles=['prefill','decode'],capture_role='prefill',
                P_only_profiler=False,nsys=True,matched_clean=clean,timeout=600,profile_finalization_timeout=180)
            name=put(c,24,c['arrival_rate'])
            write(ROOT/'state/extension_progress.json',dict(stage='PROFILE',name=name,time=time.time()))
            okay=execute(name);record=dict(name=name,complete=okay)
            if okay:
                record['profiles']={role:analyze(name,clean,role) for role in ('prefill','decode')}
                record['gates']=one(c)
            ledger.append(record);write(ROOT/'state/extension_results.json',ledger)
    summary.main()
    write(ROOT/'state/extension_progress.json',dict(stage='CAPACITY_PROFILE_ANALYZED',time=time.time(),
        next_action='Inspect measured gates. Implement a real D probe only on a safe validated path. Do not declare joint success without the probe.'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/extension_progress.json',dict(stage='FAILED_REVIEW_REQUIRED',time=time.time(),error=repr(error)));raise
