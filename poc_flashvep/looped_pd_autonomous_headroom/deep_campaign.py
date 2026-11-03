"""New bounded research round. Use short native Nsight captures first."""
import copy
from .common import *
from .prepare import base,put
from .run import execute


def main():
    activate()
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    from ..looped_pd_headroom_recovery.gates import one
    results=[]
    for model in ('ouro','huginn'):
        source=json.loads((ROOT/'configs'/f'pilot_{model}_baseline_n24.json').read_text())
        for role in ('prefill','decode'):
            clean=copy.deepcopy(source)
            clean.update(name=f'deep_{model}_{role}_clean24',phase='qps_pilot',projection_graph_rows=0)
            clean['native_transport_limits']=dict(clean.get('native_transport_limits',{}),shutdown_timeout=180)
            name=put(clean,24,source['arrival_rate'])
            write(ROOT/'state/deep_campaign.json',dict(stage='MATCHED_CLEAN',name=name,time=time.time()))
            if not execute(name):
                results.append(dict(name=name,status='FAILED_CLEAN'));continue
            prof=copy.deepcopy(clean)
            prof.update(name=f'deep_{model}_{role}_nsys24',phase='bounded_stage_profiling',
                P_only_profiler=True,capture_role=role,capture_seconds=3.,capture_after_submissions=2,
                capture_prewarmed=False,capture_until_close=False,nsys=True,cuda_only_profile=False,
                matched_clean=name,nsys_cuda_flush_ms=300000,timeout=600,profile_finalization_timeout=180)
            pname=put(prof,24,source['arrival_rate'])
            write(ROOT/'state/deep_campaign.json',dict(stage='SHORT_NATIVE_NSYS',name=pname,time=time.time()))
            okay=execute(pname);r=dict(name=pname,complete=okay)
            if okay:
                r['profile']=analyze(pname,name)
                p=ROOT/'raw'/pname/'profile_analysis.json'
                if p.exists():
                    value=json.loads(p.read_text());value['bounded_capture']=True;write(p,value)
                r['gates']=one(prof)
            results.append(r);write(ROOT/'state/deep_profile_results.json',results)
    write(ROOT/'state/deep_campaign.json',dict(stage='PROFILE_CONTROLS_READY',time=time.time(),results=results))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/deep_campaign.json',dict(stage='FAILED_REVIEW_REQUIRED',time=time.time(),error=repr(error)));raise
