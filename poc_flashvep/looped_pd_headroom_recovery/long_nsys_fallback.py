"""Use one existing Nsight path if the long native Ouro profiles remain perturbed."""
import copy
from .common import *
from .initial_campaign import run
from .profile_analysis import analyze
from .gates import one


def main():
    write(ROOT/'state/long_nsys_fallback.json',dict(stage='WAITING_FOR_LONG_NATIVE_PROFILES',time=time.time()))
    while True:
        p=ROOT/'state/long_window_profile.json'
        if p.exists() and json.loads(p.read_text()).get('stage')=='LONG_WINDOW_PROFILE_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    qualified=[]
    for path in (ROOT/'raw').glob('long_profile_ouro_*/profile_analysis.json'):
        v=json.loads(path.read_text())
        if v.get('eligible_for_matched_clean'):qualified.append(path.parent.name)
    result=dict(qualified_native=qualified)
    if not qualified:
        plan=json.loads((ROOT/'state/confirmation_plan.json').read_text())
        clean=next(r['PD'] for r in plan['runs'] if r['model']=='ouro' and r['start']==1)
        cfg=copy.deepcopy(json.loads((ROOT/'configs'/(clean+'.json')).read_text()))
        cfg.update(name='long_profile_ouro_PD_start1_nsys_fallback_n128',phase='bounded_stage_profiling',
            cuda_only_profile=False,nsys=True,P_only_profiler=True,capture_role='prefill',
            capture_seconds=3.,capture_prewarmed=True,capture_until_close=True,profile_stop_at_exit=True,
            nsys_cuda_flush_ms=300000,profile_finalization_timeout=600,matched_clean=clean,timeout=1200)
        cfg.pop('capture_roles',None)
        cfg['native_transport_limits']=dict(cfg.get('native_transport_limits',{}),shutdown_timeout=600)
        write(ROOT/'configs'/(cfg['name']+'.json'),cfg)
        write(ROOT/'state/long_nsys_fallback.json',dict(stage='ALTERNATIVE_COLLECTOR_ACTIVE',time=time.time(),name=cfg['name']))
        okay=run(cfg['name']);result=dict(name=cfg['name'],complete=okay)
        if okay:
            proof=analyze(cfg['name'],clean);result['profile']=proof
            if proof.get('status')=='PARSED':
                folder=ROOT/'raw'/cfg['name']
                write(folder/'profile_analysis_prefill.json',dict(proof,role='prefill'))
                write(folder/'aligned_kernels_prefill.json',json.loads((folder/'aligned_kernels.json').read_text()))
            result['same_start_gates']=one(cfg)
    write(ROOT/'state/long_nsys_fallback.json',dict(stage='LONG_NSYS_FALLBACK_COMPLETE_REVIEW_REQUIRED',time=time.time(),result=result,
        scope='At most one extra collector test. Same frozen inputs, arrivals, runtime, and 5% perturbation gate. No threshold relaxation or synthetic GPU activity.'))


if __name__=='__main__':main()
