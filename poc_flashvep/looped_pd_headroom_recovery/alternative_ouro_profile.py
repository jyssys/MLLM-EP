"""Test the existing selected-stage Nsight path after perturbed native traces."""
import copy
from .common import *
from .initial_campaign import run
from .profile_analysis import analyze
from .gates import one


def main():
    plan=json.loads((ROOT/'state/confirmation_plan.json').read_text())
    item=next(r for r in plan['selection'] if r['model']=='ouro')
    clean=item['PD_pilot'];cfg=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
    cfg.update(name='profile_warm_ouro_nsys_alternative24',phase='bounded_stage_profiling',
        cuda_only_profile=False,nsys=True,P_only_profiler=True,capture_role='prefill',
        capture_seconds=3.,capture_prewarmed=True,capture_until_close=True,profile_stop_at_exit=True,
        nsys_cuda_flush_ms=300000,profile_finalization_timeout=600,matched_clean=clean,timeout=900)
    cfg.pop('capture_roles',None)
    cfg['native_transport_limits']=dict(cfg.get('native_transport_limits',{}),shutdown_timeout=600)
    write(ROOT/'configs'/(cfg['name']+'.json'),cfg)
    okay=run(cfg['name']);record=dict(name=cfg['name'],complete=okay)
    if okay:
        record['profile']=analyze(cfg['name'],clean)
        record['same_start_gates']=one(cfg)
    write(ROOT/'state/alternative_ouro_profile.json',dict(stage='COMPLETE_REVIEW_REQUIRED',time=time.time(),result=record,
        reason='Both native selected traces retained exact outputs but exceeded the fixed 5% wall-perturbation limit. Test the already implemented selected-stage Nsight collector once. Preserve all failed qualification results. No threshold change.'))
    restore_if_idle()


if __name__=='__main__':main()
