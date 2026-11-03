"""Test selected conditions with 128 requests and actual additional D work."""
import copy
from .common import *
from .prepare import put
from .run import execute


def main():
    while time.time()<LONG_START_CUTOFF:
        p=ROOT/'state/adaptive_search.json'
        if p.exists() and json.loads(p.read_text())['stage'] in ('SEARCH_COMPLETE','FAILED_REVIEW_REQUIRED'):break
        time.sleep(5)
    plan=ROOT/'state/adaptive_selection.json'
    from .aligned_gates import refresh_selection, one
    refresh_selection()
    selected=json.loads(plan.read_text()) if plan.exists() else []
    activate()
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    from ..looped_pd_atomic_prefill import evidence
    ledger=[]
    for selection in selected:
        original=selection['configuration'];model=original['model']
        confirmed=False
        for start in (1,2,3):
            if start>1 and not confirmed:break
            c=copy.deepcopy(original)
            c.update(name=f'long_deep_{model}_start{start}_clean128',count=128,phase='final_repeat',
                arrival_seed=2026110311+start,timeout=900)
            name=put(c,128,c['arrival_rate'],split='confirmation')
            write(ROOT/'state/long_validation.json',dict(stage='CLEAN_128',name=name,time=time.time()))
            if not execute(name):break
            if not evidence.full_work(name)['passed']:
                ledger.append(dict(name=name,status='BLOCKED_WORK_ACCOUNTING'));break
            probe=copy.deepcopy(c);probe.update(name=f'long_deep_{model}_start{start}_probe16_128',
                D_probe_tokens=16,D_probe_count=8,D_probe_interval_s=2.)
            pname=put(probe,128,c['arrival_rate'],split='confirmation')
            write(ROOT/'state/long_validation.json',dict(stage='REAL_D_PROBE_128',name=pname,time=time.time()))
            if not execute(pname):break
            parity=evidence.compare(name,pname)
            write(ROOT/'raw'/pname/'background_output_parity.json',parity)
            if not parity['passed']:
                ledger.append(dict(name=pname,status='BLOCKED_OUTPUT_PARITY'));break
            prof=copy.deepcopy(probe);prof.update(name=pname+'_nsys',phase='bounded_stage_profiling',
                P_only_profiler=True,nsys=True,capture_role='prefill',capture_seconds=3.,
                capture_prewarmed=True,capture_until_close=True,profile_stop_at_exit=True,
                nsys_cuda_flush_ms=300000,matched_clean=pname,profile_finalization_timeout=300)
            prof['native_transport_limits']=dict(prof.get('native_transport_limits',{}),shutdown_timeout=300)
            if model == 'huginn':
                prof.update(name=pname+'_nsys_timed', capture_seconds=0.,
                    autonomous_capture_seconds=3., capture_arrival_fraction=.35,
                    capture_prewarmed=False, capture_until_close=False,
                    profile_stop_at_exit=False)
            qname=put(prof,128,c['arrival_rate'],split='confirmation')
            write(ROOT/'state/long_validation.json',dict(stage='SAME_START_P_AND_D_PROBE',name=qname,time=time.time()))
            okay=execute(qname);record=dict(model=model,start=start,name=qname,complete=okay)
            confirmed=False
            if okay:
                record['profile']=analyze(qname,pname);g=one(prof);record['gates']=g
                path=ROOT/'raw'/qname/'D_probe_requests.jsonl'
                probes=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
                central=[r for r in probes if g['observation_start']<=r['time']<=r['finish']<=g['observation_end']]
                record['completed_probes_in_same_interval']=len(central)
                # Exact native reference validation for these prompt operands is
                # still required before any final positive research verdict.
                confirmed=bool(g.get('P_compute_gate') and g.get('D_SLO_gate') and g.get('D_credit_gate') and central)
                record['provisional_joint_screen']=confirmed
            ledger.append(record);write(ROOT/'state/long_validation_results.json',ledger)
    write(ROOT/'state/long_validation.json',dict(stage='LONG_VALIDATION_COMPLETE',time=time.time(),results=ledger))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/long_validation.json',dict(stage='FAILED_REVIEW_REQUIRED',time=time.time(),error=repr(error)));raise
