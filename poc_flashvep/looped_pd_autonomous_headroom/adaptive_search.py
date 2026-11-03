"""Continue from collector controls through measured workload selection."""
import copy
from .common import *
from .prepare import base,put
from .run import execute


def main():
    write(ROOT/'state/adaptive_search.json',dict(stage='WAITING_FOR_COLLECTOR_CONTROLS',time=time.time()))
    while time.time()<LONG_START_CUTOFF:
        p=ROOT/'state/deep_campaign.json'
        if p.exists() and json.loads(p.read_text())['stage'] in ('PROFILE_CONTROLS_READY','FAILED_REVIEW_REQUIRED'):break
        time.sleep(5)
    activate()
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    from ..looped_pd_headroom_recovery.gates import one
    from ..looped_pd_atomic_prefill import evidence
    ledger=[];selected=[]
    for model,cells in [('ouro',[(6144,32,2.5),(8192,32,3.0),(8192,32,4.0)]),
                        ('huginn',[(2560,16,1.2),(3072,16,1.2),(2560,32,1.5)])]:
        candidates=[]
        for length,cap,rate in cells:
            tag=f'{model}_L{length}_O{cap}_q{str(rate).replace(".","p")}'
            c=base(model,'search_'+tag,length=length,cap=cap,projection_graph_rows=0,phase='qps_pilot')
            c['native_transport_limits']=dict(c.get('native_transport_limits',{}),shutdown_timeout=180)
            # Each new geometry gets an independent, valid-state service reference.
            local=copy.deepcopy(c);local.update(name='capacity_search_'+tag,worker='capacity_worker',capacity_mode=True,
                phase='independent_capacity',jobs=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4)],timeout=600)
            lname=put(local,4)
            write(ROOT/'state/adaptive_search.json',dict(stage='CAPACITY',name=lname,time=time.time()))
            if not execute(lname):
                ledger.append(dict(name=lname,status='FAILED_CAPACITY'));continue
            name=put(c,24,rate)
            write(ROOT/'state/adaptive_search.json',dict(stage='CLEAN_SEARCH',name=name,time=time.time()))
            if not execute(name):
                ledger.append(dict(name=name,status='FAILED_CLEAN'));continue
            work=evidence.full_work(name);write(ROOT/'raw'/name/'full_work_check.json',work)
            if not work['passed']:
                ledger.append(dict(name=name,status='BLOCKED_CORRECTNESS'));break
            prof=copy.deepcopy(c);prof.update(name=name+'_nsys',phase='bounded_stage_profiling',
                P_only_profiler=True,nsys=True,capture_role='prefill',capture_seconds=3.,
                capture_prewarmed=model=='ouro',capture_until_close=model=='ouro',
                profile_stop_at_exit=model=='ouro',nsys_cuda_flush_ms=300000,matched_clean=name,
                profile_finalization_timeout=180,timeout=600)
            pname=put(prof,24,rate)
            write(ROOT/'state/adaptive_search.json',dict(stage='PROFILE_SEARCH',name=pname,time=time.time()))
            okay=execute(pname);record=dict(name=name,profile_name=pname,complete=okay)
            if okay:
                record['profile']=analyze(pname,name)
                record['gates']=one(prof)
                candidates.append((c,record))
            ledger.append(record);write(ROOT/'state/adaptive_search_results.json',ledger)
            if record.get('gates',{}).get('P_compute_gate') and record['gates'].get('D_SLO_gate'):
                break
        if candidates:
            def score(item):
                g=item[1]['gates'];p=item[1]['profile']
                return (bool(g.get('P_compute_gate') and g.get('D_SLO_gate') and g.get('D_credit_gate')),
                    bool(p.get('eligible_for_matched_clean')),bool(g.get('D_SLO_gate')),
                    g.get('P_actual_kernel_busy_central') or 0,-(g.get('D_credit_refusal_sample_fraction') or 0))
            chosen=max(candidates,key=score)
            selected.append(dict(configuration=chosen[0],evidence=chosen[1],
                selection_scope='Best diagnostic candidate among all retained cells. This is not a successful regime verdict.'))
            write(ROOT/'state/adaptive_selection.json',selected)
    write(ROOT/'state/adaptive_search.json',dict(stage='SEARCH_COMPLETE',time=time.time(),selected=selected))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/adaptive_search.json',dict(stage='FAILED_REVIEW_REQUIRED',time=time.time(),error=repr(error)));raise
