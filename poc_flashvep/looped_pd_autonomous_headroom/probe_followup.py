"""Complete profile recovery and bounded real-D probes after initial capacity."""
import copy
from .common import *
from .prepare import put
from .run import execute


def main():
    while time.time()<GPU_DEADLINE-900:
        p=ROOT/'state/extension_progress.json'
        if p.exists() and json.loads(p.read_text())['stage'] in ('CAPACITY_PROFILE_ANALYZED','FAILED_REVIEW_REQUIRED'):break
        time.sleep(5)
    activate()
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_headroom_recovery.cuda_only_profile import analyze
    from ..looped_pd_headroom_recovery.gates import one
    ledger=[]
    for model in ('ouro','huginn'):
        clean=f'pilot_{model}_graph1024_n24'
        original=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
        c=copy.deepcopy(original)
        c.update(name=f'profile_retry_{model}_graph1024_P_only24',phase='bounded_stage_profiling',
            cuda_only_profile=True,capture_roles=['prefill'],capture_role='prefill',nsys=True,
            matched_clean=clean,timeout=600,profile_finalization_timeout=180)
        c['native_transport_limits']=dict(c.get('native_transport_limits',{}),shutdown_timeout=180)
        name=put(c,24,c['arrival_rate'])
        write(ROOT/'state/probe_followup.json',dict(stage='PROFILE_RECOVERY',name=name,time=time.time()))
        okay=execute(name);record=dict(name=name,complete=okay)
        if okay:record.update(profile=analyze(name,clean,'prefill'),gates=one(c))
        ledger.append(record);write(ROOT/'state/probe_followup_results.json',ledger)
        # This probe can establish D headroom alone. Joint success still needs
        # simultaneous P device evidence and all frozen P/D gates.
        c=copy.deepcopy(original);c.update(name=f'D_probe_{model}_tokens16_n24',D_probe_tokens=16,
            D_probe_count=4,D_probe_interval_s=1.,phase='qps_pilot')
        name=put(c,24,c['arrival_rate'])
        write(ROOT/'state/probe_followup.json',dict(stage='REAL_D_PREFILL_PROBE',name=name,time=time.time()))
        okay=execute(name);record=dict(name=name,complete=okay)
        if okay:
            record['background_output_parity']=evidence.compare(clean,name)
            p=ROOT/'raw'/name/'D_probe_completion.json'
            record['probe']=json.loads(p.read_text()) if p.exists() else {'status':'NOT_MEASURED'}
        ledger.append(record);write(ROOT/'state/probe_followup_results.json',ledger)
    write(ROOT/'state/probe_followup.json',dict(stage='PROBE_RESULTS_AVAILABLE',time=time.time(),results=ledger))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/probe_followup.json',dict(stage='FAILED_REVIEW_REQUIRED',time=time.time(),error=repr(error)));raise
