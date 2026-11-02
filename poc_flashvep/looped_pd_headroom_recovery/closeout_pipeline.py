"""Run CPU closeout only after all registered measured work is terminal."""
import argparse
import fcntl
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    target=ROOT/'state/long_window_profile.json'
    while True:
        ready=target.exists() and json.loads(target.read_text()).get('stage')=='LONG_WINDOW_PROFILE_COMPLETE_REVIEW_REQUIRED'
        control=ROOT/'state/source_credit_control.json'
        if control.exists():
            ready=ready and json.loads(control.read_text()).get('stage')=='SOURCE_CREDIT_CONTROL_COMPLETE_REVIEW_REQUIRED'
        fallback=ROOT/'state/long_nsys_fallback.json'
        if fallback.exists():
            ready=ready and json.loads(fallback.read_text()).get('stage')=='LONG_NSYS_FALLBACK_COMPLETE_REVIEW_REQUIRED'
        if ready:break
        if not a.wait:raise RuntimeError('GPU campaign is not terminal. Do not run bulk CPU analysis during clean measurements.')
        time.sleep(5)
    preserve()
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        active=json.loads((ROOT/'state/active.json').read_text())
        assert active.get('status')=='IDLE','A registered experiment remains active'
        owner=resources();before=owner.processes();owner.restore(list(range(8)))
        time.sleep(5)
        write(ROOT/'state/resource_closeout.json',dict(time=time.time(),before=before,after=owner.processes(),
            owned_burners=owner.burners(),GPU_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],text=True),
            scope='Only approved owned burners start on unoccupied physical GPUs. No other user process is signaled. All measured request windows ended before restoration.'))
    modules=['tests','warmup_contract_test','profile_trace_export_test','analyze','capacity_analysis','queues','gates','accounting',
        'physical_memory_accounting','final_audit','paired_analysis','kernel_gap_accounting','device_activity_accounting',
        'reporting','closeout_enrichment','phase_closeout','figures','timeline_figures','secondary_figures','repeat_figure','handoff_lifetime_analysis']
    ledger=[]
    for module in modules:
        write(ROOT/'state/closeout_pipeline.json',dict(stage='CPU_ANALYSIS',module=module,time=time.time(),results=ledger))
        command=[str(PYTHON),'-m','poc_flashvep.looped_pd_headroom_recovery.'+module]
        started=time.time()
        with (ROOT/'logs'/('closeout_'+module+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        row=dict(module=module,returncode=result.returncode,seconds=time.time()-started,command=command)
        ledger.append(row)
        if result.returncode:
            write(ROOT/'state/closeout_pipeline.json',dict(stage='CPU_ANALYSIS_FAILED_REVIEW_REQUIRED',module=module,time=time.time(),results=ledger))
            raise RuntimeError('CPU closeout requires review: '+module)
    preserve()
    write(ROOT/'state/closeout_pipeline.json',dict(stage='REPORTS_READY_MANUAL_EVIDENCE_REVIEW_REQUIRED',time=time.time(),results=ledger,
        scope='No automatic scientific success verdict or push. Review actual causal gates before publication.'))
    print('REPORTS_READY_MANUAL_EVIDENCE_REVIEW_REQUIRED',flush=True)


if __name__=='__main__':main()
