"""Compare clean and Nsight P-only cohorts. Do not infer Serving speedup."""
import copy
import sqlite3
import subprocess
from .common import *
from .evidence import status,records
from .extension_discovery import enqueue
from .profiles import union

PLAN=ROOT/'state/P_duty_diagnostic_plan_20261102.json'
LONG_PLAN=ROOT/'state/p_duty_long_plan_20261102.json'


def plan():
    if PLAN.exists():return
    cases=[('ouro','qual_ouro_d128_kv25088_l8192_o32_v1',[1,4,8]),
        ('huginn','ext_huginn_sync_g4_qualification8_v1',[1,2,3])];new=[]
    for model,source,batches in cases:
        source=json.loads((ROOT/'configs'/(source+'.json')).read_text())
        for profiled in (False,True):
            c=copy.deepcopy(source);c.update(name=f'diagnostic_P_duty_{model}_{"profile" if profiled else "clean"}_v1',
                worker='p_duty_worker',capacity_mode='P_ONLY',phase='independent_capacity',P_duty_diagnostic=True,
                nsys=profiled,nsys_graph_trace='node',verify_transfer=False,state_probe=False,gpus=[0],timeout=1800,
                jobs=[dict(batch=b,cap=1,repeats=3) for b in batches],
                scope='P-only device-activity diagnostic. Same qualified model, BF16, full depth, and numerical fixes. No NIXL or D reservation. Not measured online capacity.')
            new.append(c)
    enqueue(new);write(PLAN,dict(time=time.time(),configs=[c['name'] for c in new]))


def _analyze():
    from .analyze import save_csv
    output=[];summary=[]
    cases=[('ouro','ouro'),('huginn','huginn')]
    if LONG_PLAN.exists():cases.append(('ouro_l16384','ouro'))
    for label,model in cases:
        name=f'diagnostic_P_duty_{label}_profile_v1';clean=f'diagnostic_P_duty_{label}_clean_v1'
        if status(name)!=status(clean) or status(name)!='COMPLETE':continue
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        folder=ROOT/'raw'/name;database=folder/'nsys_report.sqlite';report=folder/'nsys_report.nsys-rep'
        if not database.exists():
            result=subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=false','--output='+str(database),str(report)],text=True,capture_output=True)
            write(folder/'nsys_export_result.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            if result.returncode:continue
        conn=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True);conn.row_factory=sqlite3.Row
        strings={r['id']:r['value'] for r in conn.execute('SELECT * FROM StringIds')}
        nvtx={}
        for r in conn.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
            label=r['text'] if r['text'] else strings.get(r['textId'],'')
            if label.startswith('APD_P_DUTY|'):nvtx[label]=r
        actual={r['tag']:r for r in records(folder/'cohorts.jsonl')}
        references={r['tag']:r for r in records(ROOT/'raw'/clean/'cohorts.jsonl')}
        for anchor in records(folder/'p_duty_anchors.jsonl'):
            marker=nvtx[anchor['marker']];a,b=anchor['P_interval'];zero=marker['start'];origin=anchor['anchor_monotonic']
            lo=zero+round((a-origin)*1e9);hi=zero+round((b-origin)*1e9);intervals=[];copies=[];count=0
            for r in conn.execute('SELECT start,end,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<?',(lo,hi)):
                pid=r['globalPid'] if r['globalPid']==anchor['pid'] else (r['globalPid']>>24)&0xffffff
                if pid!=anchor['pid']:continue
                intervals.append((max(lo,r['start']),min(hi,r['end'])));count+=1
            tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
                for r in conn.execute('SELECT start,end,globalPid FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end>? AND start<?',(lo,hi)):
                    pid=r['globalPid'] if r['globalPid']==anchor['pid'] else (r['globalPid']>>24)&0xffffff
                    if pid==anchor['pid']:copies.append((max(lo,r['start']),min(hi,r['end'])))
            tag=anchor['tag'];reference=references[tag];candidate=actual[tag]
            parity=candidate['outputs']==reference['outputs'] and candidate['input_ids']==reference['input_ids']
            overhead=(b-a)/reference['prefill_host_service_s']-1;busy=union(intervals)/(hi-lo)
            value=dict(model=model,length=cfg['length'],name=name,batch=anchor['batch'],tag=tag,clean_P_host_service_ms=1000*reference['prefill_host_service_s'],
                profiled_P_host_service_ms=1000*(b-a),profile_wall_overhead_fraction=overhead,exact_output_parity=parity,
                recorded_P_kernel_busy_fraction=busy,recorded_P_kernel_and_copy_union_fraction=union(intervals+copies)/(hi-lo),
                recorded_kernel_invocations=count,recorded_copy_events=len(copies),
                profile_eligible_for_matched_clean=abs(overhead)<=.05 and parity,
                full_depth_rows=anchor['full_depth_rows'],observed_full_depth_rows=anchor['observed_full_depth_rows'],
                evidence='MEASURED_NSYS_LOCAL_P_SERVICE',
                scope='Local qualified P-only cohort, not native online PD. A non-kernel gap is not automatically CPU overhead or GPU idle. No architectural impossibility or E2E speedup follows.')
            output.append(value)
        conn.close()
    for model,length in sorted({(r['model'],r['length']) for r in output}):
        for batch in sorted({r['batch'] for r in output if r['model']==model and r['length']==length}):
            values=[r for r in output if r['model']==model and r['length']==length and r['batch']==batch]
            summary.append(dict(model=model,length=length,batch=batch,within_start_repeats=len(values),
                clean_host_service_ms_mean=sum(r['clean_P_host_service_ms'] for r in values)/len(values),
                recorded_kernel_fraction_mean=sum(r['recorded_P_kernel_busy_fraction'] for r in values)/len(values),
                matched_profile_pairs_within_5_percent=sum(r['profile_eligible_for_matched_clean'] for r in values),
                scope='Three within-start repeats. No independent-start confidence interval. Not a proven ceiling for every supported native P configuration.'))
    save_csv('p_duty_windows.csv',output);save_csv('p_duty_summary.csv',summary);write(ROOT/'p_duty_summary.json',summary)
    marker=ROOT/'state'/('p_duty_long_analysis_state.json' if LONG_PLAN.exists() else 'p_duty_analysis_state.json')
    previous=json.loads(marker.read_text()) if marker.exists() else {}
    write(marker,dict(previous,status='CPU_ANALYSIS_COMPLETE',completed_time=time.time(),
        analyzed_windows=len(output),summary_rows=len(summary),
        scope='CPU parsing completed. Per-window profiling eligibility remains separate. This status does not certify Serving saturation.'))
    return summary


def analyze():
    import fcntl
    with (ROOT/'state/gpu_lease.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        return _analyze()


def advance():
    if not PLAN.exists():return
    summary=ROOT/'p_duty_summary.json'
    if summary.exists() and (not LONG_PLAN.exists() or any(r.get('length')==16384 for r in json.loads(summary.read_text()))):return
    path=ROOT/'state'/('p_duty_long_analysis_state.json' if LONG_PLAN.exists() else 'p_duty_analysis_state.json')
    if path.exists():return
    from .campaign import terminal
    names=json.loads(PLAN.read_text())['configs']
    if LONG_PLAN.exists():names+=json.loads(LONG_PLAN.read_text())['configs']
    states={name:terminal(name) for name in names}
    if any(s not in ('COMPLETE','FAILED','SKIPPED') for s in states.values()):return
    if any(s!='COMPLETE' for s in states.values()):
        write(path,dict(time=time.time(),status='BLOCKED_DIAGNOSTIC_RUN',states=states));return
    with (ROOT/'logs/p_duty_analysis_20261102.log').open('a') as log:
        child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.p_duty','analyze'],cwd=REPO,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(path,dict(time=time.time(),status='CPU_ANALYSIS_WAITING_FOR_QUIET_LEASE',pid=child.pid))


def plan_long():
    if LONG_PLAN.exists():return
    source=json.loads((ROOT/'configs/ext_ouro_l16384_o32_native_qualification8_v1.json').read_text());new=[]
    for profiled in (False,True):
        c=copy.deepcopy(source);c.update(name=f'diagnostic_P_duty_ouro_l16384_{"profile" if profiled else "clean"}_v1',
            worker='p_duty_worker',capacity_mode='P_ONLY',phase='independent_capacity',P_duty_diagnostic=True,
            nsys=profiled,nsys_graph_trace='node',verify_transfer=False,state_probe=False,gpus=[0],timeout=1800,
            jobs=[dict(batch=b,cap=1,repeats=3) for b in (1,2,4)],
            scope='Matched qualified 16K P-only diagnostic. Remove D reservation and NIXL dependencies. Keep model math and numerical fixes. Not online capacity.')
        new.append(c)
    enqueue(new);write(LONG_PLAN,dict(time=time.time(),configs=[c['name'] for c in new]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','plan_long','analyze'));args=parser.parse_args()
    if args.action=='plan':plan()
    elif args.action=='plan_long':plan_long()
    else:analyze()
