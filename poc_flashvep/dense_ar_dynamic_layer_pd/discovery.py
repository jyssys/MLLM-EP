"""Run the output sweep first and refine loads from observed trade-offs."""
import fcntl,statistics
from .common import *
from .prepare import base
from .run import execute
from .analyze import metrics,compare,records


def main():
    initialize();progress=ROOT/'state/phase_A_initial_progress.json'
    initial=json.loads(progress.read_text())
    assert len(initial['results'])==8 and not initial.get('failed_job'),'Initial qualification is incomplete'
    assert initial['results'][1]['output_gate']['passed'],'Native repetition failed'
    candidates=[r for r in initial['results'] if r['name'].startswith('qual_coloc_') and r['output_gate']['passed']]
    assert candidates,'No exact Colocated baseline is available'
    # Simultaneous N8 requests do not expose streaming Prefill interference.
    # Keep every exact candidate for the first real online output sweep.
    chosen={r['name'] for r in candidates}
    templates={name:json.loads((ROOT/'configs'/(name+'.json')).read_text()) for name in chosen}
    p=records(ROOT/'raw/capacity_P_B124_v1/cohorts.jsonl')
    mu=max(statistics.median(r['batch']/r['P_plus_first_sample_s'] for r in p if r['batch']==b) for b in (1,2,4))
    rate=mu*1.35
    write(ROOT/'analysis/initial_capacity_selection.json',dict(P_capacity_requests_s=mu,first_QPS=rate,
        candidate_Colocated_configs=sorted(templates),
        rule='Use independently measured P capacity times1.35. Keep all exact Colocated configurations for the streaming sweep.',
        evidence='MEASURED_LOCAL_CAPACITY',scope='Local capacity selects an online pilot. It does not establish online P saturation or physical D slack.'))
    results=[];pairs=[]
    try:
        control=base('qual_native_continuous_O200_N8_v1',count=8,cap=200,dataset='qualification.json')
        control.update(arrival_offsets=[0.]*8,arrival_rate=None,serialize_D_imports=False,cohort_barrier=False,
            phase='CONTINUOUS_NATIVE_KV_IMPORT_QUALIFICATION')
        if not execute(save(control),restore=False):return
        continuous_gate=compare('qual_native_O200_N8_A_v1',control['name'])
        write(ROOT/'analysis/continuous_Native_output_gate.json',continuous_gate)
        for cap in (16,32,64,128,200):
            native_name=None
            labels=['native16']+(['native_continuous'] if continuous_gate['passed'] else [])+[name.removeprefix('qual_coloc_').split('_O200')[0] for name in sorted(templates)]
            for label in labels:
                c=base(f'coarse_O{cap}_N64_q{rate:.3f}_{label}_s1_v1',cap=cap,rate=rate)
                if label=='native16':c.update(D_admission_batch=16)
                elif label=='native_continuous':c.update(serialize_D_imports=False,cohort_barrier=False)
                else:
                    old=next(v for k,v in templates.items() if k.removeprefix('qual_coloc_').split('_O200')[0]==label)
                    c.update({k:old[k] for k in ('arm','reuse_prefill_token','serialize_D_imports','cohort_barrier','colocated_cohorts',
                        'chunk','token_budget','D_slots','atomic_prompt','warmup_batch_sizes')})
                c.update(phase='OUTPUT_LENGTH_CROSSOVER_COARSE_DISCOVERY')
                name=save(c)
                if not execute(name,restore=False):return
                row=dict(name=name,metrics=metrics(name),label=label,cap=cap)
                if label=='native16':native_name=name
                else:
                    row['output_gate']=compare(native_name,name)
                    if not row['output_gate']['passed']:
                        row['qualification']='FAILED_MATCHED_OUTPUT_GATE'
                        results.append(row);write(ROOT/'state/discovery_progress.json',dict(results=results,pairs=pairs,
                            invalid_streaming_config=name,scope='Reject this configuration. Continue the other already-qualified baseline configurations.'))
                        print('COARSE_OUTPUT_REJECTED',name,row['output_gate']['failures'],flush=True)
                        continue
                    n=next(r['metrics'] for r in results if r['name']==native_name);m=row['metrics']
                    pairs.append(dict(cap=cap,QPS=rate,Native=native_name,Candidate=name,
                        candidate_arm=c['arm'],comparison_type='COLOCATED' if c['arm']=='DP2' else 'NATIVE_IMPORT_VARIANT',
                        TTFT_ratio=m['TTFT_p95_ms']/n['TTFT_p95_ms'],ITL_ratio=m['ITL_p99_ms']/n['ITL_p99_ms'],
                        TPOT_ratio=m['TPOT_p95_ms']/n['TPOT_p95_ms'],completed_QPS_ratio=m['completed_QPS']/n['completed_QPS'],
                        scope='Single-start pilot. No physical saturation or repeatability claim.'))
                results.append(row);write(ROOT/'state/discovery_progress.json',dict(results=results,pairs=pairs))
                print('COARSE',cap,label,row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
        write(ROOT/'analysis/coarse_crossover_pairs.json',dict(pairs=pairs,phase_A_verdict='PILOT_ONLY',
            physical_same_window_qualification='NOT_MEASURED',independent_confirmation='NOT_RUN'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
