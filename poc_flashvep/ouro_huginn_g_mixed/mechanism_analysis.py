"""Separate actual plans, heavy interventions, and event service intervals."""
from collections import defaultdict
import json
import numpy as np
from .common import ROOT,write
from .analyze import csvfile

def signature(plan):
    return (plan['operation'],plan['group'],tuple((s['id'],s['depth'],s['start'],s['count'],s['prompt']) for s in plan['spans']))

def main():
    counts=[];interventions=[];events=[];observers=[]
    for path in (ROOT/'raw').glob('*/results.json'):
        rows=json.loads(path.read_text());refs={r['repeat']:r for r in rows if r['arm']=='mixed'}
        for row in rows:
            if not row['plans']:continue
            plans=row['plans']
            target=[p for p in plans if any(s['id']=='long' for s in p['spans'])]
            counts.append(dict(job=path.parent.name,model=row['model'],P=row['P'],N=row['N'],C=row['C'],
                arm=row['arm'],G=row['G'],repeat=row['repeat'],
                target_recurrent_submissions=sum(p['operation']=='recurrent' for p in target),
                target_mixed_submissions=sum(bool(p.get('mixed')) for p in target),
                target_pure_grouped_submissions=sum(p['group']>1 and p['operation']=='recurrent' for p in target),
                evidence='LIVE_MEASURED_PLAN_OBSERVER'))
    for path in (ROOT/'raw').glob('*/interventions.json'):
        rows=json.loads(path.read_text());reference=rows[0]
        for row in rows:
            info=row['intervention'];ref_info=reference['intervention']
            record=dict(job=path.parent.name,model=row['model'],N=row['N'],action=row['arm'],G=row['G'],
                fired=bool(info),same_source_state=bool(info and ref_info and info['state_sha256']==ref_info['state_sha256']),
                TTFT_observed_ms=row['TTFT_long_ms'],ITL_p99_observed_ms=row['ITL_p99_ms'],
                evidence='LIVE_MEASURED_HEAVY_PAUSED_STATE_INTERVENTION',
                qualifier='Includes diagnostic pause. Not clean serving speedup.')
            record['fingerprint_scope']='FULL_VALID_KV' if path.parent.name.startswith('full_state') else 'VALID_KV_TAIL_SAMPLES'
            if info:record.update(state_sha=info['state_sha256'],state_hash_pause_ms=info['pause_for_state_hash_ms'])
            interventions.append(record)
    for path in (ROOT/'raw').glob('*/profile.json'):
        for row in json.loads(path.read_text()):
            for i,interval in enumerate(row['event_intervals']):
                events.append(dict(job=path.parent.name,model=row['model'],arm=row['arm'],G=row['G'],
                    interval=i,**interval,evidence='LIVE_MEASURED_CUDA_STREAM_ELAPSED_NOT_KERNEL_BUSY'))
    for path in (ROOT/'raw').glob('*/observer_comparison.json'):
        rows=json.loads(path.read_text())
        off=[r for r in rows if not r['plans']];on=[r for r in rows if r['plans']]
        observers.append(dict(job=path.parent.name,model=rows[0]['model'],
            wall_on_over_off=np.mean([r['wall_s'] for r in on])/np.mean([r['wall_s'] for r in off]),
            token_exact=all([r['tokens'] for r in a['requests']]==[r['tokens'] for r in b['requests']]
                for a,b in zip(off,on)),evidence='LIVE_MEASURED_ON_OFF_TECHNICAL_PAIRS'))
    csvfile('actual_mode_counts.csv',counts)
    csvfile('state_interventions.csv',interventions)
    csvfile('paired_interventions.csv',interventions)
    csvfile('gpu_event_intervals.csv',events)
    csvfile('observer_parity.csv',observers)
    write(ROOT/'mechanism_counts.json',dict(plan_cohorts=len(counts),interventions=len(interventions),
        matched_source_interventions=sum(r['same_source_state'] for r in interventions),
        stream_intervals=len(events),observer_blocks=len(observers)))

if __name__=='__main__':main()
