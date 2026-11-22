"""Save restart state without changing GPU ownership or experiment results."""
import argparse
import collections
import json
import os
from pathlib import Path
import time
from .common import ROOT,START,BRANCH,write,time_available

def snapshot():
    lanes={}
    for p in (ROOT/'state').glob('*_progress*.json'):
        record=json.loads(p.read_text());results=record.get('results',[])
        lanes[p.name]=dict(record_time=record.get('time'),finished=len(results),
            last=results[-1] if results else None,
            statuses=dict(collections.Counter(r.get('status','PASS' if r.get('passed') else 'FAILED') for r in results)))
    leases=[]
    for p in (ROOT/'state').glob('pair_lease_*.json'):
        r=json.loads(p.read_text())
        if r['status']=='RUNNING':
            proc=Path('/proc',str(r['pid']));r['process_exists']=proc.exists();r['elapsed_seconds']=time.time()-r['time']
            if proc.exists():r['process_owned']=proc.stat().st_uid==os.getuid()
        leases.append(r)
    statuses=collections.Counter();cancellations=0
    for p in (ROOT/'raw').glob('*/dispatcher_result.json'):
        result=json.loads(p.read_text());status='COMPLETE' if result['complete'] else result.get('status','FAILED')
        statuses[status]+=1;cancellations+=result.get('launched') is False
    checkpoints=[p.name for p in (ROOT/'state').glob('*complete*.json')]
    row=dict(time=time.time(),elapsed_research_hours=(time.time()-START)/3600,branch=BRANCH,
        GPU_leases=leases,lanes=lanes,raw_status_counts=dict(statuses),never_launched_cancellations=cancellations,
        checkpoints=sorted(checkpoints),deadline_active=time_available(),
        scope='Read-only process and receipt observation. A missing completion file is not a failed scientific result. '
            'No GPU process,request,dataset,index or frozen raw evidence is modified.')
    write(ROOT/'state/resume_snapshot_v2.json',row);return row

def compact(row):
    return dict(elapsed_hours=round(row['elapsed_research_hours'],2),raw=row['raw_status_counts'],
        c0_finished=sum(v['finished'] for k,v in row['lanes'].items() if k.startswith('c0_output_matrix_v2')),
        c3_qualified=sum(v['statuses'].get('PASS',0) for k,v in row['lanes'].items() if k.startswith('c3_')),
        active=[r['name'] for r in row['GPU_leases'] if r['status']=='RUNNING'],checkpoints=len(row['checkpoints']))

def watch():
    last=None
    while time_available():
        row=snapshot();brief=compact(row);key=json.dumps({k:v for k,v in brief.items() if k!='elapsed_hours'},sort_keys=True)
        if key!=last:
            with (ROOT/'logs/progress_watch_v2.jsonl').open('a') as stream:stream.write(json.dumps(brief)+'\n')
            print(json.dumps(brief),flush=True);last=key
        time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('snapshot','watch'));args=p.parse_args()
    watch() if args.action=='watch' else print(json.dumps(compact(snapshot())),flush=True)
