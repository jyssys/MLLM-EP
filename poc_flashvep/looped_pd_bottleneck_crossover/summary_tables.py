"""Separate finite service capacity, host spans, and device spans."""
import csv
import statistics
import collections
import random
import math
from .common import *

def records(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []

def percentile(values,q):
    if not values:return None
    a=sorted(values);i=(len(a)-1)*q;lo=int(i);hi=min(lo+1,len(a)-1)
    return a[lo]*(hi-i)+a[hi]*(i-lo) if hi!=lo else a[lo]

def bootstrap_mean(values):
    if len(values)<2:return [None,None]
    rng=random.Random(8727);n=len(values)
    means=[statistics.mean(rng.choices(values,k=n)) for _ in range(2000)]
    return [percentile(means,.025),percentile(means,.975)]

def wilson(successes,total):
    if not total:return [None,None]
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-radius,center+radius]

def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=fields);out.writeheader();out.writerows(rows)

def capacity():
    groups=collections.defaultdict(list);raw=[];blocked=[]
    for folder in (ROOT/'raw').iterdir():
        if not folder.name.startswith('phase_a_'):continue
        status_path=folder/'status.json'
        run_status=json.loads(status_path.read_text()).get('status') if status_path.exists() else 'NO_STATUS'
        for r in records(folder/'cohorts.jsonl'):
            if r.get('status')=='BLOCKED_KV_CAPACITY':blocked.append(dict(run=folder.name,**r));continue
            if r.get('job',{}).get('profile'):continue
            model=r['model'];job=r['job'];length=len(r.get('input_ids',[[0]])[0])
            batch=r['batch'];chunk=r.get('chunk','native_full_prefill');pool=r.get('KV_blocks','native_vLLM')
            p=r['prefill_plus_first_token_service_s'];d=r['decode_rest_service_s']
            row=dict(run=folder.name,model=model,cell=job.get('cell',''),input_tokens=length,output_tokens=job['cap'],batch=batch,
                run_status=run_status,eligible_primary_capacity=run_status=='COMPLETE',
                chunk=chunk,KV_blocks=pool,repeat=r['repeat'],prefill_service_s=p,decode_rest_service_s=d,
                mu_P_finite=batch/p,mu_D_post_first_finite=batch/d,capacity_ratio=d/p,
                mu_P_native_boundary_finite=batch/r['prefill_host_service_s'] if 'prefill_host_service_s' in r else 'NOT_MEASURED',
                mu_D_native_boundary_finite=batch/r['decode_native_boundary_service_s'] if 'decode_native_boundary_service_s' in r else 'NOT_MEASURED',
                capacity_ratio_native_boundary=r['decode_native_boundary_service_s']/r['prefill_host_service_s'] if 'prefill_host_service_s' in r else 'NOT_MEASURED',
                pre_CODA_prefill_s=r.get('prefill_host_service_s','NOT_MEASURED'),
                first_CODA_s=r.get('first_coda_host_service_s','NOT_MEASURED'),
                evidence=r['evidence'],scope='Finite prepared-cache cohort. Not a sustainable capacity measurement.')
            if job['cap']==1:
                for field in ('mu_D_post_first_finite','capacity_ratio','mu_D_native_boundary_finite','capacity_ratio_native_boundary'):
                    row[field]='NOT_APPLICABLE_PREFILL_ONLY'
            raw.append(row)
            if row['eligible_primary_capacity']:
                groups[(model,job.get('cell',''),length,job['cap'],batch,chunk,pool,r['evidence'])].append(row)
    summary=[]
    for key,rows in groups.items():
        model,cell,length,cap,batch,chunk,pool,evidence=key
        row=dict(model=model,cell=cell,input_tokens=length,output_tokens=cap,batch=batch,chunk=chunk,KV_blocks=pool,
            repeats=len(rows),independent_starts=len(set(r['run'] for r in rows)),evidence=evidence,
            CI_scope='Within-start repeated cohorts unless independent_starts exceeds 1. No between-start CI claim.')
        for metric in ('prefill_service_s','decode_rest_service_s','mu_P_finite','mu_D_post_first_finite','capacity_ratio'):
            v=[r[metric] for r in rows if isinstance(r[metric],(int,float))];bounds=bootstrap_mean(v)
            row.update({metric+'_mean':statistics.mean(v) if v else 'NOT_APPLICABLE',metric+'_p50':statistics.median(v) if v else 'NOT_APPLICABLE',metric+'_CI95_low':bounds[0],metric+'_CI95_high':bounds[1]})
        for metric in ('mu_P_native_boundary_finite','mu_D_native_boundary_finite','capacity_ratio_native_boundary'):
            values=[r[metric] for r in rows if isinstance(r[metric],(int,float))]
            row[metric+'_mean']=statistics.mean(values) if values else 'NOT_MEASURED'
        summary.append(row)
    save_csv(ROOT/'capacity_raw.csv',raw);save_csv(ROOT/'capacity_summary.csv',summary)
    write(ROOT/'capacity_blocked.json',blocked)
    return summary

if __name__=='__main__':print('CAPACITY_GROUPS',len(capacity()))
