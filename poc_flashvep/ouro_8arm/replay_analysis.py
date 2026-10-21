"""Matched replay ratios. Request and repeat uncertainty remain separate."""
import collections
import numpy as np
from .contract import *
from .serving_analysis import table


def ratio_CI(a,b,request_samples=True):
    assert len(a)==len(b) and len(a)>0
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    point=float(a.mean()/b.mean())
    if len(a)<2:return point,None,None
    rng=np.random.default_rng(20261021);idx=rng.integers(0,len(a),(5000,len(a)))
    ratios=a[idx].mean(axis=1)/b[idx].mean(axis=1)
    return point,float(np.quantile(ratios,.025)),float(np.quantile(ratios,.975))


def main():
    sources={};summary=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.name.startswith(('replay_','depth_replay_')) or not (folder/'complete.json').exists():continue
        values=[];times={};resident={};preparation={}
        for file in folder.glob('*.json'):
            r=read(file)
            if not isinstance(r,dict):continue
            if r.get('lane')=='FROZEN_TOKEN_HARDWARE_REPLAY' and r['cohort_index']>0:values.append(r)
            if file.name.startswith('cohort_') and r.get('rank')==0 and r['repeat']>0:
                times[r['repeat']]=r['makespan_s'] if 'makespan_s' in r else r['end']-r['origin']
                if 'resident_model_cohort_wall_s' in r:resident[r['repeat']]=r['resident_model_cohort_wall_s']
                if 'admission_to_service_s' in r:preparation[r['repeat']]=r['admission_to_service_s']
        if not values or not times:continue
        arm=values[0]['arm'];case=next(read(p)['replay_case'] for p in folder.glob('cohort_*_rank0.json'))
        family='TRACE_LOCKED_DYNAMIC_TOKENS' if folder.name.startswith('depth_replay_') else 'STATIC_FROZEN_TOKENS'
        byrequest=collections.defaultdict(list)
        for r in values:byrequest[r['id']].append(r)
        requests={rid:dict(E2E=float(np.median([r['E2E_s'] for r in rr])),TTFT=float(np.median([r['TTFT_s'] for r in rr])),
            TPOT=float(np.median([r['TPOT_s'] for r in rr if r.get('TPOT_s') is not None])),
            output_tokens=rr[0]['output_tokens'],prompt_tokens=len(rr[0]['prompt_ids'])) for rid,rr in byrequest.items()}
        sources[family,str(case),arm]=dict(run=folder.name,times=times,requests=requests,resident=resident,preparation=preparation)
    for (family,case,arm),r in sources.items():
        baseline='SD' if ARMS[arm]['mode']=='dynamic' else 'S4'
        reference=sources.get((family,case,baseline));overall=sources.get((family,case,'S4'))
        ids=sorted(r['requests']);elapsed=float(np.median(list(r['times'].values())))
        row=dict(family=family,case=case,run=r['run'],arm=arm,GPUs=len(ARMS[arm]['GPUs']),C=len(ids),
            measured_repeats=len(r['times']),median_cohort_makespan_s=elapsed,
            min_cohort_makespan_s=min(r['times'].values()),max_cohort_makespan_s=max(r['times'].values()),
            mean_request_E2E_s=float(np.mean([r['requests'][k]['E2E'] for k in ids])),
            mean_TTFT_s=float(np.mean([r['requests'][k]['TTFT'] for k in ids])),
            mean_TPOT_s=float(np.mean([r['requests'][k]['TPOT'] for k in ids])),
            output_tokens_per_s=sum(v['output_tokens'] for v in r['requests'].values())/elapsed,
            resident_model_cohort_median_s=float(np.median(list(r['resident'].values()))) if len(r['resident'])==len(r['times']) else None,
            cohort_preparation_median_s=float(np.median(list(r['preparation'].values()))) if len(r['preparation'])==len(r['times']) else None,
            resident_model_speedup_vs_mode_Single=None,
            speedup_cohort_vs_mode_Single=None,request_paired_E2E_speedup=None,request_CI95_low=None,request_CI95_high=None,
            overall_cohort_speedup_vs_S4=None,repeat_CI95_low=None,repeat_CI95_high=None,
            status='EXPLORATORY_NUMERICAL_QUALIFICATION_PENDING',
            uncertainty='Request-paired bootstrap uses medians over repeats. C1 has no request CI. Repeat CI uses five repeat indices and is weak.')
        if reference:
            assert set(ids)==set(reference['requests'])
            row['speedup_cohort_vs_mode_Single']=float(np.median(list(reference['times'].values()))/elapsed)
            if row['resident_model_cohort_median_s'] and len(reference['resident'])==len(reference['times']):
                row['resident_model_speedup_vs_mode_Single']=float(np.median(list(reference['resident'].values()))/row['resident_model_cohort_median_s'])
            point,lo,hi=ratio_CI([reference['requests'][k]['E2E'] for k in ids],[r['requests'][k]['E2E'] for k in ids])
            row.update(request_paired_E2E_speedup=point,request_CI95_low=lo,request_CI95_high=hi)
            shared=sorted(r['times'].keys() & reference['times'].keys())
            _,lo,hi=ratio_CI([reference['times'][k] for k in shared],[r['times'][k] for k in shared],False)
            row.update(repeat_CI95_low=lo,repeat_CI95_high=hi)
        if overall:row['overall_cohort_speedup_vs_S4']=float(np.median(list(overall['times'].values()))/elapsed)
        summary.append(row)
    table(ROOT/'replay_summary.csv',summary)
    print('REPLAY_COMPARISONS',len(summary),flush=True)


if __name__=='__main__':main()
