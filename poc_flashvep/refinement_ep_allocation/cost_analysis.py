"""Analyze isolated physical operator costs without inventing request speedups."""
import json
import numpy as np
from .common import ROOT,rows,write
from .refinement_analysis import csvout

def interval_ratio(old,new):
    old=np.asarray(old,dtype=float)
    new=np.asarray(new,dtype=float)
    rng=np.random.default_rng(20261028)
    indices=rng.integers(0,len(old),size=(5000,len(old)))
    samples=old[indices].mean(1)/new[indices].mean(1)
    return np.percentile(samples,[2.5,97.5]).tolist()

def padded(geometry):
    ids=np.asarray(geometry['received_ids'],dtype=int)
    block=16 if geometry['received_rows']<=64 else 64
    count=np.bincount(ids[ids>=0],minlength=64)
    return int((np.ceil(count/block)*block).sum())

def main():
    latency=[]
    latency_samples=[]
    for group in ('A','B'):
        destination=ROOT/'ep4'/f'{group}_isolated_operator_cost'
        data=[]
        for path in destination.glob('gsm*_repeat*/bypass.result.rank0.json'):
            repeat=int(path.parent.name.rsplit('_repeat',1)[1])
            if repeat<3:
                continue
            record=json.loads(path.read_text())
            data.append((repeat,record))
        data.sort(key=lambda pair:pair[0])
        if data:
            old=data[0][1]
            assert all(r['output_ids']==old['output_ids'] and r['nfe']==old['nfe'] for _,r in data)
            times=[r['request_preparation_to_text_ms'] for _,r in data]
            rng=np.random.default_rng(20261028)
            interval=np.percentile(rng.choice(times,size=(5000,len(times)),replace=True).mean(1),[2.5,97.5])
            latency.append(dict(method='P0_FULL',group=group,unique_requests=1,repetitions=len(data),
                mean_e2e_ms=float(np.mean(times)),p50_e2e_ms=float(np.median(times)),
                p90_e2e_ms=float(np.percentile(times,90)),p95_e2e_ms=float(np.percentile(times,95)),
                std_e2e_ms=float(np.std(times,ddof=1)) if len(times)>1 else 0.,
                repeat_bootstrap_95_mean_ms=interval.tolist(),mean_nfe=float(np.mean([r['nfe'] for _,r in data])),
                output_tokens=old['generation_tokens'],prompt_tokens=old['prompt_tokens'],
                output_tokens_per_second=1000*sum(r['generation_tokens'] for _,r in data)/sum(times),
                requests_per_second=1000*len(times)/sum(times),
                peak_rank0_hbm_gib=max(r['peak_allocated_bytes'] for _,r in data)/2**30,
                exact_repeat_output_parity=True,ttft_ms='NOT_MEASURED',candidate_speedup='NOT_MEASURED',
                evidence='MEASURED_WALL_ISOLATED_EP4_REQUEST',
                scope='One natural validation prompt. One server start per group. Repeat CI is not a workload-population CI.',
                boundary='Rank-0 request preparation, generation and text decode. Excludes load, warm-up, quiet-lane checks and artifact I/O.'))
            for repeat,r in data:
                latency_samples.append(dict(group=group,repeat=repeat,sample_id=r['sample_id'],method='P0_FULL',
                    wall_ms=r['request_preparation_to_text_ms'],generate_wall_ms=r['generate_wall_ms'],nfe=r['nfe'],
                    output_tokens=r['generation_tokens'],evidence='MEASURED_WALL_ISOLATED_EP4_REQUEST'))
    write(ROOT/'pilot/isolated_baseline_summary.json',dict(conditions=latency,
        note='No candidate request executes in this timing experiment. No speedup is measured.'))
    csvout(ROOT/'pilot/isolated_baseline_summary.csv',latency)
    csvout(ROOT/'latency_samples.csv',latency_samples)
    raw=[]
    observer=[]
    for group in ('A','B'):
        clean=ROOT/'ep4'/f'{group}_clean_operator_cost'
        source=clean if len(list(clean.glob('operator_complete.rank*.json')))==4 else ROOT/'ep4'/f'{group}_isolated_operator_cost'
        for path in source.glob('operator_samples.rank*.jsonl'):
            for row in rows(path):
                row=dict(row,group=group)
                if row.get('timing_lane')=='CLEAN_OPERATOR':
                    raw.append(row)
                elif row.get('timing_lane')=='REGION_OBSERVER':
                    observer.append(row)
                else:
                    # Interim legacy observer rows stay explicitly labeled.
                    row['timing_lane']='LEGACY_REGION_OBSERVER_NOT_CLEAN_OPERATOR'
                    raw.append(row)
                    observer.append(row)
    summaries=[]
    costs={}
    for row in raw:
        key=(row['group'],row['layer'],row['full_rows'],row['method'],row['repeat'])
        costs.setdefault(key,[]).append(row)
    for group,layer,shape in sorted({k[:3] for k in costs}):
        full={repeat:max(r['wall_ms'] for r in rs) for (g,l,s,m,repeat),rs in costs.items()
            if (g,l,s,m)==(group,layer,shape,'FULL') and len(rs)==4}
        if not full:
            continue
        for method in ('FULL','PRECOMPUTED_ROUTER_05','PRECOMPUTED_ROUTER_20'):
            selected={repeat:rs for (g,l,s,m,repeat),rs in costs.items()
                if (g,l,s,m)==(group,layer,shape,method) and len(rs)==4 and repeat in full}
            if not selected:
                continue
            ordered=sorted(selected)
            wall=[max(r['wall_ms'] for r in selected[i]) for i in ordered]
            old=[full[i] for i in ordered]
            one=selected[ordered[0]]
            edges=sum(r['geometry']['active_source_edges'] for r in one)
            tile=sum(padded(r['geometry']) for r in one)
            original=next(rs for (g,l,s,m,i),rs in costs.items()
                if (g,l,s,m)==(group,layer,shape,'FULL') and len(rs)==4)
            tile0=sum(padded(r['geometry']) for r in original)
            summary=dict(group=group,layer=layer,full_rows=shape,method=method,n_repetitions=len(wall),
                timing_lane=one[0]['timing_lane'],
                mean_operator_wall_ms=float(np.mean(wall)),p50_operator_wall_ms=float(np.median(wall)),
                p95_operator_wall_ms=float(np.percentile(wall,95)),
                baseline_ratio_of_mean_operator_wall=float(np.mean(old)/np.mean(wall)),
                paired_repeat_bootstrap_95_ratio=interval_ratio(old,wall),
                actual_routed_edge_reduction=1-edges/(shape*8),
                actual_received_rows=sum(r['geometry']['received_rows'] for r in one),
                source_derived_padded_expert_rows=tile,source_derived_padded_row_reduction=1-tile/tile0,
                evidence='MEASURED_PHYSICAL_TRUE_EP4_OPERATOR_NOT_REQUEST_E2E',
                mask='Precomputed exact current-router mask. No online selector or confidence-softmax cost.',
                memory='Original local BF16 Mini experts. Original source activation shape.')
            for stage in ('dispatch','expert','combine','mask_materialization'):
                observed={}
                for r in observer:
                    if (r['group'],r['layer'],r['full_rows'],r['method'])==(group,layer,shape,method):
                        observed.setdefault(r['repeat'],[]).append(r)
                values=[max(r['stages_ms'][stage] for r in rs) for rs in observed.values() if len(rs)==4]
                summary[f'mean_max_rank_{stage}_ms']=float(np.mean(values)) if values else None
            summary['stage_timing_scope']='Separate REGION_OBSERVER pass. CUDA intervals include submission gaps. Not the clean operator wall pass.'
            summaries.append(summary)
    # Observer regions are not a clean critical path. Keep the projection conditional.
    breakdown=ROOT/'pilot/observer_breakdown.csv'
    projections=[]
    if breakdown.exists():
        import csv
        observed=list(csv.DictReader(breakdown.open()))
        wall=sum(float(r['observer_wall_ms']) for r in observed)
        for row in summaries:
            if row['method']=='FULL':
                continue
            baseline=next(r for r in summaries if (r['group'],r['layer'],r['full_rows'],r['method'])==
                (row['group'],row['layer'],row['full_rows'],'FULL'))
            region=sum(sum(float(r[stage+'_ms']) for stage in ('dispatch','expert','combine')) for r in observed)/wall
            old=sum(baseline['mean_max_rank_'+s+'_ms'] for s in ('dispatch','expert','combine'))
            new=sum(row['mean_max_rank_'+s+'_ms'] for s in ('dispatch','expert','combine'))
            gain=region*(1-new/old)
            projections.append(dict(group=row['group'],layer=row['layer'],full_rows=row['full_rows'],method=row['method'],
                conditional_observer_based_e2e_reduction_percent=100*gain,
                evidence='TRACE_ESTIMATE_CONDITIONAL_NOT_CALIBRATED_CLEAN_E2E',
                assumptions='Observer shares transfer to clean path. Component envelopes add. NFE and trajectories do not change. Online policy is free.',
                limitations='These assumptions are unvalidated. Do not use this projection as measured speedup or a conservative guarantee.'))
    write(ROOT/'pilot/operator_cost_summary.json',dict(conditions=summaries,conditional_projections=projections,
        candidate_request_speedup='NOT_MEASURED',calibrated_request_cost_model='NOT_VALIDATED',
        note='Rank maximum of synchronized operator wall intervals. Stage envelopes can overlap and must not be summed as request E2E.'))
    csvout(ROOT/'pilot/operator_cost_summary.csv',summaries)
    csvout(ROOT/'pilot/conditional_cost_projections.csv',projections)
    print(json.dumps(dict(conditions=summaries,conditional_projections=projections),indent=2))

if __name__=='__main__':
    main()
