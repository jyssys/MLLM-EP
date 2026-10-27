"""Analyze retained-all validation traces. Prediction of commitment is not edge sensitivity."""
import csv
import json
import numpy as np
from .common import ROOT,write

def auc(scores,labels):
    scores=np.asarray(scores)
    labels=np.asarray(labels,dtype=bool)
    pos=int(labels.sum())
    neg=len(labels)-pos
    if not pos or not neg:
        return None
    unique,inverse,counts=np.unique(scores,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts)
    ranks=(ends-counts+1+ends)/2
    return float((ranks[inverse][labels].sum()-pos*(pos+1)/2)/(pos*neg))

def csvout(path,data):
    if not data:
        return
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def main():
    request_rows=[]
    breakdown=[]
    bins=[]
    for group in ('A','B'):
        base=ROOT/'ep4'/f'{group}_pilot_validation_trace'
        for folder in sorted(base.glob('gsm8k_validation_*')):
            if not all((folder/f'refinement_history.rank{r}.json').exists() for r in range(4)):
                continue
            results=[json.loads((folder/f'trace.result.rank{r}.json').read_text()) for r in range(4)]
            reference=json.loads((folder/'bypass.result.rank0.json').read_text())
            noop=json.loads((folder/'hooked_noop.result.rank0.json').read_text())
            assert all(r['output_ids']==reference['output_ids'] and r['nfe']==reference['nfe'] for r in results+[noop])
            longest=max(range(4),key=lambda r:results[r]['wall_ms'])
            before_counts=np.zeros(4,dtype=np.int64)
            scores=[]
            outcomes=[]
            missing=0
            masked=0
            for rank in range(4):
                meta=json.loads((folder/f'trace.rank{rank}.json').read_text())
                history=json.loads((folder/f'refinement_history.rank{rank}.json').read_text())
                data=np.load(folder/f'trace.rank{rank}.npz')
                prior=np.load(folder/f'refinement_history.rank{rank}.npz')
                by_route={r['route_index']:r for r in history['rows']}
                phases={r['call_id']:r['phase'] for r in meta['events']}
                for index,route in enumerate(meta['routes']):
                    tag=route['tag']
                    ids=data[tag+'_ids'].astype(np.int64)
                    before_counts+=np.bincount((ids//64).flatten(),minlength=4)
                    # One physical layer per model forward avoids 19-fold duplicate labels.
                    if route['layer']!=1 or phases[route['call_id']]!='denoise':
                        continue
                    before=data[tag+'_before'].astype(bool).flatten()
                    after=data[tag+'_after'].astype(bool).flatten()
                    confidence=prior[tag+'_prior_confidence'].flatten()
                    step=prior[tag+'_prior_completed_decode'].flatten()
                    recorded=by_route[index]
                    valid=(step>=0)&np.isfinite(confidence)
                    assert np.all(step[valid]<recorded['completed_decodes_before_dispatch'])
                    eligible=before&valid
                    masked+=int(before.sum())
                    missing+=int((before&~valid).sum())
                    scores.extend(confidence[eligible].tolist())
                    outcomes.extend((~after[eligible]).tolist())
                if rank==longest:
                    stage={name:sum(r['ms'] for r in meta['stages'] if r['stage']==name)
                           for name in ('router','dispatch','expert','combine')}
                    wall=results[rank]['wall_ms']
                    row=dict(sample_id=reference['sample_id'],group=group,longest_observed_wall_rank=rank,
                        observer_wall_ms=wall,denominator='OBSERVER_RANK_WALL_NOT_CLEAN_GLOBAL_CRITICAL_PATH')
                    for name,value in stage.items():
                        row[name+'_ms']=value
                        row[name+'_wall_percent']=100*value/wall
                    row['other_ms']=wall-sum(stage.values())
                    row['other_wall_percent']=100*row['other_ms']/wall
                    row['other_definition']='Residual includes attention, shared experts, dense work, LM head, decode, host gaps and synchronization.'
                    breakdown.append(row)
            scores=np.asarray(scores,dtype=float)
            outcomes=np.asarray(outcomes,dtype=bool)
            edges=int(before_counts.sum())
            request_rows.append(dict(sample_id=reference['sample_id'],group=group,nfe=reference['nfe'],
                output_tokens=reference['generation_tokens'],output_parity=True,nfe_parity=True,
                observed_full_routed_edges=edges,owner_edge_cv=float(before_counts.std()/before_counts.mean()),
                owner_max_mean=float(before_counts.max()/before_counts.mean()),
                masked_position_observations=masked,missing_prior_history=missing,
                eligible_position_observations=len(scores),next_commit_positives=int(outcomes.sum()),
                prior_confidence_commit_auc=auc(scores,outcomes),
                bypass_quality_observer_ms=reference['wall_ms'],trace_wall_ms=max(r['wall_ms'] for r in results),
                observer_to_bypass_ratio=max(r['wall_ms'] for r in results)/reference['wall_ms']))
            boundaries=[0,.5,.8,.9,.95,1.000001]
            for lower,upper in zip(boundaries[:-1],boundaries[1:]):
                use=(scores>=lower)&(scores<upper)
                bins.append(dict(sample_id=reference['sample_id'],group=group,lower=lower,upper=upper,
                    n=int(use.sum()),next_commits=int(outcomes[use].sum()),
                    evidence='TRACE_ANALYSIS_CAUSAL_FEATURE_RETROSPECTIVE_COMMIT_LABEL'))
    macro=[r['prior_confidence_commit_auc'] for r in request_rows if r['prior_confidence_commit_auc'] is not None]
    interval=None
    if macro:
        rng=np.random.default_rng(20261027)
        bootstrap=np.mean(rng.choice(macro,size=(5000,len(macro)),replace=True),axis=1)
        interval=np.percentile(bootstrap,[2.5,97.5]).tolist()
    summary=dict(n_requests=len(request_rows),requests=request_rows,
        macro_prior_confidence_commit_auc=float(np.mean(macro)) if macro else None,
        question_bootstrap_95_auc=interval,auc_question_denominator=len(macro),
        observer_overhead_median_ratio=float(np.median([r['observer_to_bypass_ratio'] for r in request_rows])) if request_rows else None,
        validity='All current routes and expert work retained. All captured history predates the current dispatch.',
        limitations=['Commit prediction is not marginal expert importance. No expert-omission quality test ran.',
                     'Observer stage shares are rank-local. They are not a calibrated clean E2E critical-path decomposition.',
                     'Two EP4 replicas ran concurrently. No primary isolated speedup is measured.'])
    write(ROOT/'pilot/refinement_trace_summary.json',summary)
    csvout(ROOT/'pilot/ep_rank_metrics.csv',request_rows)
    csvout(ROOT/'pilot/observer_breakdown.csv',breakdown)
    csvout(ROOT/'pilot/refinement_confidence_bins.csv',bins)
    print(json.dumps({k:v for k,v in summary.items() if k!='requests'},indent=2))

if __name__=='__main__':
    main()
