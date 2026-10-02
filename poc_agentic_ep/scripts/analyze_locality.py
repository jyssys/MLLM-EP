"""Fixed controls, layer + aggregate metrics, held-out trajectory bootstrap."""
import argparse
import json
import numpy as np
import pandas as pd
from poc_agentic_ep.contract import ROOT,write
from poc_agentic_ep.metrics import cosine,jsd,coverage,overlap,clustered


def metrics(pred,truth):
    rank_pred=pred.reshape(len(pred),4,32).sum(axis=-1)
    rank_true=truth.reshape(len(truth),4,32).sum(axis=-1)
    rp=rank_pred/np.maximum(rank_pred.sum(axis=-1,keepdims=True),1)
    rt=rank_true/np.maximum(rank_true.sum(axis=-1,keepdims=True),1)
    result=dict(expert_cosine=cosine(pred,truth),expert_jsd=jsd(pred,truth),
        rank_cosine=cosine(rank_pred,rank_true),rank_l1=np.abs(rp-rt).sum(axis=-1),
        hottest_rank_persistence=(rank_pred.argmax(axis=-1)==rank_true.argmax(axis=-1)).astype(float))
    for m in (4,8,16,32):
        result[f'coverage_{m}']=coverage(pred,truth,m)
        result[f'hot_overlap_{m}']=overlap(pred,truth,m)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args()
    root=ROOT/'results'/a.version
    if (root/'locality_summary.json').exists():raise FileExistsError('analysis immutable')
    frozen=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text())
    controls=json.loads((ROOT/'results/frozen_controls.json').read_text())['pairs']
    data=np.load(root/'aggregated.npz');ids=data['request_ids'].tolist()
    # [request,phase,layer,source,expert] -> sum SOURCE axis, not expert axis.
    counts=data['source_expert_counts'].sum(axis=3)
    lookup={r['request_id']:r for r in frozen['rows']};index={rid:i for i,rid in enumerate(ids)}
    calib=[i for i,rid in enumerate(ids) if lookup[rid]['split']=='calibration']
    global_counts=counts[calib].sum(axis=0)
    pairs=[];layer_rows=[]
    for pair in controls:
        target=pair['target'];previous=pair['previous'];ti=index[target]
        for phase_idx,phase in enumerate(('new_prefill','decode')):
            truth=counts[ti,phase_idx]
            predictors={
                'adjacent':[counts[index[previous],phase_idx]],
                'same_shuffled':[counts[index[rid],phase_idx] for rid in pair['nonadjacent_sources']],
                'cross_matched':[counts[index[r['request_id']],phase_idx] for r in pair['cross_trajectory_sources']],
                'global':[global_counts[phase_idx]],'oracle':[truth]}
            for policy,pred in predictors.items():
                evaluations=[metrics(value,truth) for value in pred]
                mean={k:np.mean([r[k] for r in evaluations],axis=0) for k in evaluations[0]}
                aggregate_evaluations=[metrics(value.sum(axis=0,keepdims=True),truth.sum(axis=0,keepdims=True)) for value in pred]
                aggregate={k:float(np.mean([r[k][0] for r in aggregate_evaluations])) for k in aggregate_evaluations[0]}
                pairs.append(dict(trajectory_id=pair['trajectory_id'],target=target,previous=previous,phase=phase,
                    target_turn=lookup[target]['turn'],policy=policy,
                    **{k:float(v.mean()) for k,v in mean.items()},**{'aggregate_'+k:v for k,v in aggregate.items()}))
                for layer in range(48):layer_rows.append(dict(trajectory_id=pair['trajectory_id'],target=target,phase=phase,policy=policy,layer=layer,**{k:float(v[layer]) for k,v in mean.items()}))
    frame=pd.DataFrame(pairs);frame.to_csv(root/'pair_metrics.csv',index=False)
    pd.DataFrame(layer_rows).to_csv(root/'pair_layer_metrics.csv',index=False)
    summary={};comparison={};sensitivity={}
    fields=list(metrics(global_counts[0],global_counts[0]))
    for phase,part in frame.groupby('phase'):
        summary[phase]={};comparison[phase]={};sensitivity[phase]={}
        for policy,g in part.groupby('policy'):
            summary[phase][policy]={k:clustered(g[k],g.trajectory_id) for k in fields}
        previous=part[part.policy=='adjacent'].set_index('target')
        for policy in ('same_shuffled','cross_matched','global'):
            other=part[part.policy==policy].set_index('target').loc[previous.index]
            delta=previous[fields]-other[fields]
            comparison[phase]['adjacent_minus_'+policy]={k:clustered(delta[k],previous.trajectory_id) for k in fields}
            mask=previous.target_turn>=2
            sensitivity[phase]['exclude_initial_predictor_minus_'+policy]={k:clustered(delta.loc[mask,k],previous.loc[mask,'trajectory_id']) for k in fields}
    write(root/'locality_summary.json',dict(phase_metrics=summary,paired_deltas=comparison,
        initial_turn_sensitivity=sensitivity,heldout_trajectories=8,heldout_pairs=64,
        calibration='fixed four disjoint trajectories; CI conditional on this calibration sample',
        primary='new_prefill per-layer metric means; aggregate metrics saved separately in CSV',
        decode='teacher-forced first32 recorded assistant tokens; causal packed forward, not natural autoregressive serving',
        counts_unit='weighted coverage means fraction of next-turn routed assignments, not routing-weight mass'))
    print(json.dumps({'primary_deltas':comparison['new_prefill']['adjacent_minus_global']['coverage_16'],
        'shuffled':comparison['new_prefill']['adjacent_minus_same_shuffled']['expert_cosine'],
        'matched':comparison['new_prefill']['adjacent_minus_cross_matched']['expert_cosine']}))


if __name__=='__main__':main()
