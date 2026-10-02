"""Request-grouped discovery analysis; never use independent layer CIs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score

from poc_speculative_moe_ep.contract import ROOT,write
from poc_speculative_moe_ep.analysis import accept_prefix,route_features,critical


def extract(path):
    workers=json.loads((path/'worker.dp0.json').read_text())['records']
    frames=[{(f['wave'],f['forward']):f for f in map(json.loads,(path/f'frames.rank{r}.jsonl').read_text().splitlines())} for r in range(4)]
    out=[];layer_rows=[];parity=[]
    for wave in workers:
        if wave['repeat']<0:continue
        wi=wave['wave'];truth=wave['outputs'][0]['tokens']
        native=[f for (w,i),f in sorted(frames[0].items()) if w==wi and f.get('sampled_tokens') is not None]
        collected=[]
        previous_native_frame=None
        captures=[]
        if wave['lane']=='trace':
            captures=[np.load(path/f'routes.wave{wi}.rank{r}.npz') for r in range(4)]
        for f in native:
            emitted=[x for x in f['sampled_tokens'][0] if x>=0]
            before=len(collected)
            collected.extend(emitted)
            actual_progress=max(0,min(len(emitted),len(truth)-before))
            if before==0:
                assert f['coordinated_dp_tokens'][0]==wave['prompt_tokens'], 'first sampled call is not full prefill'
                previous_native_frame=f
                continue  # Prefill emits first token; not a verify row.
            draft=list(next(iter(f['scheduler']['draft_ids'].values()),[]))
            decoded=accept_prefix(draft,emitted)
            fs=[group[(wi,f['forward'])] for group in frames]
            timing=critical(fs)
            # Critical complete language forward (includes LM head on real DP).
            forward_rank=int(np.argmax([x['target_forward_ms'] for x in fs]))
            timing['target_forward_ms']=fs[forward_rank]['target_forward_ms']
            timing['target_forward_critical_rank']=forward_rank
            common=dict(run=path.name,lane=wave['lane'],request_id=wave['request_id'],
                        gamma=wave['gamma'],repeat=wave['repeat'],wave=wi,forward=f['forward'],
                        committed_position=before,verified_candidate_tokens=len(draft),
                        verified_rows=f['coordinated_dp_tokens'][0],
                        committed_progress=actual_progress,raw_sampler_progress=len(emitted),
                        accepted_draft_tokens=min(decoded['accepted'],actual_progress),
                        raw_accepted_draft_tokens=decoded['accepted'],
                        correction=decoded['correction'],bonus=decoded['bonus'],
                        correction_token_committed=int(wave['gamma']>0 and decoded['correction'] and actual_progress>decoded['accepted']),
                        bonus_token_committed=int(wave['gamma']>0 and decoded['bonus'] and actual_progress>decoded['accepted']),
                        target_token_committed=int(actual_progress>decoded['accepted']),
                        drafted_tokens=len(draft),acceptance=decoded['accepted']/len(draft) if draft else None,
                        draft_token_ids=draft,sampler_token_ids=emitted,
                        draft_ms=previous_native_frame.get('draft_ms',0.),
                        next_draft_ms=f.get('draft_ms',0.),**timing)
            previous_native_frame=f
            common['ep_comm_ms']=common['dispatch_ms']+common['combine_ms']
            assert common['committed_progress']==common['accepted_draft_tokens']+common['target_token_committed']
            if captures:
                features=[]
                for layer in range(48):
                    ids=[captures[r][fs[r]['layers'][layer]['route_key']+'_ids'] for r in range(4)]
                    # Enforce the frozen single-active-DP contract even for old
                    # smoke data preceding the explicit idle-row counter fix.
                    counts=[fs[r]['layers'][layer]['valid_source_rows'] if r<2 else 0 for r in range(4)]
                    value=route_features(ids,counts,decoded['accepted'])
                    assert value['token_rows']==common['verified_rows'], (path,wi,layer,value['token_rows'],common['verified_rows'])
                    features.append(value)
                    layer_rows.append({**{k:common[k] for k in ('run','request_id','gamma','repeat','wave','forward')},'layer':layer,**value})
                for key in ('unique_experts','remote_routed_paths','local_routed_paths','remote_destination_dedup_rows',
                            'rank_fanout','source_destination_edges','max_rank_bytes','mean_rank_bytes',
                            'rank_byte_skew','expert_skew','actual_unique_experts','actual_max_rank_bytes','actual_remote_dispatch_bytes','logical_dispatch_bytes','logical_combine_bytes',
                            'logical_remote_dispatch_bytes','rejected_logical_dispatch_bytes','executed_paths','dummy_padding_paths'):
                    common[key]=float(np.mean([v[key] for v in features]))
                common['expert_union']=sum(v['unique_experts'] for v in features)
                common['max_active_experts_rank']=float(np.mean([max(v['active_experts_per_rank']) for v in features]))
                common['actual_max_active_experts_rank']=float(np.mean([max(v['actual_active_experts_per_rank']) for v in features]))
                common['actual_expert_histogram_skew']=float(np.mean([max(v['actual_expert_histogram'])/(sum(v['actual_expert_histogram'])/128) for v in features]))
                common['active_expert_weight_bytes']=common['expert_union']*9437184
                common['expert_histogram_skew']=float(np.mean([max(v['expert_histogram'])/(sum(v['expert_histogram'])/128) for v in features]))
                common['max_bytes_per_layer']=max(v['max_rank_bytes'] for v in features)
            out.append(common)
        parity.append(dict(run=path.name,request_id=wave['request_id'],repeat=wave['repeat'],gamma=wave['gamma'],
            tokens=truth,finish_reason=wave['outputs'][0]['finish_reason'],host_wave_ms=wave['host_wave_ms'],
            actual_request_completion_ms=wave.get('actual_request_completion_ms'),
            trace_sampler_reconstruction_equal=collected[:len(truth)]==truth,
            trace_uncapped_sampler_tokens=len(collected),actual_output_tokens=len(truth)))
        for capture in captures:capture.close()
    return out,layer_rows,parity


def bootstrap_mean(values,n=2000):
    values=np.asarray(values,float)
    if not len(values):return None
    rng=np.random.default_rng(3407)
    draws=values[rng.integers(len(values),size=(n,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                independent_unit='request',n=len(values),bootstrap=n)


def cost_models(table,width_fixed_effects=False):
    if table.request_id.nunique()<8:return dict(status='INSUFFICIENT_REQUESTS',n=table.request_id.nunique())
    table=table.copy()
    table['acceptance_feature']=table['acceptance'].fillna(1.)
    families=[['gamma','verified_rows','committed_progress','accepted_draft_tokens','acceptance_feature'],
        ['unique_experts','expert_union','max_active_experts_rank','expert_skew','expert_histogram_skew'],
        ['remote_routed_paths','remote_destination_dedup_rows','rank_fanout','max_rank_bytes',
         'rank_byte_skew','source_destination_edges','max_bytes_per_layer','actual_max_rank_bytes','actual_remote_dispatch_bytes']]
    # Existing expert-count baseline must see actual idle/padded expert work,
    # so M2 cannot win merely because M1 omitted runtime dummy execution.
    families[1]+=['actual_unique_experts','executed_paths','dummy_padding_paths']
    if 'actual_max_active_experts_rank' in table:
        families[1]+=['actual_max_active_experts_rank','actual_expert_histogram_skew']
    if width_fixed_effects:
        # A nonlinear width-only baseline prevents traffic features from
        # winning merely by encoding four different collective batch sizes.
        for width in (0,2,4,8):
            name=f'width_is_{width}'
            table[name]=(table.gamma==width).astype(float)
            families[0].append(name)
    result={};features=[]
    folds=list(GroupKFold(5).split(table,groups=table.request_id))
    for i,extra in enumerate(families):
        features+=extra
        result[f'M{i}']={}
        for target in ('ep_comm_ms','moe_ms','target_forward_ms'):
            truth=table[target].to_numpy();pred=np.zeros(len(table))
            for train,test in folds:
                assert not set(table.iloc[train].request_id)&set(table.iloc[test].request_id)
                model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                model.fit(table.iloc[train][features],truth[train]);pred[test]=model.predict(table.iloc[test][features])
            tmp=pd.DataFrame(dict(request_id=table.request_id.values,error=np.abs(pred-truth)))
            request_mae=tmp.groupby('request_id').error.mean()
            result[f'M{i}'][target]=dict(mae=float(mean_absolute_error(truth,pred)),r2=float(r2_score(truth,pred)),
                request_weighted_mae=bootstrap_mean(request_mae.values),features=list(features),
                oof_predictions=pred.tolist())
    a=result['M1']['ep_comm_ms'];b=result['M2']['ep_comm_ms']
    result['ep_mae_reduction_pct']=100*(a['mae']-b['mae'])/a['mae']
    result['ep_r2_increment']=b['r2']-a['r2']
    errors1=np.abs(np.asarray(a['oof_predictions'])-table.ep_comm_ms.to_numpy())
    errors2=np.abs(np.asarray(b['oof_predictions'])-table.ep_comm_ms.to_numpy())
    paired=pd.DataFrame(dict(request_id=table.request_id.values,m1=errors1,m2=errors2)).groupby('request_id').mean()
    result['paired_request_mae_improvement_ms']=bootstrap_mean((paired.m1-paired.m2).values)
    result['folds']=[dict(train_requests=sorted(set(table.iloc[train].request_id)),test_requests=sorted(set(table.iloc[test].request_id))) for train,test in folds]
    result['family']='fixed standardized ridge alpha=1; request-grouped 5-fold CV; no target tuning'
    result['width_fixed_effects']=width_fixed_effects
    return result


def matched_pairs(table):
    """Select by pre-timing covariates only, not by largest latency gap."""
    pairs=[]
    groups=table.groupby(['gamma','verified_rows','committed_progress'])
    for key,g in groups:
        if key[0]==0:continue
        rows=g.to_dict('records')
        # At most one pair per source request in each width/progress stratum.
        used=set()
        for low in sorted(rows,key=lambda r:r['max_rank_bytes']):
            if low['request_id'] in used:continue
            candidates=[hi for hi in rows if hi['request_id']!=low['request_id']
                and hi['request_id'] not in used
                and abs(hi['unique_experts']-low['unique_experts'])<=.5
                and abs(hi['actual_unique_experts']-low['actual_unique_experts'])<=.5
                and hi['max_rank_bytes']>=low['max_rank_bytes']*1.15]
            if not candidates:continue
            hi=min(candidates,key=lambda r:(abs(r['unique_experts']-low['unique_experts']),r['max_rank_bytes']))
            used.update([low['request_id'],hi['request_id']])
            pairs.append(dict(gamma=int(key[0]),verified_rows=int(key[1]),committed_progress=int(key[2]),
                low_request=low['request_id'],high_request=hi['request_id'],
                low_forward=int(low['forward']),high_forward=int(hi['forward']),
                unique_expert_difference=hi['unique_experts']-low['unique_experts'],
                traffic_delta_pct=100*(hi['max_rank_bytes']/low['max_rank_bytes']-1),
                ep_delta_pct=100*(hi['ep_comm_ms']/low['ep_comm_ms']-1)))
    return dict(status='INSUFFICIENT_MATCHED_PAIRS' if len(pairs)<10 else 'DIAGNOSTIC_TRACE_PAIRS',
                n=len(pairs),pairs=pairs,median_ep_delta_pct=float(np.median([r['ep_delta_pct'] for r in pairs])) if pairs else None,
                matching='same gamma/verified rows/committed; user and executed unique experts within .5/layer; traffic >=15%; no latency selection')


def depth_disagreement(table):
    choices=[]
    for rid,g in table.groupby('request_id'):
        totals=g.groupby('gamma')[['expert_union','ep_comm_ms','target_forward_ms','committed_progress']].sum()
        if set(totals.index)!={0,2,4,8}:continue
        cost=totals.div(totals.committed_progress,axis=0)
        choices.append(dict(request_id=rid,expert_proxy_gamma=int(cost.expert_union.idxmin()),
            ep_gamma=int(cost.ep_comm_ms.idxmin()),target_gamma=int(cost.target_forward_ms.idxmin()),
            per_gamma={str(k):dict(expert_union_per_progress=float(v.expert_union),ep_ms_per_progress=float(v.ep_comm_ms),
                target_ms_per_progress=float(v.target_forward_ms)) for k,v in cost.iterrows()}))
    flags=[int(x['expert_proxy_gamma']!=x['ep_gamma']) for x in choices]
    return dict(status='DIAGNOSTIC_REQUEST_AGGREGATE_NOT_OPPORTUNITY_ORACLE',choices=choices,
                disagreement_fraction=float(np.mean(flags)) if flags else None,
                uncertainty=bootstrap_mean(flags),gamma0='target-only AR; not SD with one draft proposal',
                scope='same request/greedy output across arms, but not aligned per-opportunity alternatives')


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--version',required=True)
    a=p.parse_args();root=ROOT/'artifacts'/a.version
    if root.exists():raise FileExistsError(root)
    records=[];layers=[];parity=[]
    for name in a.runs:
        path=ROOT/'raw'/name
        if not json.loads((path/'status.json').read_text())['ok']:raise ValueError('failed run '+name)
        r,l,q=extract(path);records+=r;layers+=l;parity+=q
    root.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(root/'verification_metrics.parquet',index=False)
    pd.DataFrame(layers).to_parquet(root/'route_metrics.parquet',index=False)
    write(root/'parity.json',parity)
    grouped=defaultdict(list)
    for row in parity:grouped[row['request_id']].append(row)
    comparisons=[]
    for rid,rows in grouped.items():
        base=next((x for x in rows if x['gamma']==0),None)
        if base:
            for row in rows:
                comparisons.append(dict(request_id=rid,gamma=row['gamma'],run=row['run'],equal=row['tokens']==base['tokens']))
    write(root/'sd_equivalence.json',dict(comparisons=comparisons,all_equal=all(x['equal'] for x in comparisons),
        request_count=len(grouped),sampler_reconstruction_all_equal=all(x['trace_sampler_reconstruction_equal'] for x in parity)))
    table=pd.DataFrame(records)
    models=cost_models(table[table.lane=='trace']) if 'unique_experts' in table else dict(status='NO_ROUTE_TRACE')
    write(root/'cost_model_results.json',models)
    if 'unique_experts' in table:write(root/'matched_pair_analysis.json',matched_pairs(table[table.lane=='trace']))
    if 'unique_experts' in table:write(root/'depth_disagreement.json',depth_disagreement(table[table.lane=='trace']))
    summaries=[]
    for gamma,g in table.groupby('gamma'):
        rows=[q for q in parity if q['gamma']==gamma]
        progress=g.committed_progress.sum();drafts=g.drafted_tokens.sum()
        summaries.append(dict(gamma=int(gamma),requests=len(rows),verifications=len(g),
            acceptance=float(g.raw_accepted_draft_tokens.sum()/drafts) if drafts else None,
            capped_committed_acceptance=float(g.accepted_draft_tokens.sum()/drafts) if drafts else None,
            committed_per_verify=float(progress/len(g)),
            ep_ms_per_committed=float(g.ep_comm_ms.sum()/progress),
            target_ms_per_committed=float(g.target_forward_ms.sum()/progress),
            host_wave_drain_ms_per_output_token=sum(q['host_wave_ms'] for q in rows)/sum(len(q['tokens']) for q in rows),
            e2e_ms_per_output_token=sum(q['actual_request_completion_ms'] for q in rows)/sum(len(q['tokens']) for q in rows) if all(q['actual_request_completion_ms'] is not None for q in rows) else None,
            ep_fraction_of_target=float(g.ep_comm_ms.sum()/g.target_forward_ms.sum()),
            mean_unique_experts=float(g.unique_experts.mean()) if 'unique_experts' in g else None,
            mean_remote_paths=float(g.remote_routed_paths.mean()) if 'remote_routed_paths' in g else None,
            mean_rank_fanout=float(g.rank_fanout.mean()) if 'rank_fanout' in g else None,
            mean_max_rank_bytes=float(g.max_rank_bytes.mean()) if 'max_rank_bytes' in g else None,
            rejected_ep_byte_fraction=float((g.rejected_logical_dispatch_bytes*g.committed_progress.ge(0)).sum()/g.logical_dispatch_bytes.sum()) if 'logical_dispatch_bytes' in g else None,
            timing_lane=sorted(set(g.lane)),scope='post-prefill verifies; E2E includes prefill/controller/draft; trace timing diagnostic only'))
    write(root/'gamma_summary.json',summaries)
    print(json.dumps(dict(summary=summaries,equivalence=comparisons,
                         m2_mae_reduction=models.get('ep_mae_reduction_pct')),indent=2))


if __name__=='__main__':main()
