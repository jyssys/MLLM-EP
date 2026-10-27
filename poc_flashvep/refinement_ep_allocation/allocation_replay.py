"""Fixed-trajectory allocation accounting. This replay does not measure speed or quality."""
import json
import numpy as np
from .common import ROOT,write
from .refinement_analysis import csvout

def deletion(weights,ids,masked,prior,policy,fraction,call,layer):
    n=len(ids)
    score=weights.astype(float)/np.maximum(weights.sum(-1,keepdims=True),1e-12)
    if policy in ('P3','P5'):
        c=np.nan_to_num(prior,nan=0.)
        sigmoid=lambda v:1/(1+np.exp(np.clip(-v,-80,80)))
        frontier=sigmoid((c-.7)/.05)*sigmoid((.95-c)/.05)
        frontier=np.where(np.isfinite(prior),frontier,0.)
        score*=np.where(masked,1+frontier,.75)[:,None]
    if policy in ('P4','P5'):
        demand=np.bincount((ids//64).flatten(),minlength=4)
        score/=np.maximum(demand[ids//64]/demand.mean(),1e-12)**2
    removed=np.zeros(ids.shape,dtype=bool)
    # Match the live diagnostic's local work budgets and protected slot exactly.
    local=n//4
    for rank in range(4):
        start=rank*local
        end=start+local
        part=score[start:end].copy()
        row=np.arange(local)
        part[row,(row+rank+layer+call+1)%8]=np.inf
        count=int(part.size*fraction)
        selected=np.argsort(part.flatten(),kind='stable')[:count]
        removed[start+selected//8,selected%8]=True
    assert int(removed.sum())==4*int((n//4)*8*fraction)
    assert np.all(removed.sum(-1)<=7)
    return removed

def accounting(ids,removed):
    owners=ids//64
    source=np.repeat(np.arange(4),len(ids)//4)
    load=[]
    tiles=[]
    remote=0
    for rank in range(4):
        selected=(owners==rank)&~removed
        hist=np.bincount(ids[selected]-64*rank,minlength=64)
        received=int(selected.any(1).sum())
        block=16 if received<=64 else 64
        tiles.append(int((((hist+block-1)//block)*block).sum()))
        load.append(int(hist.sum()))
        remote+=int((selected.any(1)&(source!=rank)).sum())
    return dict(edges=sum(load),remote_pairs=remote,max_rank_edges=max(load),
        max_rank_padded_rows=max(tiles),total_padded_rows=sum(tiles),
        owner_cv=float(np.std(load)/np.mean(load)))

def main():
    observations=[]
    for group in ('A','B'):
        for folder in sorted((ROOT/'ep4'/f'{group}_pilot_validation_trace').glob('gsm*')):
            if not (folder/'trace.rank3.npz').exists():
                continue
            metadata=[json.loads((folder/f'trace.rank{rank}.json').read_text()) for rank in range(4)]
            data=[np.load(folder/f'trace.rank{rank}.npz') for rank in range(4)]
            history=[np.load(folder/f'refinement_history.rank{rank}.npz') for rank in range(4)]
            phase={row['call_id']:row['phase'] for row in metadata[0]['events']}
            calls=sorted(phase)
            # Frozen evenly spaced forward sample. No quality-label selection.
            selected=set(np.asarray(calls)[np.linspace(0,len(calls)-1,min(24,len(calls))).astype(int)].tolist())
            for index,route in enumerate(metadata[0]['routes']):
                if route['call_id'] not in selected:
                    continue
                tag=route['tag']
                ids=np.concatenate([arr[tag+'_ids'] for arr in data]).astype(np.int64)
                weights=np.concatenate([arr[tag+'_weights'] for arr in data])
                masked=np.concatenate([arr[tag+'_before'] for arr in data]).astype(bool)
                prior=np.concatenate([arr[tag+'_prior_confidence'] for arr in history])
                before=accounting(ids,np.zeros_like(ids,dtype=bool))
                for fraction in (.05,.20):
                    masks={}
                    for policy in ('P2','P3','P4','P5'):
                        mask=deletion(weights,ids,masked,prior,policy,fraction,route['call_id'],route['layer'])
                        masks[policy]=mask
                        after=accounting(ids,mask)
                        observations.append(dict(sample_id=folder.name,group=group,phase=phase[route['call_id']],
                            call_id=route['call_id'],layer=route['layer'],fraction=fraction,policy=policy,
                            before=before,after=after,
                            p3_selection_differs=(bool(np.any(mask!=masks['P3'])) if 'P3' in masks else None),
                            evidence='TRACE_ESTIMATE_FIXED_BASELINE_TRAJECTORY_NOT_GPU_SPEEDUP',
                            ep_signal='Current global destination counts. Additional online synchronization is NOT_MEASURED.'))
            for arr in data+history:
                arr.close()
    aggregates=[]
    for fraction in (.05,.20):
        for policy in ('P2','P3','P4','P5'):
            chosen=[r for r in observations if r['fraction']==fraction and r['policy']==policy]
            if not chosen:
                continue
            out=dict(policy=policy,requested_fraction=fraction,request_n=len({r['sample_id'] for r in chosen}),
                sampled_physical_moe_calls=len(chosen),evidence='TRACE_ESTIMATE_NOT_LATENCY_OR_QUALITY')
            for metric in ('edges','remote_pairs','max_rank_edges','max_rank_padded_rows','total_padded_rows'):
                out[metric+'_reduction']=1-sum(r['after'][metric] for r in chosen)/sum(r['before'][metric] for r in chosen)
            out['mean_owner_cv_before']=float(np.mean([r['before']['owner_cv'] for r in chosen]))
            out['mean_owner_cv_after']=float(np.mean([r['after']['owner_cv'] for r in chosen]))
            aggregates.append(out)
    write(ROOT/'pilot/allocation_replay.json',dict(aggregates=aggregates,
        sampling='24 evenly spaced model forwards per request, all 19 MoE layers and four ranks.',
        limitations=['Baseline trajectory fixed. Omissions can alter later routes, quality and NFE.',
            'NumPy score arithmetic and stable sorting form a trace proxy, not a bit-exact live selector replay.',
            'P4/P5 use current global rank counts. Online exchange and selector costs are not free.',
            'Padded rows follow audited fallback tiles. They are not measured instructions or time.']))
    csvout(ROOT/'pilot/allocation_replay.csv',aggregates)
    with (ROOT/'pilot/allocation_replay_samples.jsonl').open('w') as stream:
        for row in observations:
            stream.write(json.dumps(row)+'\n')
    print(json.dumps(aggregates,indent=2))

if __name__=='__main__':
    main()
