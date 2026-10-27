"""Held-out route statistics. Expert-set probabilities are multi-label probabilities."""
from collections import defaultdict
import gzip
import json
import numpy as np
from .common import ROOT,write
from .aggregate_quality import csv_write

def rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)

def entropy(p):
    p=np.asarray(p,dtype=float);p=p[p>0]
    return float(-(p*np.log2(p)).sum())

def coactivation(ids,e):
    ids=np.asarray(ids,dtype=int).reshape(-1,2)
    matrix=np.zeros((e,e),dtype=np.int64)
    np.add.at(matrix,(np.minimum(ids[:,0],ids[:,1]),np.maximum(ids[:,0],ids[:,1])),1)
    return matrix

def slot_null(ids,rng):
    """Swap second slots. Preserve both slot marginals and distinct top-2 choices."""
    out=np.asarray(ids,dtype=int).copy()
    for _ in range(12):
        order=rng.permutation(len(out));n=len(order)//2
        left=order[:2*n:2];right=order[1:2*n:2]
        legal=(out[left,0]!=out[right,1])&(out[right,0]!=out[left,1])
        left=left[legal];right=right[legal]
        out[left,1],out[right,1]=out[right,1].copy(),out[left,1].copy()
    assert np.all(out[:,0]!=out[:,1])
    return out

def tv(a,b):
    return float(np.abs(a/a.sum()-b/b.sum()).sum()/2)

def top_recall(prob,truth,k):
    selected=np.argsort(-prob,axis=1,kind='stable')[:,:min(k,prob.shape[1])]
    return np.mean((selected[:,:,None]==truth[:,None,:]).any(axis=1).sum(axis=1)/2)

def set_information(previous,next_ids,e):
    x=np.sort(previous,axis=1);y=np.sort(next_ids,axis=1)
    x=x[:,0]*e+x[:,1];y=y[:,0]*e+y[:,1]
    _,nx=np.unique(x,return_counts=True);_,ny=np.unique(y,return_counts=True)
    codes,count=np.unique(x*(e*e)+y,return_counts=True)
    xcount=dict(zip(*np.unique(x,return_counts=True)));ycount=dict(zip(*np.unique(y,return_counts=True)))
    mi=sum(n/len(x)*np.log2(n*len(x)/(xcount[int(code)//(e*e)]*ycount[int(code)%(e*e)])) for code,n in zip(codes,count))
    hx,hy=entropy(nx/len(x)),entropy(ny/len(y))
    return float(mi),float(mi/max(np.sqrt(hx*hy),1e-12))

def fit_transition(previous,next_ids,e):
    count=np.zeros((e,e));den=np.zeros(e)
    for i in range(2):
        np.add.at(den,previous[:,i],1)
        for j in range(2):np.add.at(count,(previous[:,i],next_ids[:,j]),1)
    pop=np.bincount(next_ids.ravel(),minlength=e)/len(next_ids)
    conditional=(count+10*pop[None,:])/(den[:,None]+10)
    assert np.allclose(conditional.sum(1),2)
    return conditional,pop

def analyze_model(model,path,phase='prefill'):
    groups=defaultdict(list);positions=defaultdict(list);tracks=defaultdict(dict)
    metadata=json.loads((ROOT/'raw'/f'{model}_native_fa3_smoke_v1'/'metadata.rank0.json').read_text())
    recurrent={site['site'] for site in metadata['physical_moe_sites'] if site['recurrent']}
    for row in rows(path):
        if row['phase']!=phase:continue
        context=row.get('original_prefix_tokens',row['context']) if phase=='decode' else row['context']
        key=(row['batch'],context,row['domain'],row['site'],row['loop'],row['split'])
        groups[key].append((row['request_id'],np.array(row['top2'],dtype=np.int16)))
        positions[key].append(np.asarray(row['positions'],dtype=np.int64)+row['batch_row']*row['context'])
        if row['site'] in recurrent:
            t=(row['batch'],context,row['domain'],row['site'],row['split'],row['request_id'],row['step'])
            # One capture per request and shape. Never pair unrelated requests or tokens.
            assert row['loop'] not in tracks[t]
            tracks[t][row['loop']]=row
    if phase=='decode':
        # Consecutive generated tokens belong to one request, not independent prompts.
        for key,values in groups.items():
            pooled=defaultdict(list);pooled_positions=defaultdict(list)
            for (request,ids),pos in zip(values,positions[key]):
                pooled[request].append(ids);pooled_positions[request].append(pos)
            groups[key]=[(request,np.concatenate(values)) for request,values in pooled.items()]
            positions[key]=[np.concatenate(pooled_positions[request]) for request,_ in groups[key]]
    pair_rows=[];transition_rows=[];aggregate=[]
    rng=np.random.default_rng(20261026)
    for key,values in groups.items():
        b,length,domain,site,loop,split=key
        if split!='heldout':continue
        cal=groups.get((b,length,domain,site,loop,'calibration'))
        if not cal:continue
        e=25 if model.startswith('ifm') else (64 if site in recurrent else 8)
        ids=np.concatenate([v for _,v in values]);train=np.concatenate([v for _,v in cal])
        counts=coactivation(ids,e);cal_counts=coactivation(train,e)
        nulls=[coactivation(slot_null(ids,rng),e) for _ in range(12)]
        null=np.mean(nulls,axis=0)
        marg=np.bincount(ids.ravel(),minlength=e)/(2*len(ids))
        chosen=np.unravel_index(cal_counts.argmax(),cal_counts.shape)
        prompt_fraction=[np.mean(np.all(np.sort(v,axis=1)==chosen,axis=1)) for _,v in values]
        frequency_boot=np.mean(rng.choice(prompt_fraction,size=(1000,len(values)),replace=True),axis=1)
        pair_rows.append(dict(model=model,phase=phase,batch=b,context=length,domain=domain,site=site,loop=loop,
            context_definition='Base prompt length, decode rows from 31 successive generated-token steps' if phase=='decode' else 'Actual prefill length',
            recurrent=site in recurrent,sampled_token_invocations=len(ids),heldout_prompts=len(values),calibration_prompts=len(cal),
            pair_tv_from_slot_null=tv(counts+.01,null+.01),heldout_calibration_tv=tv(counts+.01,cal_counts+.01),
            max_pair_fraction=float(counts.max()/len(ids)),marginal_entropy_bits=entropy(marg),
            max_pair_lift=float(((counts+1)/(null+1)).max()),
            calibration_top_pair=str(chosen),heldout_top_pair_fraction=float(np.mean(prompt_fraction)),
            heldout_top_pair_ci95_low=float(np.quantile(frequency_boot,.025)),heldout_top_pair_ci95_high=float(np.quantile(frequency_boot,.975)),
            null_replicates=12,evidence='LIVE_TRACE_STATISTIC_NOT_E2E_SPEEDUP'))
        if b==4 and length==512:
            write(ROOT/'figures'/f'{model}_{phase}_{site.replace(".","_")}_loop{loop}_coactivation.json',
                dict(matrix=counts.tolist(),null=null.tolist(),calibration=cal_counts.tolist(),model=model,domain=domain,
                    note='Figure input only. Domain-specific cells use a separate filename below.')) if domain=='general' else None
    # Pool requests only within the same B, length, domain, site and pass pair.
    transitions=defaultdict(lambda:defaultdict(list))
    for key,passes in tracks.items():
        b,length,domain,site,split,request,step=key
        for loop,row in passes.items():
            for lag in [1,2,4]:
                if loop+lag not in passes:continue
                nxt=passes[loop+lag]
                assert row['positions']==nxt['positions']
                transitions[(b,length,domain,site,loop,lag)][split].append((request,
                    np.asarray(row['top2'],dtype=int),np.asarray(nxt['top2'],dtype=int)))
    for key,by_split in transitions.items():
        if not all(s in by_split for s in ['calibration','heldout']):continue
        b,length,domain,site,loop,lag=key
        if phase=='decode':
            for split,items in by_split.items():
                pooled=defaultdict(lambda:[[],[]])
                for request,previous,nxt in items:pooled[request][0].append(previous);pooled[request][1].append(nxt)
                by_split[split]=[(request,np.concatenate(items[0]),np.concatenate(items[1])) for request,items in pooled.items()]
        e=25 if model.startswith('ifm') else 64
        train_prev=np.concatenate([x[1] for x in by_split['calibration']]);train_next=np.concatenate([x[2] for x in by_split['calibration']])
        test_prev=np.concatenate([x[1] for x in by_split['heldout']]);test_next=np.concatenate([x[2] for x in by_split['heldout']])
        cond,pop=fit_transition(train_prev,train_next,e)
        prob=cond[test_prev].mean(axis=1)
        assert np.allclose(prob.sum(1),2)
        popularity=np.broadcast_to(pop,prob.shape)
        shuffled=prob[rng.permutation(len(prob))]
        overlap=(test_prev[:,:,None]==test_next[:,None,:]).any(axis=2).sum(axis=1)
        mi,nmi=set_information(test_prev,test_next,e)
        null_mi,null_nmi=set_information(test_prev,test_next[rng.permutation(len(test_next))],e)
        gains=[]
        for _,prev,nxt in by_split['heldout']:
            gains.append(top_recall(cond[prev].mean(1),nxt,2)-top_recall(np.broadcast_to(pop,(len(prev),e)),nxt,2))
        boots=np.mean(rng.choice(gains,size=(1000,len(gains)),replace=True),axis=1)
        row=dict(model=model,phase=phase,batch=b,context=length,domain=domain,site=site,previous_loop=loop,lag=lag,
            context_definition='Base prompt length. Actual full-prefix decode length grows by 1..31.' if phase=='decode' else 'Actual prefill length',
            heldout_prompts=len(gains),calibration_prompts=len(by_split['calibration']),token_pairs=len(test_prev),
            repeated_expert_fraction=float(np.mean(overlap/2)),iou=float(np.mean(overlap/(4-overlap))),
            marginal_next_entropy_bits=entropy(pop/2),conditional_next_entropy_bits=float(np.mean([entropy(x/2) for x in prob])),
            empirical_set_mi_bits=mi,empirical_set_normalized_mi=nmi,shuffled_set_mi_bits=null_mi,
            mi_warning='Finite-sample plug-in estimate. Matched shuffle reports sparse-table bias. Prediction uses separate prompts.',
            cross_entropy_improvement_bits=float(np.mean(np.log2(np.maximum(prob[np.arange(len(prob))[:,None],test_next],1e-9))-
                np.log2(np.maximum(pop[test_next],1e-9)))),
            recall2_gain_ci95_low=float(np.quantile(boots,.025)),recall2_gain_ci95_high=float(np.quantile(boots,.975)),
            probability_definition='P(j belongs to next top-2 set). Rows sum to 2, not 1.',
            evidence='HELDOUT_TRACE_PREDICTION_NOT_EARLY_DISPATCH')
        for k in [2,4,8]:
            row[f'conditional_recall{k}']=float(top_recall(prob,test_next,k))
            row[f'popularity_recall{k}']=float(top_recall(popularity,test_next,k))
            row[f'shuffled_recall{k}']=float(top_recall(shuffled,test_next,k))
        transition_rows.append(row)
    # Preserve complete per-cell inputs for constrained, common-memory placement analysis.
    prefix=model if phase=='prefill' else model+'_decode'
    archive=ROOT/'raw'/f'{prefix}_route_arrays.npz'
    packed={}
    for index,(key,values) in enumerate(groups.items()):
        packed[f'ids_{index}']=np.concatenate([v for _,v in values])
        packed[f'positions_{index}']=np.concatenate(positions[key])
        aggregate.append(dict(index=index,key=list(key),requests=[r for r,_ in values]))
    np.savez_compressed(archive,**packed)
    write(ROOT/'raw'/f'{prefix}_route_arrays_manifest.json',aggregate)
    return pair_rows,transition_rows

def main():
    import sys
    decode='--decode' in sys.argv
    pairs=[];transitions=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        from .native_resume import directory
        p=(directory(model)/'decode_router_corpus.jsonl.gz' if decode else
            ROOT/'raw'/f'{model}_route_corpus_v2/router_corpus.jsonl.gz')
        if not p.exists() or not (p.parent/'complete.json').exists():continue
        c,t=analyze_model(model,p,phase='decode' if decode else 'prefill');pairs.extend(c);transitions.extend(t)
    csv_write(ROOT/('coactivation_decode.csv' if decode else 'coactivation_matrix_summary.csv'),pairs)
    csv_write(ROOT/('transition_decode.csv' if decode else 'transition_affinity.csv'),transitions)

if __name__=='__main__':main()
