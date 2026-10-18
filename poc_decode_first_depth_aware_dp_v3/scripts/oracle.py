"""Clairvoyant admission placement on measured-cost CDB traces. Not a live scheduler."""
import argparse
import collections
import itertools
import random
import statistics
import numpy as np
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_decode_first_depth_aware_dp_v3.simulator import CostModel,simulate_dp

def canonical_partitions(n,k=4):
    def recurse(values,maximum):
        if len(values)==n:
            yield values;return
        for label in range(min(k-1,maximum+1)+1):
            yield from recurse(values+[label],max(label,maximum))
    yield from recurse([0],0)

def packed(records,placement):
    result=[[] for _ in range(4)]
    for r,rank in zip(records,placement):result[rank].append(r)
    return result

def greedy(weights):
    loads=[0.]*4
    result=[]
    for w in weights:
        rank=min(range(4),key=lambda i:(loads[i],i))
        result.append(rank);loads[rank]+=w
    return result

def optimize(records,model,starts,seed):
    cache={}
    def evaluate(placement):
        key=tuple(placement)
        if key not in cache:cache[key]=simulate_dp(packed(records,placement),model)
        return cache[key]
    if len(records)<=4:
        candidates=list(canonical_partitions(len(records)))
        best=min(candidates,key=lambda p:evaluate(p)['E2E_P95_ms'])
        return best,evaluate(best),'SOLVED_OPTIMAL_SYMMETRIC_SMALL_BURST',len(cache)
    rng=random.Random(seed)
    best=min(starts,key=lambda p:evaluate(p)['E2E_P95_ms'])
    score=evaluate(best)['E2E_P95_ms']
    # The budget is fixed before results. This is a best-known search, not an exact optimum.
    for _ in range(2):
        candidates=[]
        for _ in range(20):
            trial=list(best)
            a=rng.randrange(len(trial))
            if rng.random()<.5:
                b=rng.randrange(len(trial));trial[a],trial[b]=trial[b],trial[a]
            else:trial[a]=rng.randrange(4)
            candidates.append(trial)
        for p in candidates:
            value=evaluate(p)['E2E_P95_ms']
            if value<score:best=list(p);score=value
    return best,evaluate(best),'BEST_KNOWN_CLAIRVOYANT_HEURISTIC_NOT_OPTIMAL',len(cache)

def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-unvalidated-sensitivity',action='store_true');args=p.parse_args()
    fit=read(ROOT/'artifacts/cdb_cost_model.json')
    if fit['status']!='VALIDATED' and not args.allow_unvalidated_sensitivity:
        write(ROOT/'artifacts/oracle_status.json',dict(status='SIMULATOR_UNVALIDATED',
            realistic_oracle_claims=False,reason='Live validation did not pass. No predictor training.'))
        return
    model=CostModel(fit['parameters'])
    rows=read(ROOT/'artifacts/discovery_frozen.json')['rows']
    source=ROOT/'outputs/discovery128_graph_b1_v1'
    observed={r['id']:read(source/(r['id'].replace('/','_')+'.json')) for r in rows}
    calibration={}
    for domain in ['gsm8k','math500','alpaca','sharegpt']:
        cal=[observed[r['id']] for r in rows if r['dataset']==domain and r['split']=='calibration']
        calibration[domain]=dict(mean_depth=statistics.mean(v['mean_depth'] for v in cal),
            expected_length=statistics.median(v['output_length'] for v in cal),
            example_depths=[e['depth'] for e in cal[0]['depth_events']])
    heldout=[r for r in rows if r['split']=='heldout']
    results=[]
    schedules=[]
    comparisons=[]
    for c in [1,2,4,8,16,32,64]:
        inputs=[]
        proxies=[]
        weights=collections.defaultdict(list)
        for order,row in enumerate(heldout[:c]):
            raw=observed[row['id']]
            prior=calibration[row['dataset']]
            d=[e['depth'] for e in raw['depth_events']]
            record=dict(id=row['id'],prompt_tokens=raw['prompt_tokens'],depths=d,arrival_ms=0.,order=order,cap=256)
            inputs.append(record)
            n=len(d)
            expected=prior['example_depths']
            assert expected
            # Future true output length is allowed here. Future true depth is not.
            proxy=[expected[(j+order)%len(expected)] for j in range(n)]
            proxies.append(dict(record,depths=proxy))
            prompt=raw['prompt_tokens']*.02
            weights['length_causal'].append(prior['expected_length']+prompt)
            weights['strong_cost'].append(prior['expected_length']*prior['mean_depth']+prompt)
            weights['true_length'].append((n+1)*prior['mean_depth']+prompt)
            weights['true_depth_only'].append(prior['expected_length']*raw['mean_depth']+prompt)
            weights['combined'].append(sum(d)+prompt)
        places={'round_robin':[i%4 for i in range(c)],'queue':[i%4 for i in range(c)]}
        for name,w in weights.items():places[name]=greedy(w)
        homogeneous=[0]*c
        for index,order in enumerate(sorted(range(c),key=lambda i:statistics.mean(inputs[i]['depths']) if inputs[i]['depths'] else 4.)):
            homogeneous[order]=min(3,index//max(1,(c+3)//4))
        places['depth_homogeneous']=homogeneous
        length_best,length_proxy,kind,search_n=optimize(proxies,model,
            [places['true_length'],places['strong_cost'],places['round_robin']],20261018+c)
        places['true_length_best_known']=length_best
        combined_best,combined_result,combined_kind,combined_n=optimize(inputs,model,
            [length_best,places['combined'],places['strong_cost'],places['round_robin']],20261018+c)
        places['combined_best_known']=combined_best
        metrics={name:simulate_dp(packed(inputs,place),model) for name,place in places.items()}
        strongest=min(['round_robin','queue','length_causal','strong_cost'],key=lambda name:metrics[name]['E2E_P95_ms'])
        generic=metrics[strongest]['E2E_P95_ms']
        length=metrics['true_length_best_known']['E2E_P95_ms']
        both=metrics['combined_best_known']['E2E_P95_ms']
        paired={}
        for name in ['true_length_best_known','combined_best_known']:
            paired[name]={o['id']:o['finish_ms']-o['arrival_ms'] for o in metrics[name]['outputs']}
        ids=sorted(paired['true_length_best_known'])
        a=np.asarray([paired['true_length_best_known'][i] for i in ids])
        b=np.asarray([paired['combined_best_known'][i] for i in ids])
        draws=np.random.default_rng(20261018).integers(0,c,size=(2000,c))
        gains=100*(1-np.quantile(b[draws],.95,axis=1)/np.quantile(a[draws],.95,axis=1))
        comparisons.append(dict(C=c,depth_increment_beyond_true_length_percent=100*(1-both/length),
            CI95_depth_increment_percent=np.quantile(gains,[.025,.975]).tolist(),
            combined_gain_beyond_generic_percent=100*(1-both/generic),
            bootstrap_unit='Paired request ID, conditional on these fixed placements. No reoptimization inside bootstrap.'))
        for name,value in metrics.items():
            summary={k:v for k,v in value.items() if k not in ['ranks','outputs']}
            results.append(dict(C=c,arrival='burst',policy=name,**summary,
                label='COST_AWARE_ORACLE_ESTIMATE' if fit['status']=='VALIDATED' else 'UNVALIDATED_MODEL_SENSITIVITY',
                strongest_generic=strongest,
                gain_vs_strongest_generic_percent=100*(1-value['E2E_P95_ms']/generic),
                cost_validation=fit['status']))
            schedules.append(dict(C=c,policy=name,placements=[dict(id=r['id'],rank=rank) for r,rank in zip(inputs,places[name])]))
        print('ORACLE',c,'generic',generic,'length',length,'combined',both,
            'depth_increment',100*(1-both/length),flush=True)
        write(ROOT/'artifacts/oracle_schedule_results.json',results)
        write(ROOT/'artifacts/oracle_placements.json',schedules)
    import csv
    with (ROOT/'artifacts/oracle_schedule_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    write(ROOT/'artifacts/oracle_status.json',dict(cost_validation=fit['status'],comparisons=comparisons,
        predictor_allowed=fit['status']=='VALIDATED' and any(r['depth_increment_beyond_true_length_percent']>=10 and
            r['CI95_depth_increment_percent'][0]>0 for r in comparisons if r['C']>=8),
        optimality='Symmetric C<=4 solved exactly. Larger bursts use a fixed-budget local search. No global optimality claim.',
        future_information='True future depth and EOS length only appear in named oracle policies and execution replay.',
        limitations=['Burst-only primary oracle. Other arrival patterns require separate validation.',
                    'Natural-EOS generation is bounded at 256. This is not full-response quality evidence.',
                    'DepthHomogeneous uses true future mean depth. Joint-microbatch optimization is NOT_RUN unless this bound is promising.']))

if __name__=='__main__':main()
