"""Small prompt-only ridge probe. No future tokens or post-prefill state are features."""
import statistics
import time
import numpy as np
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_decode_first_depth_aware_dp_v3.simulator import CostModel,simulate_dp
from poc_decode_first_depth_aware_dp_v3.scripts.oracle import greedy,packed

def ridge(x,y,alpha):
    penalty=np.eye(x.shape[1])*alpha;penalty[0,0]=0
    return np.linalg.solve(x.T@x+penalty,x.T@y)

def main():
    refinement=read(ROOT/'artifacts/oracle_refinement_v2.json')
    assert refinement['predictor_allowed'],'No predictor without a positive cost-aware oracle gate.'
    frozen=read(ROOT/'artifacts/discovery_frozen.json')['rows']
    observed={r['id']:read(ROOT/'outputs/discovery128_graph_b1_v1'/(r['id'].replace('/','_')+'.json')) for r in frozen}
    domains=['gsm8k','math500','alpaca','sharegpt']
    started=time.perf_counter()
    features=[]
    for r in frozen:
        text=r['prompt'];n=max(1,len(text))
        features.append([observed[r['id']]['prompt_tokens']/1000,n/4000,text.count('\n')/30,
            sum(ch.isdigit() for ch in text)/n,text.count('?')/10,text.count('\\')/30,
            sum(ch in '{}=+-*/' for ch in text)/n,sum(ord(ch)>127 for ch in text)/n]+
            [int(r['dataset']==d) for d in domains[1:]])
    x=np.asarray(features,float)
    train=np.asarray([r['split']=='calibration' for r in frozen]);test=~train
    mean=x[train].mean(0);scale=x[train].std(0);scale[scale<1e-8]=1
    x=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    targets={'mean_depth':np.asarray([observed[r['id']]['mean_depth'] for r in frozen]),
        'output_length':np.asarray([observed[r['id']]['output_length'] for r in frozen])}
    predictions={};models=[]
    for target,y in targets.items():
        choices=[]
        xt=x[train];yt=y[train]
        for alpha in [.1,1.,10.,100.]:
            errors=[]
            for i in range(len(yt)):
                keep=np.arange(len(yt))!=i
                errors.append((float(xt[i]@ridge(xt[keep],yt[keep],alpha))-yt[i])**2)
            choices.append((statistics.mean(errors),alpha))
        _,alpha=min(choices)
        coef=ridge(xt,yt,alpha)
        pred=x@coef
        pred=np.clip(pred,2,4) if target=='mean_depth' else np.clip(pred,1,256)
        predictions[target]=pred
        constant=float(np.mean(yt))
        models.append(dict(target=target,alpha=alpha,calibration_n=int(train.sum()),heldout_n=int(test.sum()),
            heldout_MAE=float(np.mean(abs(pred[test]-y[test]))),
            constant_MAE=float(np.mean(abs(y[test]-constant))),coefficients=coef.tolist()))
    overhead=(time.perf_counter()-started)*1000
    actual_work=np.asarray([sum(e['depth'] for e in observed[r['id']]['depth_events']) for r in frozen])
    predicted_work=predictions['mean_depth']*np.maximum(0,predictions['output_length']-1)
    derived_work=dict(target='total_recurrent_work',definition='Predicted mean depth times predicted decode-token count.',
        heldout_MAE_loop_tokens=float(np.mean(abs(predicted_work[test]-actual_work[test]))),
        constant_MAE_loop_tokens=float(np.mean(abs(actual_work[test]-np.mean(actual_work[train])))))
    model=CostModel(read(ROOT/'artifacts/cdb_cost_model.json')['parameters'])
    selected=[(i,r) for i,r in enumerate(frozen) if r['split']=='heldout']
    results=[];placements=[]
    old=read(ROOT/'artifacts/oracle_schedule_results.json')
    for c in [8,16,32,64]:
        inputs=[];weights=[];weights_matched_length=[]
        for order,(i,r) in enumerate(selected[:c]):
            raw=observed[r['id']]
            inputs.append(dict(id=r['id'],prompt_tokens=raw['prompt_tokens'],depths=[e['depth'] for e in raw['depth_events']],
                arrival_ms=0.,order=order,cap=256))
            domain_cal=[observed[z['id']]['output_length'] for z in frozen if z['split']=='calibration' and z['dataset']==r['dataset']]
            length_prior=statistics.median(domain_cal)
            weights.append(predictions['mean_depth'][i]*predictions['output_length'][i]+raw['prompt_tokens']*.02)
            weights_matched_length.append(predictions['mean_depth'][i]*length_prior+raw['prompt_tokens']*.02)
        control=min((r for r in old if r['C']==c and r['policy'] in ['round_robin','queue','length_causal','strong_cost']),key=lambda r:r['E2E_P95_ms'])
        for name,w in [('ridge_depth_length',weights),('ridge_depth_matched_length',weights_matched_length)]:
            begin=time.perf_counter();placement=greedy(w);placement_ms=(time.perf_counter()-begin)*1000
            value=simulate_dp(packed(inputs,placement),model)
            reference=next(r for r in refinement['rows'] if r['C']==c)
            gain=control['E2E_P95_ms']-value['E2E_P95_ms']
            oracle=control['E2E_P95_ms']-reference['oracle_E2E_P95_ms']
            results.append(dict(C=c,policy=name,E2E_P95_ms=value['E2E_P95_ms'],
                gain_vs_generic_percent=100*gain/control['E2E_P95_ms'],recovered_oracle_fraction=gain/oracle if oracle else None,
                placement_ms=placement_ms,label='COST_MODEL_HELDOUT_CAUSAL_PLACEMENT_NOT_MEASURED_RUNTIME_GAIN'))
            placements.append(dict(C=c,policy=name,ranks=placement))
    write(ROOT/'artifacts/causal_predictor_v2.json',dict(models=models,scheduling=results,placements=placements,
        derived_total_work=derived_work,analysis_version='v1 retained. v2 adds derived work forecast error without changing placements.',
        training_and_feature_total_ms=overhead,training='32 calibration requests only. Alpha uses calibration leave-one-out.',
        features='Known prompt length and text statistics plus declared dataset metadata. No future token, exit trace, or post-prefill hidden state.',
        depth_and_length_targets_separate=True,no_RL=True,quality_changes=False,
        limitation='Small exploratory train/held-out split. Frozen-output cost simulation does not prove live quality or scheduling gain.'))
    print('CAUSAL_PREDICTOR',models,results,flush=True)

if __name__=='__main__':main()
