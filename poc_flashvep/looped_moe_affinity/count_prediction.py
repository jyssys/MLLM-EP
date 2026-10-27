"""Test full split matrices. A wrong sampled route alone does not disprove exact counts."""
import json
import numpy as np
from .common import ROOT
from .affinity import fit_transition
from .aggregate_quality import csv_write
from .native_resume import directory

def rounded_counts(expected,total):
    out=np.floor(expected).astype(np.int64)
    remainder=int(total-out.sum())
    assert 0<=remainder<=len(out)
    order=np.argsort(-(expected-out),kind='stable')
    out[order[:remainder]]+=1
    assert out.sum()==total
    return out

def main():
    result=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        base=directory(model)
        if not (base/'complete.json').exists():continue
        calibration=np.load(ROOT/'raw'/f'{model}_route_arrays.npz')
        manifest=json.loads((ROOT/'raw'/f'{model}_route_arrays_manifest.json').read_text())
        lookup={tuple(r['key']):r['index'] for r in manifest}
        model_meta=json.loads((base/'metadata.rank0.json').read_text())
        recurrent={r['site'] for r in model_meta['physical_moe_sites'] if r['recurrent']}
        for b in [1,2,4,8,16]:
            for length in [128,512,2048]:
                p=base/f'full_routes_b{b}_p{length}_heldout.npz'
                if not p.exists():continue
                full=np.load(p);info=json.loads(p.with_suffix('.json').read_text())
                ep_counts={}
                for ep in [4,8]:
                    ranks=[]
                    for rank in range(ep):
                        trace=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1'/f'ep_trace_b{b}_p{length}.rank{rank}.json'
                        ranks.append({(r['site'],r['loop']):r['source_counts'] for r in json.loads(trace.read_text())})
                    ep_counts[ep]=ranks
                for key in full.files:
                    site,loop=key.rsplit('.r',1);loop=int(loop)
                    if site not in recurrent or loop==0:continue
                    previous=full[f'{site}.r{loop-1}'].astype(int);actual=full[key].astype(int)
                    e=25 if model.startswith('ifm') else 64
                    predicted=[];probabilities=[];popularities=[]
                    for i,domain in enumerate(info['domains']):
                        k=(b,length,domain,site,loop-1,'calibration')
                        x=calibration[f'ids_{lookup[k]}'];y=calibration[f'ids_{lookup[(*k[:4],loop,"calibration")]}']
                        conditional,pop=fit_transition(x,y,e)
                        prob=conditional[previous[i]].mean(1)
                        predicted.append(np.argsort(-prob,axis=1,kind='stable')[:,:2])
                        probabilities.append(prob);popularities.append(np.broadcast_to(pop,prob.shape))
                    predicted=np.stack(predicted);probabilities=np.stack(probabilities);popularities=np.stack(popularities)
                    for ep in [4,8]:
                        sources=np.arange(b*length)%ep
                        ids=actual.reshape(-1,2);pred=predicted.reshape(-1,2)
                        exact=np.bincount((sources[:,None]*e+ids).ravel(),minlength=ep*e).reshape(ep,e)
                        discrete=np.bincount((sources[:,None]*e+pred).ravel(),minlength=ep*e).reshape(ep,e)
                        estimates=[];pop_estimates=[]
                        for rank in range(ep):
                            select=sources==rank;n=int(select.sum())
                            estimates.append(rounded_counts(probabilities.reshape(-1,e)[select].sum(0),n*2))
                            pop_estimates.append(rounded_counts(popularities.reshape(-1,e)[select].sum(0),n*2))
                        estimates=np.stack(estimates);pop_estimates=np.stack(pop_estimates)
                        measured=np.asarray([rank[(site,loop)] for rank in ep_counts[ep]])
                        result.append(dict(model=model,ep=ep,batch=b,prefix_tokens=length,site=site,previous_loop=loop-1,
                            requests=b,full_token_rows=b*length,
                            top2_route_exact_fraction=float(np.all(np.sort(predicted,axis=-1)==np.sort(actual,axis=-1),axis=-1).mean()),
                            native_vs_live_ep_full_count_matrix_exact=bool(np.array_equal(exact,measured)),
                            discrete_split_matrix_exact=bool(np.array_equal(exact,discrete)),
                            conditional_expected_split_matrix_exact=bool(np.array_equal(exact,estimates)),
                            popularity_expected_split_matrix_exact=bool(np.array_equal(exact,pop_estimates)),
                            conditional_expected_count_l1_fraction=float(np.abs(exact-estimates).sum()/(2*b*length)),
                            popularity_expected_count_l1_fraction=float(np.abs(exact-pop_estimates).sum()/(2*b*length)),
                            evidence='FULL_NATIVE_ROUTER_DERIVED_METADATA_PREDICTION_NOT_LIVE_EP_SPEEDUP',
                            warning='Exact counts do not provide the next hidden activation early. No speculative collective was submitted.'))
    csv_write(ROOT/'full_count_prediction.csv',result)

if __name__=='__main__':main()
