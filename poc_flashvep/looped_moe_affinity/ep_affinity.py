"""Evaluate native-calibrated predictors on actual EP router observations."""
from collections import defaultdict
import json
import numpy as np
from .common import ROOT
from .affinity import fit_transition,top_recall,entropy
from .aggregate_quality import csv_write

def main():
    output=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        archive=np.load(ROOT/'raw'/f'{model}_route_arrays.npz')
        manifest=json.loads((ROOT/'raw'/f'{model}_route_arrays_manifest.json').read_text())
        lookup={tuple(row['key']):row['index'] for row in manifest}
        models={}
        for ep in [4,8]:
            path=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1/routes.jsonl'
            if not path.exists():continue
            values=[json.loads(line) for line in path.open()]
            previous={}
            for row in values:
                key=(row['batch'],row['context'],row['phase'],row['request_id'],row['site'],row['loop'])
                previous[key]=row
            for row in values:
                b=row['batch'];phase=row['phase'];site=row['site'];loop=row['loop']
                base=row['context'] if phase=='prefill' else row['context']-row['step']
                e=25 if model.startswith('ifm') else (64 if row['recurrent'] else 8)
                ids=np.asarray(row['top2'],dtype=int)
                for role in (['sampled_prefill'] if phase=='prefill' else ['prefix_recompute','new_token_frontier']):
                    select=np.ones(len(ids),dtype=bool) if phase=='prefill' else np.asarray(row['roles'])==role
                    current=ids[select]
                    counts=np.bincount(current.ravel(),minlength=e)
                    owners=current%ep
                    out=dict(model=model,ep=ep,batch=b,prefix_tokens=base,phase=phase,role=role,domain=row['domain'],
                        site=site,loop=loop,request_id=row['request_id'],token_invocations=len(current),
                        unique_destinations_mean=float(1+np.mean(owners[:,0]!=owners[:,1])),
                        expert_choice_entropy_bits=entropy(counts/max(counts.sum(),1)),
                        evidence='ACTUAL_EP_HELDOUT_ROUTE_OBSERVATION',
                        uncertainty='One performance cohort per cell. Native corpus supplies the larger prompt-held-out study.')
                    if row['recurrent'] and loop>0:
                        k=(b,base,row['domain'],site,loop-1,'calibration')
                        if k not in models:
                            train=archive[f'ids_{lookup[k]}'];nxt=archive[f'ids_{lookup[(*k[:4],loop,"calibration")]}']
                            models[k]=fit_transition(train,nxt,e)
                        conditional,pop=models[k]
                        prev=previous[(b,row['context'],phase,row['request_id'],site,loop-1)]
                        assert prev['positions']==row['positions']
                        prior=np.asarray(prev['top2'],dtype=int)[select]
                        prob=conditional[prior].mean(1)
                        for n in [2,4,8]:
                            out[f'conditional_recall{n}']=float(top_recall(prob,current,n))
                            out[f'popularity_recall{n}']=float(top_recall(np.broadcast_to(pop,prob.shape),current,n))
                        out['calibration_transfer']='Prefill calibration transfers to actual decode routes' if phase=='decode' else 'Matched prefill calibration'
                    output.append(out)
    csv_write(ROOT/'ep_affinity_metrics.csv',output)

if __name__=='__main__':main()
