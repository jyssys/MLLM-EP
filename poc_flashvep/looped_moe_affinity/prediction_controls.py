"""Add joint-set and exact-preparation controls from recorded, prompt-held-out routes."""
from collections import Counter,defaultdict
import csv
import json
import numpy as np
from .common import ROOT
from .affinity import fit_transition
from .aggregate_quality import csv_write

def main():
    path=ROOT/'transition_affinity.csv'
    result=list(csv.DictReader(path.open()))
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        archive=np.load(ROOT/'raw'/f'{model}_route_arrays.npz')
        metadata=json.loads((ROOT/'raw'/f'{model}_route_arrays_manifest.json').read_text())
        lookup={tuple(r['key']):r for r in metadata}
        cache={}
        def array(key):
            index=lookup[key]['index']
            if index not in cache:cache[index]=archive[f'ids_{index}'].astype(int)
            return cache[index]
        for row in result:
            if row['model']!=model:continue
            b=int(row['batch']);length=int(row['context']);site=row['site'];loop=int(row['previous_loop']);lag=int(row['lag'])
            key=(b,length,row['domain'],site,loop,'calibration')
            x=array(key);y=array((*key[:4],loop+lag,'calibration'))
            test_key=(*key[:5],'heldout')
            tx=array(test_key);ty=array((*key[:4],loop+lag,'heldout'))
            e=25 if model.startswith('ifm') else 64
            conditional,pop=fit_transition(x,y,e)
            scores=conditional[tx].mean(1)
            predicted=np.argsort(-scores,axis=1,kind='stable')[:,:2]
            truth=np.sort(ty,axis=1)
            exact=np.all(np.sort(predicted,axis=1)==truth,axis=1)
            sorted_y=np.sort(y,axis=1);pair_mode=Counter(map(tuple,sorted_y)).most_common(1)[0][0]
            pattern=np.broadcast_to(pair_mode,truth.shape)
            pattern_recall=np.mean((pattern[:,:,None]==truth[:,None,:]).any(1).sum(1)/2)
            next_by_previous=defaultdict(Counter)
            for prev,nxt in zip(np.sort(x,axis=1),sorted_y):next_by_previous[tuple(prev)][tuple(nxt)]+=1
            joint=np.asarray([next_by_previous[tuple(prev)].most_common(1)[0][0] if tuple(prev) in next_by_previous else pair_mode
                for prev in np.sort(tx,axis=1)],dtype=int)
            joint_exact=np.all(joint==truth,axis=1)
            n_requests=len(lookup[test_key]['requests']);assert len(exact)%n_requests==0
            row.update(conditional_top2_set_exact=float(exact.mean()),pattern_top2_set_exact=float(np.mean(np.all(pattern==truth,axis=1))),
                pattern_recall2=float(pattern_recall),joint_set_conditional_exact=float(joint_exact.mean()),
                joint_set_conditional_recall2=float(np.mean((joint[:,:,None]==truth[:,None,:]).any(1).sum(1)/2)),
                all_sampled_rows_exact_prompt_fraction=float(exact.reshape(n_requests,-1).all(1).mean()),
                exact_preparation_warning='A wrong sampled row rejects exact per-token routing. It does not disprove an exact full count histogram.')
    csv_write(path,result)

if __name__=='__main__':main()
