"""Common-memory fixed placement controls. These are trace models, not GPU results."""
from collections import defaultdict
import json
import numpy as np
from .common import ROOT,write
from .aggregate_quality import csv_write

def cost(owner,co,marginal,penalty=.25):
    cut=float(co[owner[:,None]!=owner[None,:]].sum())/max(co.sum(),1)
    loads=np.bincount(owner,weights=marginal,minlength=owner.max()+1)
    return cut+penalty*loads.max()/max(loads.mean(),1e-12)

def optimize(initial,co,marginal,seed=20261026):
    """Swap expert pairs. Keep each rank's original expert count and weight capacity."""
    owner=initial.copy();best=cost(owner,co,marginal)
    rng=np.random.default_rng(seed)
    for _ in range(12):
        changed=False
        for i,j in rng.permutation([(i,j) for i in range(len(owner)) for j in range(i)]):
            if owner[i]==owner[j]:continue
            owner[i],owner[j]=owner[j],owner[i]
            value=cost(owner,co,marginal)
            if value<best-1e-9:best=value;changed=True
            else:owner[i],owner[j]=owner[j],owner[i]
        if not changed:break
    return owner

def stats(owner,ids,positions,ep):
    owners=owner[ids]
    counts=np.bincount(owners.ravel(),minlength=ep)
    # The original dispatcher sends one row for each selected expert edge.
    source=positions%ep
    remote=(owners!=source[:,None]).sum()
    # Generic current-top1 source control chooses an already replicated row copy.
    generic_remote=(owners[:,1]!=owners[:,0]).sum()
    return dict(token_invocations=len(ids),original_remote_edges=int(remote),
        generic_current_owner_remote_edges=int(generic_remote),
        unique_destinations_mean=float(1+np.mean(owners[:,0]!=owners[:,1])),
        rank_edge_loads=counts.tolist(),rank_load_cv=float(counts.std()/max(counts.mean(),1e-12)),
        rank_load_max_mean=float(counts.max()/max(counts.mean(),1e-12)))

def main():
    result=[];maps=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        manifest=ROOT/'raw'/f'{model}_route_arrays_manifest.json'
        if not manifest.exists():continue
        groups=json.loads(manifest.read_text());arrays=np.load(ROOT/'raw'/f'{model}_route_arrays.npz')
        bysite=defaultdict(list)
        for row in groups:bysite[row['key'][3]].append(row)
        for ep in [4,8]:
            for site,items in bysite.items():
                calibration=[r for r in items if r['key'][-1]=='calibration']
                train=np.concatenate([arrays[f'ids_{r["index"]}'] for r in calibration])
                e=25 if model.startswith('ifm') else (64 if 'stack' in site else 8)
                co=np.zeros((e,e));np.add.at(co,(train[:,0],train[:,1]),1);co=co+co.T
                marg=np.bincount(train.ravel(),minlength=e)
                static=np.arange(e)%ep
                generic=optimize(static,co,marg)
                popularity=optimize(static,np.zeros_like(co),marg)
                # Recurrent graph connects consecutive pass selections on the same token.
                transitions=np.zeros_like(co)
                keyed={tuple(r['key']):r for r in calibration}
                for row in calibration:
                    key=list(row['key']);key[4]+=1;nxt=keyed.get(tuple(key))
                    if nxt is None:continue
                    x=arrays[f'ids_{row["index"]}'];y=arrays[f'ids_{nxt["index"]}']
                    assert row['requests']==nxt['requests'] and len(x)==len(y)
                    for i in [0,1]:
                        for j in [0,1]:np.add.at(transitions,(x[:,i],y[:,j]),1)
                tscale=co.sum()/max(transitions.sum(),1)
                recurrent=optimize(generic,co+.25*tscale*(transitions+transitions.T)/2,marg)
                for name,owner in [('static',static),('popularity_balanced',popularity),('generic_coactivation',generic),('recurrent_transition_graph',recurrent)]:
                    assert np.array_equal(np.bincount(owner,minlength=ep),np.bincount(static,minlength=ep))
                    expert_bytes=22413312 if model.startswith('ifm') else 9437184
                    maps.append(dict(model=model,ep=ep,site=site,method=name,owner=owner.tolist(),
                        rank_expert_counts=np.bincount(owner,minlength=ep).tolist(),extra_replica_bytes=0,
                        changed_expert_initialization_bytes=int(np.count_nonzero(owner!=static)*expert_bytes),
                        online_weight_migration_bytes=0,calibration_only=True,
                        note='Offline resident layout. No per-loop remapping. Setup byte count is structural, not a timed transfer.'))
                    for row in items:
                        b,length,domain,_,loop,split=row['key']
                        if split!='heldout':continue
                        ids=arrays[f'ids_{row["index"]}'];pos=arrays[f'positions_{row["index"]}']
                        value=stats(owner,ids,pos,ep)
                        result.append(dict(model=model,ep=ep,batch=b,context=length,domain=domain,site=site,loop=loop,
                            method=name,**value,evidence='HELDOUT_TRACE_FIXED_PLACEMENT_NOT_E2E_SPEEDUP',
                            transition_cost='No direct inter-loop owner edge in the replicated-dense EP bridge.'))
    csv_write(ROOT/'placement_structural.csv',result)
    write(ROOT/'runs/fixed_placement_maps.json',maps)

if __name__=='__main__':main()
