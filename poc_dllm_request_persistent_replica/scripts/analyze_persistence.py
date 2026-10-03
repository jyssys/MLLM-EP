"""Cross-block locality, calibration-only global and matched other-request controls."""
import json
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from poc_dllm_request_persistent_replica.contract import *
from poc_dllm_request_persistent_replica.data import load_profile
from poc_dllm_block_local_replica.metrics import bootstrap_mean


def normalize(x):
    sums=x.sum(axis=-1,keepdims=True)
    return np.divide(x,sums,out=np.zeros_like(x,dtype=float),where=sums>0)


def jsd(p,q):
    midpoint=(p+q)/2
    # Direct divergence avoids sqrt-of-roundoff in scipy's distance helper.
    return np.maximum(0,(rel_entr(p,midpoint).sum(axis=1)+rel_entr(q,midpoint).sum(axis=1))/(2*np.log(2)))


def compare(prior,truth):
    p=normalize(prior);q=normalize(truth);valid=(p.sum(axis=2)>0)&(q.sum(axis=2)>0)
    if not valid.any():return dict(cosine=None,jsd=None,rank_cosine=None,rank_jsd=None)
    pv=p[valid];qv=q[valid]
    cosine=(pv*qv).sum(axis=1)/(np.linalg.norm(pv,axis=1)*np.linalg.norm(qv,axis=1))
    divergence=jsd(pv,qv)
    pr=pv.reshape(-1,4,64).sum(axis=2);qr=qv.reshape(-1,4,64).sum(axis=2)
    rankcos=(pr*qr).sum(axis=1)/(np.linalg.norm(pr,axis=1)*np.linalg.norm(qr,axis=1))
    result=dict(cosine=float(cosine.mean()),jsd=float(divergence.mean()),rank_cosine=float(rankcos.mean()),
                rank_jsd=float(jsd(pr,qr).mean()))
    for m in (4,8,16,32):
        top=np.argsort(-p,axis=2,kind='stable')[:,:,:m]
        selected=np.zeros_like(p,dtype=bool);np.put_along_axis(selected,top,True,axis=2)
        selected &= (p>0)
        result[f'coverage_at_{m}']=float(truth[selected].sum()/truth.sum()) if truth.sum() else 0.
        true_top=np.argsort(-q,axis=2,kind='stable')[:,:,:m]
        result[f'top_overlap_{m}']=float(np.mean([len(set(a)&set(b))/m for a,b in zip(top[valid],true_top[valid])]))
    return result


def main():
    out=ROOT/'results/persistence_v2'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    rows=[]
    for lane in ('short','long'):
        infos=json.loads((ROOT/'results'/(lane+'_manifest.json')).read_text())['requests']
        profiles={i['request']:load_profile(i) for i in infos}
        counts={k:[p.subset(b,b+1).frequency() for b in range(p.nblocks)] for k,p in profiles.items()}
        cal=[i for i in infos if i['split']=='calibration']
        global_prior=np.mean([normalize(profiles[i['request']].frequency()) for i in cal],axis=0)
        for info in infos:
            name=info['request'];blocks=counts[name]
            # Fixed other-request control: same native block size, source layout,
            # generation budget and domain; no held-out expert data selects it.
            others=sorted([i for i in cal if i['domain']==info['domain'] and i['request']!=name],key=lambda i:i['request'])
            other=counts[others[0]['request']]
            profilesets=[('first_block',0,blocks[0]),('first2_blocks',1,sum(blocks[:2]))]
            for target,truth in enumerate(blocks):
                destinations=profiles[name].subset(target,target+1).groups()
                priors=[('global',-1,global_prior),('other_request',-1,other[min(target,len(other)-1)])]
                priors += [x for x in profilesets if target>x[1]]
                for distance in (1,2,4,8):
                    if target>=distance:priors.append((f'distance_{distance}',target-distance,blocks[target-distance]))
                for policy,origin,prior in priors:
                    metrics=compare(prior,truth)
                    # Whole destination closure, not independent expert hit rate.
                    for m in (4,8,16,32):
                        closed=total=0
                        for s in range(4):
                            for l in range(1,20):
                                candidates=[e for e in np.argsort(-prior[l-1,s],kind='stable') if e//64!=s and prior[l-1,s,e]>0][:m]
                                mask=sum(1<<int(e) for e in candidates)
                                for (_,needed),weight in destinations[s][l].items():
                                    total+=weight
                                    if needed&~mask==0:closed+=weight
                        metrics[f'destination_closure_at_{m}']=closed/total if total else 0.
                    rows.append(dict(lane=lane,request=name,domain=info['domain'],split=info['split'],
                        target_block=target,origin_block=origin,distance=target-origin if origin>=0 else -1,
                        policy=policy,**metrics))
            print('PERSISTENCE',lane,name,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'block_locality.csv',index=False)
    contrasts=[]
    for (lane,split,policy,distance),r in df[df.policy.isin(['first_block','first2_blocks'])].groupby(['lane','split','policy','distance']):
        # Pair at identical target block, then bootstrap requests, not blocks.
        for control in ('global','other_request'):
            c=df[(df.lane==lane)&(df.split==split)&(df.policy==control)]
            pair=r.merge(c,on=['request','target_block'],suffixes=('_p','_c'),validate='one_to_one')
            for metric in ('cosine','coverage_at_16','destination_closure_at_16','rank_cosine'):
                values=(pair[metric+'_p']-pair[metric+'_c']).groupby(pair.request).mean()
                contrasts.append(dict(lane=lane,split=split,policy=policy,control=control,distance=int(distance),metric=metric,
                                      **bootstrap_mean(values.tolist())))
    new_json(out/'paired_ci.json',dict(unit='request',repeats=2000,contrasts=contrasts,
        matched_other_request='calibration-only same domain/block size/native generation setting; fixed lexicographic other task',
        closure='top-M remote experts PER LAYER/SOURCE, not a whole-rank budget; diagnostic only'))


if __name__=='__main__':main()
