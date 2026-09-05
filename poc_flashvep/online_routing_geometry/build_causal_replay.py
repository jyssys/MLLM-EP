"""Build histogram-preserving low/high fanout route variants.

Only swaps of expert IDs between token rows are performed.  Therefore M,
top-k, every per-expert count, active experts, and per-rank assignment totals
are exact invariants.  The output is a diagnostic route-transfer case, not a
model-routing intervention.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def fan(r): return np.asarray(np.unique(r.reshape(r.shape[0], -1)//32, axis=1).shape[1] if False else [len(np.unique(x//32)) for x in r])

def improve(routes, high, steps=60000, seed=7):
    rng=np.random.default_rng(seed); r=routes.copy(); m,k=r.shape
    def score(row): return len(np.unique(row//32))
    rowscore=np.asarray([score(x) for x in r], dtype=float)
    for _ in range(steps):
        i,j=rng.integers(0,m,2)
        if i==j: continue
        a,b=rng.integers(0,k,2)
        if a==b: continue
        va,vb=int(r[i,a]),int(r[j,b])
        if va==vb or va in r[j] or vb in r[i]: continue
        ni=r[i].copy(); nj=r[j].copy(); ni[a]=vb; nj[b]=va
        si,sj=score(ni),score(nj)
        delta=(si+sj)-(rowscore[i]+rowscore[j])
        if (high and delta>0) or ((not high) and delta<0):
            r[i,a],r[j,b]=vb,va; rowscore[i],rowscore[j]=si,sj
    return r

def add_case(cases, base, name, routes):
    e=np.bincount(routes.reshape(-1),minlength=128); rr=np.bincount((routes//32).reshape(-1),minlength=4)
    f=np.asarray([len(np.unique(x//32)) for x in routes])
    x=dict(base); x.update({"case_id":name,"routes":routes.tolist(),"fanout_ranks_mean":float(f.mean()),"fanout_ranks_median":float(np.median(f)),"fanout_histogram":{str(int(q)):int((f==q).sum()) for q in (1,2,3,4)},"expert_counts":e.astype(int).tolist(),"active_experts":int((e>0).sum()),"rank_assignments":rr.astype(int).tolist(),"route_sha256":hashlib.sha256(routes.tobytes()).hexdigest()}); cases.append(x)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--M',type=int,default=512); ap.add_argument('--layer',type=int,default=24); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
 files=sorted(args.source.glob('routing.dp0.*.npz')); chosen=None; sid=None
 for p in files:
  z=np.load(p); a=z['routed_experts']
  if a.shape[0]>=args.M and a.ndim==3: chosen=a[:args.M,args.layer,:].astype(np.int64); sid=p.stem; break
 if chosen is None: raise RuntimeError('no real route source')
 base={"request_id":sid,"category":"real_qwen3vl","modality":"real_route_transfer","layer":args.layer,"M":args.M,"token_count":args.M,"total_assignments":int(chosen.size)}; cases=[]
 add_case(cases,base,'real_original',chosen)
 add_case(cases,base,'real_low_fanout',improve(chosen,False))
 add_case(cases,base,'real_high_fanout',improve(chosen,True))
 (args.output/'cases.json').write_text(json.dumps(cases,separators=(',',':'))+'\n')
 (args.output/'manifest.json').write_text(json.dumps({"kind":"histogram_preserving_causal_replay","source":str(chosen),"layer":args.layer,"M":args.M,"invariants":["M","top_k","per_expert_histogram","active_experts","rank_load"],"note":"diagnostic route transfer; not online model routing"},indent=2)+'\n')
 print(json.dumps([{k:c[k] for k in ('case_id','fanout_ranks_mean','fanout_histogram','active_experts','rank_assignments')} for c in cases],indent=2))
if __name__=='__main__': main()
