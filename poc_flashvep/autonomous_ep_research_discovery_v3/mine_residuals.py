#!/usr/bin/env python3
"""Residual-cluster miner for fresh logical stage rows (descriptive only)."""
import csv, math, statistics
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parents[2]
ROOT=HERE/'poc_flashvep/deepep_revalidation/results/autonomous_ep_research_discovery_v3_20260906_172149/discovery_atlas'

def q(x,p):
    x=sorted(x); i=(len(x)-1)*p; lo=math.floor(i); hi=math.ceil(i); return x[lo] if lo==hi else x[lo]+(x[hi]-x[lo])*(i-lo)
def main():
    rows=list(csv.DictReader((ROOT/'logical_invocations_all.csv').open()))
    for r in rows:
        for k in ('M','cuda_ms','active_experts','rank_cv','expert_cv','fanout_mean','fanout_f4'):
            r[k]=float(r[k] or 0)
    groups=defaultdict(list)
    for r in rows: groups[(r['trace'],r['phase'],int(r['M']))].append(r['cuda_ms'])
    out=[]
    for r in rows:
        base=statistics.median(groups[(r['trace'],r['phase'],int(r['M']))]); r['baseline_ms']=base; r['residual_ms']=r['cuda_ms']-base; r['normalized']=r['cuda_ms']/max(base,1e-9)
        out.append(r)
    out.sort(key=lambda r:r['residual_ms'],reverse=True)
    with (ROOT/'residual_top.csv').open('w',newline='') as f:
        fields=['trace','phase','M','cuda_ms','baseline_ms','residual_ms','normalized','active_experts','rank_cv','expert_cv','fanout_mean','fanout_f4']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k,0) for k in fields} for r in out[:max(1000,len(out)//100)])
    n=max(1,len(out)//100); top=out[:n]
    def avg(k,z): return statistics.mean(r[k] for r in z)
    summary={'n':len(out),'top_1pct_n':len(top),'top_1pct_threshold_ms':top[-1]['residual_ms'] if top else None,'all_residual_p50_ms':q([r['residual_ms'] for r in out],.5),'top_residual_p50_ms':q([r['residual_ms'] for r in top],.5),'top_vs_all':{k:avg(k,top)-avg(k,out) for k in ('M','active_experts','rank_cv','expert_cv','fanout_mean','fanout_f4')}}
    (ROOT/'residual_summary.json').write_text(__import__('json').dumps(summary,indent=2))
    print(__import__('json').dumps(summary,indent=2))
if __name__=='__main__': main()
