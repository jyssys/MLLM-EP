#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[2]
OUT=HERE/'poc_flashvep/deepep_revalidation/results/autonomous_ep_research_discovery_v3_20260906_172149/discovery_atlas'

def main():
    rows=list(csv.DictReader((OUT/'phase_summary.csv').open()))
    # Stage-mass composition by trace; descriptive only.
    labels=[f"{r['trace'].replace('live_','')}\n{r['phase']}" for r in rows]
    vals=[[float(r[k]) for r in rows] for k in ('dispatch_share_pct','expert_share_pct','combine_share_pct','wait_share_pct')]
    fig,ax=plt.subplots(figsize=(11,4.8)); bottom=[0]*len(rows)
    for v,n,c in zip(vals,['dispatch','expert','combine','event wait'],['#d95f02','#1b9e77','#7570b3','#e7298a']):
        ax.bar(labels,v,bottom=bottom,label=n,color=c); bottom=[a+b for a,b in zip(bottom,v)]
    ax.set_ylabel('share of summed logical CUDA stage (%)'); ax.set_title('Fresh live MoE phase mass'); ax.legend(ncol=4,fontsize=8); ax.tick_params(axis='x',labelsize=7); fig.tight_layout(); fig.savefig(OUT/'phase_mass.png',dpi=160); plt.close(fig)
    # Normal T_MoE curve per trace/phase.
    fig,ax=plt.subplots(figsize=(10,4.8))
    for r in rows:
        ax.plot([r['trace']+' '+r['phase']],[float(r['T_MoE_p50_ms'])], 'o', label=r['trace']+' '+r['phase'])
    ax.set_ylabel('T_MoE p50 (ms)'); ax.set_title('Layer-local T_MoE across fresh regimes'); ax.tick_params(axis='x',labelrotation=65,labelsize=7); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(OUT/'tmoe_regime.png',dpi=160); plt.close(fig)
    # Fixed-text request-level concurrency points (the c4 point is added as
    # soon as its run is complete).  They are a workload-regime observation,
    # not an apples-to-apples method speedup.
    vals=[]
    import json
    for name,c in [('live_text_c4',4),('live_text_c8',8),('live_text_c8_rep2',8),('live_text_c16',16)]:
        p=HERE/'poc_flashvep/deepep_revalidation/results/autonomous_ep_research_discovery_v3_20260906_172149'/name/'atlas_measured/atlas_summary.json'
        if p.exists(): vals.append((c,name.replace('live_text_',''),json.loads(p.read_text())['e2e'].get('p50_ms')))
    if vals:
        fig,ax=plt.subplots(figsize=(7,4.2)); ax.scatter([v[0] for v in vals],[v[2] for v in vals],s=55)
        for c,n,y in vals: ax.annotate(n,(c,y),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_xlabel('concurrency'); ax.set_ylabel('request E2E p50 (ms)'); ax.set_title('Fixed-text concurrency regime (fresh runs)'); fig.tight_layout(); fig.savefig(OUT/'e2e_vs_concurrency.png',dpi=160); plt.close(fig)
if __name__=='__main__': main()
