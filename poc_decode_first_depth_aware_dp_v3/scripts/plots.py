"""Plot measured observations and model estimates in separate panels."""
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_decode_first_depth_aware_dp_v3.contract import *

def save(fig,name):
    target=ROOT/'figures';target.mkdir(exist_ok=True)
    fig.tight_layout();fig.savefig(target/name,dpi=150);plt.close(fig)

def main():
    metrics=read(ROOT/'artifacts/live_dp4_metrics.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for policy in ['round_robin','queue','length','strong_cost']:
        rows=[r for r in metrics if not r['trace'] and r['arrival']=='burst' and r['policy']==policy]
        for ax,key,label in zip(axes,['E2E_P95_ms','ITL_P95_ms'],['P95 E2E (ms)','P95 host-visible ITL (ms)']):
            ax.plot([r['concurrency'] for r in rows],[r[key] for r in rows],'-o',label=policy,markersize=3)
            ax.set(xlabel='Total burst requests across DP4',ylabel=label,title='Measured online, one discovery run/arm')
    axes[0].legend(fontsize=7);save(fig,'live_dp4_scaling.png')
    fit=read(ROOT/'artifacts/cdb_cost_model.json')
    fig,ax=plt.subplots(figsize=(5,4))
    for field in ['E2E_P95_ms','ITL_P95_ms']:
        x=[r['actual'][field] for r in fit['validation']]
        y=[r['predicted'][field] for r in fit['validation']]
        ax.scatter(x,y,s=12,label=field)
    limit=max(max(r['actual'].values()) for r in fit['validation'])
    ax.plot([0,limit],[0,limit],'k--',linewidth=.8)
    ax.set(xlabel='Live observation (ms)',ylabel='Cost-model prediction (ms)',title='Held-out validation, not a speedup')
    ax.legend(fontsize=7);save(fig,'cost_model_validation.png')
    refined=read(ROOT/'artifacts/oracle_refinement_v2.json')['rows']
    fig,ax=plt.subplots(figsize=(6,4))
    for r in refined:
        low,high=r['CI95_gain_percent']
        ax.errorbar(r['C'],r['gain_percent'],yerr=[[r['gain_percent']-low],[high-r['gain_percent']]],fmt='o',capsize=3)
    ax.axhline(0,color='k',linewidth=.8);ax.axhline(10,color='gray',linestyle='--')
    ax.set(xlabel='Total burst requests',ylabel='P95 gain over best non-depth control (%)',
        title='Modeled clairvoyant headroom, not measured speedup\n95% paired request-bootstrap intervals')
    ax.set_xticks([r['C'] for r in refined])
    save(fig,'oracle_headroom.png')
    physical=read(ROOT/'artifacts/live_hindsight_confirmation.json')['rows']
    fig,ax=plt.subplots(figsize=(6,4))
    for r in physical:
        low,high=r['CI95_gain_percent']
        ax.errorbar(r['C'],r['gain_percent'],yerr=[[r['gain_percent']-low],[high-r['gain_percent']]],fmt='s',capsize=3)
    ax.axhline(0,color='k',linewidth=.8)
    ax.set(xlabel='Total burst requests',ylabel='Measured paired P95 E2E gain (%)',
        title='Live hindsight placement, three repeats, same pilot\n95% paired hierarchical bootstrap intervals')
    ax.set_xticks([r['C'] for r in physical])
    save(fig,'live_hindsight_confirmation.png')
    predictor=read(ROOT/'artifacts/causal_predictor.json')
    fig,ax=plt.subplots(figsize=(6,4))
    for policy in ['ridge_depth_length','ridge_depth_matched_length']:
        rows=[r for r in predictor['scheduling'] if r['policy']==policy]
        ax.plot([r['C'] for r in rows],[r['gain_vs_generic_percent'] for r in rows],'-o',label=policy)
    ax.axhline(0,color='k',linewidth=.8)
    ax.set(xlabel='Total burst requests',ylabel='Modeled P95 gain vs generic (%)',title='Held-out prompt-only predictor, no RL')
    ax.legend(fontsize=7);save(fig,'causal_predictor_scheduling.png')
    costs=read(ROOT/'outputs/cost_surface_v1/cost_surface.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for context in [1024,4096,8192]:
        batch=sorted({r['batch'] for r in costs if r['context']==context and r['status']=='MEASURED_SYNTHETIC_PROMPT'})
        for ax,key,label in zip(axes,['mean_decode_span_ms','mean_TTFT_ms'],['Mean decode span (ms)','Mean TTFT (ms)']):
            y=[statistics.mean(r[key] for r in costs if r['context']==context and r['batch']==b and r.get('repeat',0)>0) for b in batch]
            ax.plot(batch,y,'-o',label=f'context={context}')
            ax.set(xlabel='Local batch size',ylabel=label,title='Physical synthetic-context diagnostic')
    axes[0].legend(fontsize=7);save(fig,'native_cdb_cost_surface.png')

if __name__=='__main__':main()
