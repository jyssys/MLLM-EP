"""All figures derive from compact request/iteration tables, not projections."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from poc_dllm_ep_comm.contract import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);a=p.parse_args()
    source=ROOT/'results'/a.result;dest=ROOT/'figures'/a.result
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    df=pd.read_csv(source/'requests.csv');it=pd.read_csv(source/'iterations.csv')
    primary=df[df.block_length==32];names=set(primary.request);it=it[it.request.isin(names)&(it.iteration>=0)]
    def save(name):
        plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=155);plt.close()
    fig,ax=plt.subplots(figsize=(7,4))
    groups=list(primary.groupby('domain'));x=np.arange(len(groups));width=.22
    for n,(label,key) in enumerate([('One-shot',None),('Live-only','Residual_Amp'),('Actual','Raw_Amp')]):
        means=[1 if key is None else g[key].mean() for _,g in groups]
        ax.bar(x+(n-1)*width,means,width,label=label)
    ax.set_xticks(x,[d for d,g in groups]);ax.set_ylabel('Remote D+C bytes / acceptance-route one-shot')
    ax.set_title('Physical EP4 routes; live-only and one-shot are structural views');ax.legend()
    save('normalized_amplification')
    fig,ax=plt.subplots(figsize=(12,5))
    order=primary.sort_values(['domain','request']).reset_index(drop=True);x=np.arange(len(order))
    for shift,label,values in [(-.25,'One-shot',np.ones(len(order))),(0,'Live-only',order.Residual_Amp),(.25,'Actual',order.Raw_Amp)]:
        ax.bar(x+shift,values,.25,label=label)
    ax.set_xticks(x,order.request.str.replace('_b32','',regex=False),rotation=90,fontsize=7)
    ax.set_ylabel('Remote D+C / each request one-shot bytes');ax.legend()
    save('per_request_three_views')
    fig,ax=plt.subplots(figsize=(7,4))
    s=json.loads((source/'summary.json').read_text())['stats']['b32_all']['Residual_Amp']
    ax.hist(primary.Residual_Amp,bins=10,alpha=.7)
    ax.axvline(s['median'],color='black',label=f"Median {s['median']:.2f}x")
    ax.axvspan(*s['ci95'],color='orange',alpha=.25,label=f"Mean95% request CI [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]")
    ax.set_xlabel('Residual amplification');ax.set_ylabel('Requests');ax.legend()
    save('residual_distribution')
    fig,ax=plt.subplots(figsize=(8,4))
    for domain,group in primary.groupby('domain'):
        ax.scatter(group.Raw_Amp,group.Residual_final_positions_Amp,label=domain)
    ax.axhline(2,color='black',linestyle='--',label='Residual GO threshold')
    ax.set_xlabel('Actual / one-shot');ax.set_ylabel('Conservative live-only / one-shot');ax.legend()
    save('request_amplification')
    representative=it[it.request==sorted(names)[0]]
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    axes[0].plot(representative.iteration,representative.live_tokens,label='Live before')
    axes[0].plot(representative.iteration,representative.newly_accepted_tokens,label='Newly accepted')
    axes[0].set_ylabel('Token positions');axes[0].legend()
    axes[1].plot(representative.iteration,representative.bytes_per_accepted_token/2**20,label='Actual MB / newly accepted')
    axes[1].set_ylabel('MiB / newly accepted')
    for idx in representative[representative.status=='ZERO_ACCEPT_ITERATION'].iteration:
        axes[1].axvline(idx,color='red',alpha=.5)
    axes[2].plot(representative.iteration,representative.actual_bytes/2**20,label='Actual remote D+C')
    axes[2].plot(representative.iteration,representative.live_only_bytes/2**20,label='Live-only structural')
    axes[2].set_ylabel('MiB / iteration');axes[2].set_xlabel('Denoising iteration (red lines: zero-accept)');axes[2].legend()
    save('iteration_acceptance_efficiency')
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    for ax,key,label in zip(axes,['active_destination_ranks','rank_cv','hottest_rank'],['Active destination ranks','Assignment rank CV','Hottest EP rank']):
        ax.plot(representative.iteration,representative[key],marker='.',label=label);ax.set_ylabel(label)
    axes[2].set_xlabel('Denoising iteration');save('rank_diagnostics_over_iterations')
    fig,ax=plt.subplots(figsize=(7,4))
    for block,group in df.groupby('block_length'):
        ax.scatter(np.full(len(group),block),group.Residual_final_positions_Amp,alpha=.55)
        ax.scatter(block,group.Residual_final_positions_Amp.mean(),color='black',marker='x',s=70)
    ax.axhline(2,color='black',linestyle='--');ax.set_xticks([32,64]);ax.set_xlabel('Official block size (secondary64 fixed before results)');ax.set_ylabel('Conservative residual amplification')
    save('block_robustness')
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(len(groups));bottom=np.zeros(len(groups))
    for key,label in [('context','Prompt/context'),('dead_output','Already accepted output'),('live_only','Live output')]:
        vals=[np.mean(g['B_'+key]/g.B_actual)*100 for d,g in groups]
        ax.bar(x,vals,bottom=bottom,label=label);bottom+=vals
    ax.set_xticks(x,[d for d,g in groups]);ax.set_ylabel('% of actual logical remote D+C payload');ax.legend()
    save('actual_traffic_decomposition')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for domain,group in primary.groupby('domain'):
        axes[0].scatter(group.rank_edge_density,group.hottest_rank_switch_rate,label=domain)
        axes[1].scatter(group.comm_share_request_wall*100,group.timing_overhead_fraction*100,label=domain)
    axes[0].set_xlabel('Mean directed rank-edge density');axes[0].set_ylabel('Hottest-rank switch rate')
    axes[1].set_xlabel('D+C / whole request wall % (conservative)');axes[1].set_ylabel('Timing observer vs bypass overhead %');axes[1].axhline(10,color='black',linestyle='--')
    axes[0].legend();save('rank_and_timing_diagnostics')

if __name__=='__main__':main()
