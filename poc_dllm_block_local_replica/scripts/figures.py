"""All figures distinguish structural references from physical runtime facts."""
import json
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_dllm_block_local_replica.contract import *

LABELS=dict(global_='Global/static',first_frequency='First-iter frequency',first_destination='First-iter destination',
            first2_destination='First-2 destination',oracle='Future-truth greedy',oracle_load='Load-capped greedy')
LABELS['global']=LABELS.pop('global_')


def main():
    dest=ROOT/'figures'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir()
    df=pd.read_csv(ROOT/'results/policy_summary.csv')
    data=df[df.cohort=='heldout_b32_all'];summary=json.loads(final_summary_path().read_text())
    stat=summary['stats']['heldout_b32_all']
    plt.rcParams.update({'figure.dpi':140,'font.size':9})
    fig,ax=plt.subplots(figsize=(8,5))
    for p,r in data.groupby('policy'):
        r=r.sort_values('budget_gib');key=[f'{p}@{b:g}' for b in r.budget_gib]
        lo=[stat[k]['residual_amp']['ci95'][0] for k in key];hi=[stat[k]['residual_amp']['ci95'][1] for k in key]
        ax.plot(r.budget_gib,r.residual_amp,'o-',label=LABELS[p]);ax.fill_between(r.budget_gib,lo,hi,alpha=.08)
    ax.axhline(5.77267,color='gray',ls='--',label='Prior32-request live-only5.773x')
    ax.axhline(data.baseline_amp.iloc[0],color='black',ls=':',label='This24-request holdout live-only5.426x')
    ax.axhline(1,color='gray',ls='-.',label='Original one-shot reference1x')
    ax.set(xlabel='Replica budget (GiB / source rank, shared across19 layers)',ylabel='Post-replica residual amplification',
           title='STRUCTURAL accounting only; not measured speedup');ax.legend(fontsize=7,ncol=2);fig.tight_layout()
    fig.savefig(dest/'residual_vs_budget.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for p,r in data[~data.policy.isin(['oracle','oracle_load'])].groupby('policy'):
        r=r.sort_values('budget_gib');ax.plot(r.budget_gib,100*r.oracle_recovery,'o-',label=LABELS[p])
    ax.axhline(50,color='black',ls='--',label='50% causal GO gate')
    ax.set(xlabel='Budget GiB/rank',ylabel='Recovered future-truth greedy headroom (%)',title='24 held-out problems');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'oracle_recovery.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for p,r in data.groupby('policy'):
        r=r.sort_values('budget_gib')
        for ax,y in zip(axs,['max_rank_load_increase','worst_stage_load_increase']):
            ax.plot(100*r.dc_reduction,100*r[y],'o-',label=LABELS[p])
        for row in r.itertuples():axs[0].annotate(f'{row.budget_gib:g}',(row.dc_reduction*100,row.max_rank_load_increase*100),fontsize=6)
    axs[0].set(ylabel='Cumulative max-rank assignment increase (%)')
    axs[1].set(ylabel='Worst iteration/layer max-rank increase (%)')
    for ax in axs:ax.set(xlabel='Live-only D+C byte reduction (%)');ax.axhline(10,color='gray',ls='--');ax.legend(fontsize=6)
    fig.suptitle('Assignment-based load proxy, NOT expert execution latency');fig.tight_layout();fig.savefig(dest/'comm_vs_load_pareto.png');plt.close(fig)
    chosen,budget=summary['descriptive_candidate'].split('@');budget=float(budget)
    plans=pd.read_csv(ROOT/'results/simulation/layer_replica_plan.csv.gz')
    heldout=set(json.loads((ROOT/'results/design_freeze.json').read_text())['heldout'])
    plans=plans[(plans.policy==chosen)&(plans.budget_gib==budget)&plans.request.str.endswith('_b32')&
                plans.request.map(lambda x:x.rsplit('_b',1)[0] in heldout)]
    counts=plans.groupby(['layer','source_rank','remote_destination']).size().unstack(['source_rank','remote_destination'],fill_value=0)
    fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(counts.values,aspect='auto',cmap='viridis')
    ax.set_xticks(range(len(counts.columns)),[f'S{s} <- owner{d}' for s,d in counts.columns],rotation=50,ha='right')
    ax.set_yticks(range(len(counts.index)),counts.index);ax.set(ylabel='Sparse layer',title=f'{chosen}, {budget:g}GiB/rank: replica allocations across held-out blocks')
    fig.colorbar(im,ax=ax,label='Allocated layer-expert replicas (all blocks/requests)');fig.tight_layout();fig.savefig(dest/'layer_replica_allocation.png');plt.close(fig)
    counts.to_csv(ROOT/'results/layer_allocation_matrix.csv')
    fig,ax=plt.subplots(figsize=(8,3));ax.axis('off')
    ax.text(.02,.80,'Runtime replica timing: NOT RUN',fontsize=15)
    ax.text(.02,.55,'Stage A failed causal-recovery and copy-ROI gates.\nNo D+C reduction, E2E gain, or speedup is claimed.',fontsize=11)
    ax.text(.02,.18,'Physical evidence collected: TRUE EP4 revalidation + expert-sized/bulk P2P copies.',fontsize=9)
    fig.tight_layout();fig.savefig(dest/'runtime_performance.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3));ax.axis('off')
    ax.text(.02,.8,'Baseline revalidation: PASS',fontsize=14)
    ax.text(.02,.52,'Two probes: token / acceptance / NFE / router-ID / weight / receive-buffer parity.',fontsize=9)
    ax.text(.02,.25,'Replica runtime parity: NOT TESTED\nGSM8K / HumanEval accuracy delta: NOT MEASURED (Stage A stopped).',fontsize=10)
    fig.tight_layout();fig.savefig(dest/'accuracy_summary.png');plt.close(fig)
    bound=pd.read_csv(ROOT/'results/copy_roi_upper_bound.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    for p,r in data.groupby('policy'):
        r=r.sort_values('budget_gib');ax.plot(r.budget_gib,r.byte_roi,'o-',label=LABELS[p])
    ax.set(xlabel='Budget GiB/rank',ylabel='Avoided D+C bytes / copied expert bytes',yscale='log',title='Cold block-local copy byte ROI; timing ROI not measured')
    ax.axhline(2,color='black',ls='--',label='ROI gate2x');ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(dest/'copy_roi.png');plt.close(fig)


if __name__=='__main__':main()
