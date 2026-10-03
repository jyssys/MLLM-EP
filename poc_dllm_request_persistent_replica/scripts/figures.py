"""Required plots; structural accounting is never labeled runtime speedup."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_dllm_request_persistent_replica.contract import *


def save(name):
    plt.tight_layout();plt.savefig(ROOT/'figures_v2'/name,dpi=160);plt.close()


def main():
    dest=ROOT/'figures_v2'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir()
    summary=json.loads(summary_path().read_text());choice=summary['selected']
    policy,budget,cap=choice['policy'],choice['budget_gib'],choice['load_guard']
    locality=pd.read_csv(ROOT/'results/persistence_v2/block_locality.csv')
    d=locality[(locality.lane=='long')&(locality.split=='heldout')]
    plt.figure(figsize=(8,4))
    for p in ('first_block','first2_blocks'):
        r=d[d.policy==p].merge(d[d.policy=='global'],on=['request','target_block'],suffixes=('_r','_g'))
        r['advantage']=100*(r.coverage_at_16_r-r.coverage_at_16_g)
        plt.plot(r.groupby('distance_r').advantage.mean(),marker='o',label=p)
    plt.axhline(0,color='black',lw=.7);plt.xlabel('Future block distance');plt.ylabel('Expert coverage@16 advantage vs Global (pp)')
    plt.title('Request-specific signal, live-only, held-out');plt.legend();save('predictor_advantage_vs_block_distance.png')
    df=pd.read_csv(ROOT/'results/simulation_long_v2/policy_budget.csv')
    held=df[df.split=='heldout'];plt.figure(figsize=(8,4))
    for p in ('global_destination','global_frequency','first_block','first2_blocks','oracle_load'):
        r=held[(held.policy==p)&(held.load_guard==.10)].groupby('budget_gib').residual_amp.mean()
        plt.plot(r,marker='o',label=p)
    oracle=held[(held.policy=='oracle')&(held.load_guard==-1)].groupby('budget_gib').residual_amp.mean()
    plt.plot(oracle,marker='o',ls='--',label='Oracle unconstrained')
    plt.axhline(held.baseline_amp.mean(),color='gray',ls=':',label='Current long live-only baseline')
    plt.axhline(1,color='black',ls=':',label='One-shot structural reference')
    plt.xlabel('Memory cap (GiB / source rank)');plt.ylabel('Residual EP-byte amplification');plt.legend(fontsize=7,ncol=2)
    plt.title('History load cap +10%; actual future load checked separately');save('residual_amplification_vs_budget.png')
    plt.figure(figsize=(8,4))
    h=summary['horizon'];x=[z['future_index'] for z in h];y=[100*z['mean'] for z in h]
    err=np.array([[100*(z['mean']-z['ci95'][0]) for z in h],[100*(z['ci95'][1]-z['mean']) for z in h]])
    plt.errorbar(x,y,yerr=err,fmt='o-');plt.axhline(0,color='black',lw=.7)
    plt.xlabel('Future block index after profiling');plt.ylabel('Additional D+C byte reduction vs Global (%)')
    plt.title('Apparent byte gain, NOT load-matched; request-bootstrap 95% CI');save('gain_over_global_vs_future_block.png')
    cost=pd.read_csv(ROOT/'results/copy_cost/request_copy_roi.csv')
    c=cost[(cost.lane=='long')&(cost.split=='heldout')&(cost.policy==policy)&(cost.budget_gib==budget)&(cost.load_guard==cap)]
    lengths=pd.read_csv(ROOT/'results/long_requests.csv')[['request','output_tokens']];c=c.merge(lengths,on='request')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(c.output_tokens,c.byte_roi);axes[0].axhline(2,color='gray',ls='--')
    axes[0].set(xlabel='Generated tokens (native EOS/block cap)',ylabel='Saved D+C bytes / copy bytes',title='Byte ROI ≠ timing ROI')
    axes[1].scatter(c.output_tokens,c.projected_byte_break_even_remaining_fraction)
    axes[1].axhline(.5,color='gray',ls='--');axes[1].set_yscale('log')
    axes[1].set(xlabel='Generated tokens',ylabel='Projected break-even / remaining iterations',title='Byte model, not measured time')
    save('copy_roi_and_break_even_vs_length.png')
    plt.figure(figsize=(8,4))
    for p in ('global_destination','first_block','first2_blocks','oracle_load','oracle'):
        r=held[held.policy==p].groupby(['budget_gib','load_guard'])[['dc_reduction','worst_stage_load_increase']].mean()
        plt.scatter(100*r.dc_reduction,100*r.worst_stage_load_increase,label=p)
    plt.axhline(10,color='gray',ls='--');plt.xlabel('Live-only D+C byte reduction (%)');plt.ylabel('Mean request worst layer/iteration load increase (%)')
    plt.title('Worst-stage hotspots, not cumulative request-average load');plt.legend(fontsize=7);save('communication_compute_pareto.png')
    allocations=pd.read_csv(ROOT/'results/simulation_long_v2/replica_allocation.csv.gz')
    r=allocations[(allocations.split=='heldout')&(allocations.policy==policy)&(allocations.budget_gib==budget)&(allocations.load_guard==cap)]
    matrix=np.zeros((19,12));edges=[(s,d) for s in range(4) for d in range(4) if s!=d]
    for (l,s,d),group in r.groupby(['layer','source','owner']):matrix[l-1,edges.index((s,d))]=len(group)/r.request.nunique()
    plt.figure(figsize=(10,5));plt.imshow(matrix,aspect='auto');plt.colorbar(label='Mean persistent replica count / request')
    plt.xticks(range(12),[f's{s} ← owner{d}' for s,d in edges],rotation=45,ha='right');plt.yticks(range(19),range(1,20))
    plt.ylabel('Sparse layer (expert IDs are layer-specific)');save('layer_source_destination_allocation.png')
    plt.figure(figsize=(8,4))
    for p in ('first_block','first2_blocks'):
        values=[]
        for b in BUDGETS:
            result=summary['paired_contrasts']['long_heldout_all'][f'{p}-global@{b:g}@0.1']['persistent_oracle_recovery']
            values.append(100*result['mean'] if result else np.nan)
        plt.plot(BUDGETS,values,marker='o',label=p)
    plt.axhline(50,color='gray',ls='--');plt.xlabel('GiB / source rank');plt.ylabel('Recovered oracle headroom beyond Global (%)')
    plt.title('Apparent recovery, NOT load-matched; not a passing oracle gate');plt.legend();save('persistent_oracle_recovery.png')


if __name__=='__main__':main()
