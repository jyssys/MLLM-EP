"""Editable data figures. Every plotted series names its evidence type."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import ROOT

NAMES={'ifm_ouro':'IFM MoE Ouro','ifm_huginn':'IFM MoE Huginn','foil1':'Foil-1'}

def read(name):
    p=ROOT/name
    return list(csv.DictReader(p.open())) if p.exists() else []

def save(fig,name):
    target=ROOT/'plots'
    target.mkdir(exist_ok=True)
    fig.savefig(target/f'{name}.svg',bbox_inches='tight')
    fig.savefig(target/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    routes=read('routing_summary.csv')
    representations=read('representation_summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for col,model in enumerate(NAMES):
        for pairing,label in [('temporal_gap1','Same token, adjacent pass'),('random_position_frequency_preserving','Position-shuffled null')]:
            rows=[r for r in routes if r['model']==model and r['split']=='heldout' and r['regime']=='prefill'
                and r['dataset']=='language' and r['metric']=='jaccard' and r['pairing']==pairing]
            rows.sort(key=lambda r:int(r['pass_index']))
            if rows:
                x=[int(r['pass_index'])+1 for r in rows]
                y=[float(r['mean']) for r in rows]
                axes[0,col].plot(x,y,'o-',label=label,markersize=3)
                axes[0,col].fill_between(x,[float(r['ci95_lo']) for r in rows],[float(r['ci95_hi']) for r in rows],alpha=.15)
        for metric,label in [('expert_output_cosine','Matched expert output cosine'),('expert_output_relative_l2','Matched expert output relative L2')]:
            rows=[r for r in representations if r['model']==model and r['split']=='heldout' and r['regime']=='prefill'
                and r['dataset']=='language' and r['metric']==metric and r['pairing']=='temporal_gap1']
            rows.sort(key=lambda r:int(r['pass_index']))
            if rows:
                axes[1,col].plot([int(r['pass_index'])+1 for r in rows],[float(r['mean']) for r in rows],'o-',label=label,markersize=3)
        axes[0,col].set_title(NAMES[model])
        axes[0,col].set_ylabel('Top-k set Jaccard')
        axes[1,col].set_ylabel('Similarity / relative error')
        for ax in axes[:,col]:
            ax.set_xlabel('Current recurrent pass (1-based)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('LIVE probes: held-out language, B1, prefix cap 512, 128 independent sequences per model\nBands: sequence bootstrap 95% CI. Same-expert output similarity is not reuse correctness.')
    fig.tight_layout()
    save(fig,'routing_vs_output')

    confidence=read('confidence_contribution_summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for col,model in enumerate(NAMES):
        for metric,label in [('normalized_input_relative_l2','Input relative L2 (all tokens)'),
                ('same_expert_input_relative_l2','Input relative L2 (matched expert)'),
                ('weighted_output_relative_l2','Weighted output relative L2')]:
            rows=sorted([r for r in representations if r['model']==model and r['split']=='heldout'
                and r['regime']=='prefill' and r['dataset']=='language' and r['pairing']=='temporal_gap1'
                and r['metric']==metric],key=lambda r:int(r['pass_index']))
            axes[0,col].plot([int(r['pass_index'])+1 for r in rows],[float(r['mean']) for r in rows],'o-',label=label,markersize=3)
        for metric,label in [('confidence','Max normalized router probability'),('cancellation_ratio','Norm of sum / sum of norms')]:
            rows=sorted([r for r in confidence if r['model']==model and r['dataset']=='language' and r['metric']==metric],
                key=lambda r:int(r['pass_index']))
            axes[1,col].plot([int(r['pass_index'])+1 for r in rows],[float(r['mean']) for r in rows],'o-',label=label,markersize=3)
        for ax in axes[:,col]:
            ax.set_xlabel('Current recurrent pass (1-based)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=6)
        axes[0,col].set_title(NAMES[model])
        axes[0,col].set_ylabel('Relative L2')
        axes[1,col].set_ylabel('Ratio / probability')
    fig.suptitle('LIVE probes, B1: upper row uses 128 held-out passages, prefix cap 512\nLower row uses 8 smoke passages/model. Normalized all-expert scores preserve each native router convention.')
    fig.tight_layout()
    save(fig,'input_contribution_confidence')

    counter=read('counterexamples.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,model in zip(axes,NAMES):
        rows=[r for r in counter if r['model']==model]
        ax.hist([float(r['output_relative_l2']) for r in rows],bins=12)
        ax.set_title(f'{NAMES[model]}: n={len(rows)}')
        ax.set_xlabel('Same-expert output relative L2')
        ax.set_ylabel('Selected counterexamples')
    fig.suptitle('LIVE probe counterexamples: largest 100 errors across all recorded datasets/lengths\nThis selected tail is not a population histogram.')
    fig.tight_layout()
    save(fig,'counterexample_tail')

    quality=read('quality_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,model in zip(axes,NAMES):
        for prefix,label in [('O1_drop_','Contribution drop oracle'),('O2_reuse_','Output reuse oracle'),('C1_router_','Causal router threshold'),('C2_reuse_','Causal reuse guard'),('N1_random_drop_','Legacy correlated random'),('N1_independent_site_drop_','Independent-site random')]:
            rows=[r for r in quality if r['model']==model and r['split']=='heldout' and r['arm'].startswith(prefix)
                and not r['arm'].endswith('renorm1')]
            rows.sort(key=lambda r:float(r['effective_removed_fraction']))
            if rows:
                ax.plot([100*float(r['effective_removed_fraction']) for r in rows],[float(r['delta_nll']) for r in rows],'o-',label=label,markersize=3)
        ax.axhline(.02,color='black',linestyle='--',label='NLL target 0.02 nats/token')
        ax.set_title(NAMES[model])
        ax.set_xlabel('Mean per-sequence omitted edges (%)')
        ax.set_ylabel('Next-token NLL change (nats/token)')
        ax.set_yscale('symlog',linthresh=.025)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1))
    fig.suptitle('LIVE full-model interventions, 128 held-out passages/model, B1, prefix cap 512\nOracles use offline expert outputs. Curves are not measured speedups.')
    fig.tight_layout()
    save(fig,'quality_work_tradeoff')

    timing=read('timing_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,model in zip(axes,NAMES):
        rows=sorted([r for r in timing if r['model']==model],key=lambda r:int(r['input_length']))
        if rows:
            x=[int(r['input_length']) for r in rows]
            ax.errorbar(x,[float(r['prefill_ms']) for r in rows],
                yerr=[[float(r['prefill_ms'])-float(r['ci95_lo_ms']) for r in rows],
                    [float(r['ci95_hi_ms'])-float(r['prefill_ms']) for r in rows]],fmt='o-',label='Clean forward wall time')
            ax.plot(x,[float(r['expert_event_span_ms']) for r in rows],'s--',label='Separate expert event spans')
        ax.set_title(NAMES[model])
        ax.set_xlabel('Input tokens')
        ax.set_ylabel('Milliseconds')
        ax.set_xscale('log',base=2)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle('LIVE B1 native forward: 3 process starts × 20 clean repetitions/cell, physical GPU0\nSpans and clean wall time come from separate runs. This is not a stacked E2E partition.')
    fig.tight_layout()
    save(fig,'timing_denominator')

    micro=read('expert_microbench_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,model in zip(axes,NAMES):
        for length in [128,512,2048]:
            rows=sorted([r for r in micro if r['model']==model and int(r['input_length'])==length],key=lambda r:float(r['target_q']))
            if rows:
                baseline=float(rows[0]['component_ms'])
                ax.plot([100*float(r['effective_q']) for r in rows],[baseline/float(r['component_ms']) for r in rows],'o-',label=f'Prefix {length}')
        ax.axhline(1,color='black',linestyle='--')
        ax.set_title(NAMES[model])
        ax.set_xlabel('Compacted expert rows (%)')
        ax.set_ylabel('Native expert component speedup (×)')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle('LIVE native-weight operator microbench: one recurrent site/model, 3 starts × 20 repetitions\nSelection, compaction, routing, cache, and combine are excluded. Not E2E speedup.')
    fig.tight_layout()
    save(fig,'expert_granularity')

    oracle=read('oracle_summary.csv')+read('candidate_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,model in zip(axes,NAMES):
        for prefix,label in [('O1_','Contribution drop oracle'),('O2_','Output reuse oracle'),('C1_','Causal router threshold'),('C2_','Causal reuse guard')]:
            rows=[r for r in oracle if r['model']==model and r['arm'].startswith(prefix) and r['upper_ci_nll_gate']=='True'
                and not r['arm'].endswith('renorm1')]
            rows.sort(key=lambda r:float(r['effective_removed_fraction_work_weighted']))
            if rows:
                ax.plot([100*float(r['effective_removed_fraction_work_weighted']) for r in rows],
                    [float(r['clean_denominator_optimistic_speedup']) for r in rows],'o-',label=label)
        ax.axhline(1.10,color='black',linestyle='--',label='1.10× research gate')
        ax.set_ylim(.995,1.12)
        ax.set_title(NAMES[model])
        ax.set_xlabel('Actually omitted expert work (%)')
        ax.set_ylabel('Analytical E2E speedup (×)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('ANALYTICAL ONLY: prefix 512, upper NLL CI ≤0.02, zero selection/cache overhead\nFLOP-proportional expert-time assumption. No candidate E2E speedup was measured.')
    fig.tight_layout()
    save(fig,'quality_constrained_amdahl')

if __name__=='__main__':
    main()
