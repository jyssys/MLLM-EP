"""Generate editable SVG figures from saved experiment records."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT

def main():
    destination=ROOT/'figures'
    destination.mkdir(exist_ok=True)
    quality=ROOT/'pilot/sensitivity_summary.json'
    if quality.exists():
        data=json.loads(quality.read_text())
        fig,ax=plt.subplots(figsize=(8,4))
        for policy,color in [('P1','gray'),('P2','tab:blue'),('P3','tab:orange')]:
            selected=sorted([r for r in data['conditions'] if r['policy']==policy and r['status']=='COMPLETE'],
                key=lambda r:r['semantic_edge_fraction'])
            for r in selected:
                lo,hi=r['wilson_95_percent']
                ax.errorbar(100*r['semantic_edge_fraction'],r['accuracy_percent'],
                    yerr=[[r['accuracy_percent']-lo],[hi-r['accuracy_percent']]],color=color,marker='o',capsize=4)
            ax.plot([100*r['semantic_edge_fraction'] for r in selected],
                [r['accuracy_percent'] for r in selected],color=color,label=policy)
        ax.axhline(100*data['baseline_correct']/data['baseline_n'],ls='--',color='black',label='P0, 15/16')
        ax.set(xlabel='Semantically omitted original edges (%) — dense kernels still execute',
            ylabel='Validation accuracy (%)',ylim=(-2,102),
            title='Full free-running validation trajectories, N=16 per condition\nWilson 95% intervals. Not a sparse speedup measurement.')
        ax.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(destination/'validation_quality_work.svg')
        plt.close(fig)
    path=ROOT/'pilot/operator_cost_summary.json'
    if path.exists():
        data=json.loads(path.read_text())['conditions']
        if data:
            fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
            for ax,group in zip(axes,('A','B')):
                selected=[r for r in data if r['group']==group]
                keys=sorted({(r['layer'],r['full_rows']) for r in selected})
                for offset,method,color in [(-.23,'FULL','gray'),(0,'PRECOMPUTED_ROUTER_05','tab:blue'),(.23,'PRECOMPUTED_ROUTER_20','tab:orange')]:
                    vals=[next(r['mean_operator_wall_ms'] for r in selected if (r['layer'],r['full_rows'],r['method'])==(l,s,method)) for l,s in keys]
                    ax.bar(np.arange(len(keys))+offset,vals,.23,label=method,color=color)
                ax.set(xticks=np.arange(len(keys)),xticklabels=[f'L{l}/T{s}' for l,s in keys],
                    title=f'Isolated TRUE EP4, Group {group}',ylabel='Max-rank operator wall time (ms)')
                ax.tick_params(axis='x',rotation=45)
            axes[0].legend(fontsize=6)
            fig.suptitle('Physical operator measurements, 30 repeats per cell. Not request E2E speedup.')
            fig.tight_layout()
            fig.savefig(destination/'physical_operator_cost.svg')
            plt.close(fig)
    path=ROOT/'pilot/allocation_replay.json'
    if path.exists():
        data=json.loads(path.read_text())['aggregates']
        fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
        for ax,fraction in zip(axes,(.05,.20)):
            selected=[r for r in data if r['requested_fraction']==fraction]
            x=np.arange(len(selected))
            ax.bar(x-.18,[100*r['remote_pairs_reduction'] for r in selected],.36,label='Remote token-rank pairs')
            ax.bar(x+.18,[100*r['max_rank_padded_rows_reduction'] for r in selected],.36,label='Max-rank padded rows')
            ax.set(xticks=x,xticklabels=[r['policy'] for r in selected],title=f'Target edge removal {100*fraction:.0f}%',ylabel='Structural count reduction (%)')
        axes[0].legend(fontsize=7)
        fig.suptitle('TRACE_ESTIMATE: 16 requests, 7,296 sampled MoE calls. Not latency.')
        fig.tight_layout()
        fig.savefig(destination/'allocation_structural_tradeoff.svg')
        plt.close(fig)

if __name__=='__main__':
    main()
