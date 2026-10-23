"""Editable measured-data figures. Missing experiments produce no fake point."""
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT

def load(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []

def main():
    target=ROOT/'figures';target.mkdir(exist_ok=True)
    rows=load('controlled_endpoint_service.csv')
    if rows:
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,P in zip(axes,(2048,8192)):
            values=np.full((3,4),np.nan)
            for i,C in enumerate((128,512,2048)):
                for j,G in enumerate((1,2,3,4)):
                    v=[float(r['service_ms']) for r in rows if int(r['P'])==P and int(r['C'])==C and int(r['G'])==G and int(r['depth'])==0]
                    if v:
                        values[i,j]=np.median(v)
            image=ax.imshow(values,aspect='auto');fig.colorbar(image,ax=ax,label='Median stream service, ms')
            ax.set(xticks=range(4),xticklabels=[1,2,3,4],yticks=range(3),yticklabels=[128,512,2048],
                xlabel='G',ylabel='Actual C',title=f'Fixed KV endpoint {P}, depth 0\nMeasured component, shared host')
            for i in range(3):
                for j in range(4):
                    if np.isfinite(values[i,j]):
                        ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white')
        fig.savefig(target/'A_cost_surface.svg');plt.close(fig)
    data=load('serving_cohort_metrics.csv')
    live=[r for r in data if r['run_id'].startswith(('cb_p3_','cb_p5_'))]
    if live:
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for policy in sorted({r['policy'] for r in live}):
            values=[r for r in live if r['policy']==policy and r['ITL_pooled_P99_ms']]
            ax.scatter([float(r['TTFT_P95_ms']) for r in values],[float(r['ITL_pooled_P99_ms']) for r in values],label=policy,alpha=.65)
        ax.set(xlabel='Cohort TTFT P95, ms',ylabel='Within-cohort pooled ITL P99, ms',
            title='Measured bounded natural cohorts\nNot sustainable capacity or mean of request P99')
        ax.legend();fig.savefig(target/'C_completed_work_pareto.svg');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for policy in sorted({r['policy'] for r in live}):
            grouped=defaultdict(list)
            for r in live:
                if r['policy']==policy and r.get('budget_ms') and r['ITL_pooled_P99_ms']:
                    grouped[float(r['budget_ms'])].append(r)
            x=sorted(grouped)
            if x:
                axes[0].plot(x,[np.mean([float(r['ITL_pooled_P99_ms']) for r in grouped[b]]) for b in x],marker='o',label=policy)
                axes[1].plot(x,[np.mean([float(r['TTFT_P95_ms']) for r in grouped[b]]) for b in x],marker='o',label=policy)
        axes[0].set(ylabel='Mean cohort pooled ITL P99, ms');axes[1].set(ylabel='Mean cohort TTFT P95, ms')
        for ax in axes:
            ax.set(xlabel='Configured prefill budget, ms');ax.legend(fontsize=8)
        fig.suptitle('Budget intervention: measured cohorts, not an ITL guarantee')
        fig.savefig(target/'B_budget_vs_stall.svg');plt.close(fig)
    print('Figures use existing measured CSV rows only:',target)

if __name__=='__main__':
    main()
