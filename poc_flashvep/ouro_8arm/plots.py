"""Figures from measured records only. Missing arms receive no synthetic bars."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contract import *


def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    numeric=read(ROOT/'numerical_gate.json')['comparisons']
    pairs=sorted({r['pair'] for r in numeric})
    fig,ax=plt.subplots(figsize=(8,4))
    for pair in pairs:
        y=[]
        for P in [128,256,512,2048]:
            values=[r['max_relative_L2'] for r in numeric if r['pair']==pair and r['P']==P]
            y.append(max(values) if values else np.nan)
        ax.plot([128,256,512,2048],y,'o-',label=pair)
    ax.axhline(.03,color='black',ls=':',label='Historical diagnostic level, not a new PASS gate')
    ax.set(xscale='log',xlabel='Prompt tokens',ylabel='Worst sampled component relative L2',
        title='Aligned last-token seams. Two synthetic prompts per length.')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'numerical_component_error.png',dpi=160);plt.close(fig)
    table=list(csv.DictReader((ROOT/'accuracy_by_arm.csv').open())) if (ROOT/'accuracy_by_arm.csv').exists() else []
    if table:
        preferred={}
        primary=dict(S4='quality_S4_canonical',T4='quality_T4_custom_pilot_v1',L4='quality_L4_pilot_v1',O4='quality_O4_pilot_v1',
            SD='quality_SD_pilot_v1',TD='quality_TD_pilot_v1',LD='quality_LD_pilot_v1',OD='quality_OD_pilot_v1')
        for r in table:
            if primary.get(r['arm'])!=r['run']:continue
            key=(r['arm'],r['dataset'])
            if key not in preferred or (int(r['N']),r['run'])>(int(preferred[key]['N']),preferred[key]['run']):preferred[key]=r
        fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
        for ax,domain in zip(axes,['gsm8k','math500']):
            values=[preferred[(a,domain)] for a in ARMS if (a,domain) in preferred]
            ax.bar([r['arm'] for r in values],[float(r['accuracy_percent']) for r in values])
            for i,r in enumerate(values):ax.text(i,float(r['accuracy_percent'])+1,f"{r['correct']}/{r['N']}",ha='center',fontsize=8)
            ax.set(title=domain,ylabel='Actual answer accuracy (%)',ylim=(0,100))
        fig.suptitle('Custom-backend pilot. Missing arms are not evaluated. TP remains exploratory.',fontsize=10)
        fig.tight_layout();fig.savefig(out/'actual_pilot_accuracy.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,domain in zip(axes,['gsm8k','math500']):
            values=[r for a in ARMS for r in table if r['run']=='quality400_'+a and r['dataset']==domain]
            ax.bar([r['arm'] for r in values],[float(r['accuracy_percent']) for r in values])
            for i,r in enumerate(values):ax.text(i,float(r['accuracy_percent'])+1,f"{r['correct']}/{r['N']}",ha='center',fontsize=8)
            ax.set(title=domain,ylabel='Actual answer accuracy (%)',ylim=(0,100))
        fig.suptitle('Fixed 400-question target: 300 GSM8K + 100 MATH. TP remains exploratory.',fontsize=10)
        fig.tight_layout();fig.savefig(out/'eight_arm_accuracy400.png',dpi=160);plt.close(fig)
    if (ROOT/'numerical/fp32_seams.json').exists():
        rows=read(ROOT/'numerical/fp32_seams.json');fig,ax=plt.subplots(figsize=(8,4))
        for key in ['attention_16heads_vs_fp32','attention_4heads_vs_fp32','projection_monolithic_vs_same_input_fp32','projection_fp32_partials_vs_same_input_fp32']:
            ax.plot([r[key]['relative_L2'] for r in rows],'.-',label=key)
        ax.set(yscale='log',xlabel='Captured same-input seam',ylabel='Relative L2 vs local FP32 operation reference',title='Local numerical diagnosis. Not a full-model FP32 run.')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'fp32_local_references.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
