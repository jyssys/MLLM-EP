"""Descriptive three-point curves; internal composite intervals labeled."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from poc_dispatch_combine_asymmetry.contract import load


def main():
    root=Path(__file__).resolve().parents[1]
    metrics=json.loads((root/'artifacts/final_metrics.json').read_text())['metrics']
    accounting=load(root/'artifacts/communication_accounting.json')['records']
    out=root/'figures';out.mkdir(exist_ok=True)
    curves=[('mllm_visual','VL visual'),('mllm_text','VL text'),('llm_text','LLM text')]
    for component in ('dispatch_ms','combine_ms','ratio'):
        fig,ax=plt.subplots(figsize=(6.8,4.4))
        for key,label in curves:
            rows=[metrics[f'{key}_{size}_clean'] for size in ('low','medium','high')]
            ax.plot([r['prompt_tokens'] for r in rows],[r[component]['mean'] for r in rows],'-o',label=label)
        ax.set(xlabel='Actual prompt tokens',ylabel=component.replace('_',' '),
               title='Native EP4: cumulative-critical rank, mean across 48 layers')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'tokens_vs_{component}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),sharey=True)
    for ax,(key,label) in zip(axes,curves):
        rows=[metrics[f'{key}_{size}_internal'] for size in ('low','medium','high')]
        x=[r['prompt_tokens'] for r in rows]
        layout=np.array([r['subcomponents']['get_dispatch_layout']['mean'] for r in rows])
        api=np.array([r['subcomponents']['dispatch']['mean'] for r in rows])
        remainder=np.array([r['dispatch_ms']['mean'] for r in rows])-layout-api
        ax.stackplot(x,layout,api,remainder,labels=['layout API','dispatch API incl. rendezvous/tail','unattributed prepare residual'])
        ax.set(title=label,xlabel='Actual prompt tokens');ax.grid(alpha=.2)
    axes[0].set_ylabel('Perturbed diagnostic duration (ms/layer), not pure wire time')
    fig.suptitle('UNVALIDATED INTERNAL DURATIONS — classification only, not clean baseline',fontsize=11)
    axes[0].legend(fontsize=7);fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/'tokens_vs_dispatch_composites.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for key,label in curves:
        x=[];yd=[];yc=[]
        for size in ('low','medium','high'):
            recs=[r for r in accounting if f"{r['model']}_{r['arm']}"==key and r['size']==size]
            x.append(np.mean([r['logical_dispatch_bytes'] for r in recs])/1e6)
            row=metrics[f'{key}_{size}_clean'];yd.append(row['dispatch_ms']['mean']);yc.append(row['combine_ms']['mean'])
        axes[0].plot(x,yd,'-o',label=label);axes[1].plot(x,yc,'-o',label=label)
    for ax,title in zip(axes,['Dispatch','Combine']):
        ax.set(xlabel='Logical useful BF16 bytes per layer (MB)',ylabel='ms/layer',title=title);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'logical_bytes_vs_latency.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
