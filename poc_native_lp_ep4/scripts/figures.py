"""Plot completed measured points only; no placeholder/extrapolated curves."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from poc_native_lp_ep4.contract import read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=read_json(a.metrics);a.output.mkdir(parents=True,exist_ok=True)
    route={(r['arm'],r['modality']):r for r in data['route_summary']}
    comp={(r['arm'],r['modality']):r for r in data['component_summary']}
    for modality in ('VISUAL_PREFILL','TOKEN_MATCHED_TEXT'):
        rows=[r for r in data['discovery_summary'] if r['modality']==modality and r['arm']!='DECODE_ONLY']
        fig,ax=plt.subplots(figsize=(10,6))
        for r in rows:
            key=r['arm'],modality;ttft=r['ttft_ms'];itl=r['peak_itl_ms']
            color='#bc4b24' if r['arm'].startswith('CP') else '#267caa' if r['arm'].startswith('LP') else '#555555'
            ax.errorbar(ttft['mean'],itl['mean'],
                xerr=[[ttft['mean']-ttft['ci95'][0]],[ttft['ci95'][1]-ttft['mean']]],
                yerr=[[itl['mean']-itl['ci95'][0]],[itl['ci95'][1]-itl['mean']]],
                fmt='o',color=color,capsize=3)
            amp=route.get(key,{}).get('expert_reload_amp')
            share=comp.get(key,{}).get('communication_share')
            desc=r['arm']+('\nreload %.2fx'%amp['mean'] if amp else '\nreload N/A')
            # This sum includes prepare/finalize (and peer readiness), not only wire transfer.
            desc+=('\nprepare+finalize %.0f%%'%(share['mean']*100) if share else '\nspan share N/A')
            ax.annotate(desc,(ttft['mean'],itl['mean']),xytext=(5,5),textcoords='offset points',fontsize=7)
        ax.set(xlabel='Model-ready TTFT (ms; CPU rendering excluded)',
               ylabel='Response-window peak victim ITL (ms)',
               title=f'Native TRUE EP4 discovery — {modality}\n3 paired-pool repeats; local span shares are diagnostic')
        ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(a.output/('pareto_'+modality.lower()+'.svg'))
        fig.savefig(a.output/('pareto_'+modality.lower()+'.png'),dpi=160)
        plt.close(fig)
    rows=[r for r in data['final_summary'] if r['modality']=='VISUAL_PREFILL' and r['arm']!='DECODE_ONLY']
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,key,label in zip(axs,('ttft_ms','peak_vsel_ms'),('Model-ready TTFT (ms)','Paired peak VSEL (ms)')):
        means=[r[key]['mean'] for r in rows]
        err=[[r[key]['mean']-r[key]['ci95'][0] for r in rows],[r[key]['ci95'][1]-r[key]['mean'] for r in rows]]
        ax.bar([r['arm'] for r in rows],means,yerr=err,capsize=4,color='#267caa')
        ax.set_ylabel(label);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen final comparison — 10 repeats, fresh decode-only pairing')
    fig.tight_layout();fig.savefig(a.output/'final_visual.svg');fig.savefig(a.output/'final_visual.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
