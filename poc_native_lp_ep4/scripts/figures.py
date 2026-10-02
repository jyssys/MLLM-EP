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
    data=read_json(a.metrics)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    route={(r['arm'],r['modality']):r for r in data['route_summary']}
    comp={(r['arm'],r['modality']):r for r in data['component_summary']}
    usable={(r['arm'],r['modality']):r['usable']
            for r in data['instrumentation_overhead']['stratified']}
    for modality in ('VISUAL_PREFILL','TOKEN_MATCHED_TEXT'):
        rows=[r for r in data['discovery_summary'] if r['modality']==modality and r['arm']!='DECODE_ONLY']
        fig,ax=plt.subplots(figsize=(13,6))
        counter_rows=[]
        offsets={'FULL':(-24,-25),'CP1024':(8,-12),'CP512':(-26,22),
                 'LP_AUTO':(-35,45),'LP_G4':(3,-30),'LP_G8':(10,3),
                 'CP256':(-15,26),'LP_G16':(10,3),'LP_G24':(8,5),'LP_G48':(-48,16)}
        for r in rows:
            key=('LP_AUTO' if r['arm']=='LP_G4' else r['arm']),modality
            ttft=r['ttft_ms'];itl=r['peak_itl_ms']
            color='#bc4b24' if r['arm'].startswith('CP') else '#267caa' if r['arm'].startswith('LP') else '#555555'
            ax.errorbar(ttft['mean'],itl['mean'],
                xerr=[[ttft['mean']-ttft['ci95'][0]],[ttft['ci95'][1]-ttft['mean']]],
                yerr=[[itl['mean']-itl['ci95'][0]],[itl['ci95'][1]-itl['mean']]],
                fmt='o',color=color,capsize=3)
            amp=route.get(key,{}).get('expert_reload_amp')
            share=comp.get(key,{}).get('communication_share')
            desc=r['arm']+(' (alias)' if r['arm']=='LP_G4' else '')
            reload_text='%.2fx'%amp['mean'] if amp else 'N/A'
            # This sum includes prepare/finalize (and peer readiness), not only wire transfer.
            share_text='%.0f%%'%(share['mean']*100) if share else 'N/A'
            if share and not usable.get(key,False):share_text+=' UNV'
            counter_rows.append([desc,reload_text,share_text])
            ax.annotate(desc,(ttft['mean'],itl['mean']),xytext=offsets.get(r['arm'],(5,5)),
                textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':color,'lw':.5})
        ax.set(xlabel='Model-ready TTFT (ms; CPU rendering excluded)',
               ylabel='Response-window peak victim ITL (ms)',
               title=f'Native TRUE EP4 discovery — {modality}\n3 paired-pool repeats; local span shares are diagnostic')
        ax.set_xscale('log')
        ax.grid(alpha=.2);fig.subplots_adjust(right=.69,bottom=.14,top=.87)
        counters=ax.table(cellText=counter_rows,colLabels=['Arm','Reload','D+C span'],
            bbox=[1.02,.26,.45,.70],colWidths=[.24,.09,.12])
        counters.auto_set_font_size(False);counters.set_fontsize(8)
        ax.text(1.02,.16,'D+C: local prepare+finalize share,\nincludes peer readiness; not wire share.\nUNV: failed overhead screen.\nX axis is logarithmic; all points measured.',
            transform=ax.transAxes,fontsize=8,va='top')
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
    # Matplotlib writes trailing spaces in SVG path commands. Normalize only
    # these newly generated vector files, never measured data or old figures.
    for path in a.output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':main()
