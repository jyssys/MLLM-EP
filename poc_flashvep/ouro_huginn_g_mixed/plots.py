"""Editable vector figures from actual measurements, with explicit missing phases."""
import csv
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,DEPTHS

def save(fig,name):
    target=ROOT/'figures';target.mkdir(exist_ok=True)
    fig.savefig(target/f'{name}.svg',bbox_inches='tight')
    fig.savefig(target/f'{name}.png',bbox_inches='tight',dpi=150)
    plt.close(fig)

def main():
    fig,ax=plt.subplots(figsize=(14,4))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[(.08,.5,'Current ready queues\nNative prelude'),(.32,.5,'Legal recurrent frontier\nFixed hybrid control'),
        (.59,.8,'Pure prompt Gk\nCommon legal remainder'),(.59,.2,'Native Mixed G1\nPrompt + decode rows'),
        (.87,.5,'Native producer event\nPer-depth KV + hidden\nFull depth before output')]
    for x,y,label in boxes:
        ax.text(x,y,label,ha='center',va='center',fontsize=10,
            bbox=dict(boxstyle='round,pad=.5',fc='#edf3f8',ec='#28475e'))
    for a,b in [((.18,.5),(.22,.5)),((.42,.55),(.49,.76)),((.42,.45),(.49,.24)),
            ((.69,.76),(.78,.57)),((.69,.24),(.78,.43))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#28475e'))
    ax.set_title('Legal execution modes — engineering contract, not measured performance\nAll prompt/decode tokens retain Ouro R4 or Huginn R16. No multi-loop heterogeneous Mixed descriptor.')
    save(fig,'legal_hybrid_dag')
    rows=list(csv.DictReader((ROOT/'fixed_matrix.csv').open()))
    for model in ['ouro','huginn']:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,n in zip(axes,[1,4]):
            selected=[r for r in rows if r['model']==model and int(r['P'])==2048 and int(r['N'])==n
                and r['lane']=='CLEAN_NATURAL_CAPPED_COHORT']
            groups=defaultdict(list)
            for row in selected:groups[(row['arm'],int(row['G']))].append(row)
            for (arm,g),items in groups.items():
                x=np.mean([float(r['TTFT_long_ms']) for r in items])
                y=np.mean([float(r['ITL_p99_ms']) for r in items])
                ax.scatter(x,y,label=f'{arm} G{g}',s=50)
                ax.annotate(f'{arm} G{g}',(x,y),fontsize=7)
            ax.set(xlabel='Target TTFT (ms, mean cohort value)',ylabel='Background ITL P99 (ms, mean cohort quantile)',
                title=f'{model}, P2048/C512, {n} background requests')
            ax.grid(alpha=.2)
        fig.suptitle('LIVE_MEASURED — clean bounded natural generation; not sustainable capacity')
        save(fig,f'{model}_TTFT_TBT_pareto')
        fig,ax=plt.subplots(figsize=(8,4))
        keys=[r for r in rows if r['model']==model and int(r['P'])==2048 and int(r['N'])==0]
        grouped=defaultdict(list)
        for row in keys:grouped[(row['arm'],int(row['G']))].append(float(row['TTFT_long_ms']))
        labels=[f'{a} G{DEPTHS[model] if a=="atomic" else g}' for a,g in grouped]
        ax.bar(np.arange(len(labels)),[np.mean(grouped[k]) for k in grouped])
        ax.set_xticks(np.arange(len(labels)),labels,rotation=65,ha='right')
        ax.set(ylabel='TTFT (ms)',title=f'{model}: no-ready-decode P2048/C512\nLIVE_MEASURED, one process, three technical repetitions')
        save(fig,f'{model}_no_decode_group_cost')
    # The page is a scope marker, not an invented attainment curve.
    for model in ['ouro','huginn']:
        fig,ax=plt.subplots(figsize=(8,3))
        ax.axis('off')
        ax.text(.5,.65,f'{model}: SLO attainment vs request rate',ha='center',fontsize=14)
        ax.text(.5,.35,'NOT_RUN\nNo adaptive method or qualified capacity result is plotted.',
            ha='center',va='center',fontsize=12)
        save(fig,f'{model}_attainment_NOT_RUN')
    event_path=ROOT/'gpu_event_intervals.csv'
    if event_path.exists():
        events=list(csv.DictReader(event_path.open()))
        for model in ['ouro','huginn']:
            fig,axes=plt.subplots(4,1,figsize=(12,6),sharex=True)
            for ax,(arm,g) in zip(axes,[('mixed',1),('hybrid',2),('group',2),('group',4)]):
                items=[r for r in events if r['model']==model and r['arm']==arm and int(r['G'])==g]
                for op,color in [('prelude','tab:green'),('recurrent','tab:blue'),('coda','tab:orange')]:
                    segments=[(float(r['GPU_start_ms']),float(r['GPU_stream_interval_ms'])) for r in items
                        if r['operation']==op and r.get('GPU_start_ms')]
                    if segments:ax.broken_barh(segments,(0,1),facecolors=color,label=op)
                ax.set(ylim=(0,1),yticks=[],ylabel=f'{arm} G{g}')
            axes[0].legend(loc='upper right',ncol=3)
            axes[-1].set_xlabel('GPU event time since common origin (ms)')
            fig.suptitle(f'{model}: LIVE_MEASURED producer-stream intervals, heavy lane\nIntervals include launch gaps. They are not exclusive GPU kernel busy time.')
            save(fig,f'{model}_gpu_stream_timeline')
    # Actual submission frontiers explain the same-batch and pure-group modes.
    paths=list((ROOT/'raw').glob('fixed_screen_ouro_P2048_N4_s0/results.json'))
    if paths:
        data=json.loads(paths[0].read_text())
        fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=False)
        for ax,arm in zip(axes,['mixed','group']):
            row=next(r for r in data if r['arm']==arm and (arm=='mixed' or r['G']==4))
            plans=[p for p in row['plans'] if any(s['id']=='long' for s in p['spans'])]
            if plans:
                origin=plans[0]['host_s']
                for i,plan in enumerate(plans):
                    x=(plan['host_s']-origin)*1000
                    color='tab:orange' if plan.get('mixed') else 'tab:blue'
                    ax.scatter(x,plan['group'],c=color,s=20)
            ax.set(ylabel='Submitted loop count',title=f'{arm}: host submission timeline, not GPU kernel intervals')
            ax.grid(alpha=.2)
        axes[-1].set_xlabel('Host time since first target submission (ms)')
        fig.suptitle('LIVE_MEASURED plan observer; orange = native Mixed G1')
        save(fig,'legal_plan_timeline')

if __name__=='__main__':main()
