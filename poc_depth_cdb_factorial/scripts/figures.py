"""Draw measured factorial results. Do not combine modeled and measured series."""
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from poc_depth_cdb_factorial.contract import *
from poc_depth_cdb_factorial.scripts.analyze import records

def save(name):
    plt.tight_layout()
    plt.savefig(ROOT/'figures'/name,dpi=170)
    plt.close()

def main():
    rows=read(ROOT/'artifacts/scaling_summary.json')
    plt.figure(figsize=(8,4))
    for arm in ['A1','A2','A3','A4']:
        points=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['C'])
        if points:plt.plot([r['C'] for r in points],[r['speedup'] for r in points],marker='o',label=arm)
    plt.axhline(1,color='gray',linestyle='--',label='A0, one GPU')
    plt.xlabel('Total burst requests C')
    plt.ylabel('Measured makespan speedup vs matching A0 (×)')
    plt.legend(ncol=3)
    save('five_arm_scaling.png')
    increments=read(ROOT/'artifacts/factorial_incremental.json')
    plt.figure(figsize=(8,4))
    for field,label in [('no_refill_hindsight','A1 / A2: no-refill'),('refill_hindsight','A3 / A4: refill')]:
        x=[r['C'] for r in increments]
        y=np.array([r[field]['speedup'] for r in increments])
        low=np.array([r[field]['CI95'][0] for r in increments])
        high=np.array([r[field]['CI95'][1] for r in increments])
        if len(x):plt.errorbar(x,y,yerr=[np.maximum(0,y-low),np.maximum(0,high-y)],marker='o',capsize=3,label=label)
    plt.axhline(1,color='gray',linestyle='--')
    plt.xlabel('Total burst requests C')
    plt.ylabel('Incremental hindsight makespan speedup (×)')
    plt.legend()
    save('incremental_hindsight.png')
    if (ROOT/'artifacts/rank_time_analysis.json').exists():
        ranks=read(ROOT/'artifacts/rank_time_analysis.json')
        plt.figure(figsize=(8,4))
        for arm in ['A1','A2','A3','A4']:
            selected=[r for r in ranks if r['arm']==arm]
            plt.scatter([r['actual_loop_work'] for r in selected],[r['core_GPU_ms'] for r in selected],label=arm,alpha=.7)
        plt.xlabel('Actual loop-token work per rank')
        plt.ylabel('Measured core stream intervals (ms), trace lane')
        plt.legend()
        save('depth_work_vs_gpu_time.png')
        plt.figure(figsize=(8,4))
        for arm in ['A1','A2','A3','A4']:
            hist=collections.Counter()
            for r in ranks:
                if r['C']==64 and r['arm']==arm:
                    for key,count in r['core_batch_histogram'].items():hist[int(key.rsplit(':',1)[1])]+=count
            total=sum(hist.values())
            if total:plt.plot(sorted(hist),[hist[n]/total for n in sorted(hist)],marker='o',label=arm)
        plt.xlabel('Active token rows per recurrent launch (padding excluded)')
        plt.ylabel('Fraction of recurrent launches, trace C64')
        plt.legend()
        save('no_refill_refill_batch_histogram.png')
    for C in [32,64]:
        fig,axes=plt.subplots(4,1,figsize=(11,8),sharex=True)
        colors={'stage_prefill_batch':'tab:orange','compute_recurrent_batch':'tab:blue',
            'stage_coda_batch':'tab:green','compute_prelude_batch':'tab:purple','exit_KV_copy':'tab:red'}
        for _,_,d,ranks,_ in records():
            c=d['case']
            if not d.get('trace') or c['kind']!='speed' or c['C']!=C or c['arm']=='A0':continue
            ax=axes[int(c['arm'][1:])-1]
            for rank,raw in ranks:
                for name,color in colors.items():
                    intervals=[(s.get('gpu_start_s',s['host_start_s']),s['gpu_ms']/1000) for s in raw['stages'] if s['stage']==name]
                    if intervals:ax.broken_barh(intervals,(rank-.35,.7),facecolors=color,linewidth=0)
            ax.set_ylabel(c['arm']+' rank')
            ax.set_yticks([0,1,2,3])
        axes[-1].set_xlabel('Seconds from host burst origin (CUDA-event clock alignment, trace lane)')
        axes[0].legend(handles=[Patch(facecolor=color,label=name) for name,color in colors.items()],
            loc='upper right',fontsize=6,ncol=3)
        fig.suptitle(f'Measured compute-stream intervals, C{C}. Gaps are not automatically whole-GPU idle.')
        save(f'rank_timeline_C{C}.png')
    if (ROOT/'artifacts/quality_summary.json').exists():
        data=read(ROOT/'artifacts/quality_summary.json')
        fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
        for ax,domain in zip(axes,['gsm8k','math500']):
            points=sorted([r for r in data if r['dataset']==domain],key=lambda r:r['arm'])
            ax.bar([r['arm'] for r in points],[r['accuracy_pct'] for r in points])
            ax.set_title(domain+' (128 questions per arm)')
            ax.set_ylim(0,100)
        axes[0].set_ylabel('Measured answer accuracy (%)')
        save('five_arm_quality.png')
    print('FIGURES_READY')

if __name__=='__main__':main()
