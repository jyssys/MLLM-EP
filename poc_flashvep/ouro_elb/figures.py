"""Small figures that separate grouping, initial state, and rebatching."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT

def main():
    rows=list(csv.DictReader((ROOT/'benchmark_summary.csv').open()))
    lanes=['PARALLEL_CHARACTERIZATION','MATCHED_DECODE_FRONTIER','NATIVE_MIXED_GROUPED_QUIET']
    names=['Ordinary Refill\n8 process/GPU blocks','Aligned frontier\n8 process/GPU blocks','Native Mixed\n3 process/GPU blocks']
    fig,ax=plt.subplots(figsize=(9,4.4));x=np.arange(3)
    for j,D in enumerate((2,4)):
        selected=[next(r for r in rows if r['lane']==lane and r['P']=='2048' and r['N']=='16' and r['D']==str(D)) for lane in lanes]
        y=np.array([float(r['throughput_speedup']) for r in selected]);lo=np.array([float(r['CI95_low']) for r in selected]);hi=np.array([float(r['CI95_high']) for r in selected])
        ax.bar(x+(j-.5)*.28,y,width=.27,label=f'D{D}/matched D1',yerr=[y-lo,hi-y],capsize=4)
    ax.axhline(1,color='black',linewidth=.8);ax.set_xticks(x,names);ax.set_ylim(.94,1.18)
    ax.set_ylabel('LIVE completed-token rate speedup, pointwise 95% CI');ax.legend()
    ax.set_title('P2048/N16, cap64. Initial-state and native co-packing controls. Not full-answer serving.')
    fig.tight_layout();fig.savefig(ROOT/'figures/frontier_and_mixed_controls.png',dpi=160);plt.close(fig)
    profile=list(csv.DictReader((ROOT/'cpu_gpu_overhead.csv').open()))
    selected=sorted((r for r in profile if r['P']=='256' and r['N']=='1'),key=lambda r:int(r['D']))
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    ax.bar(x-.15,[int(r['decode_submissions']) for r in selected],.3,label='Decode submissions')
    ax.bar(x+.15,[int(r['graph_replays']) for r in selected],.3,label='Recurrent graph replays')
    ax.set_xticks(x,['D1','D2','D4']);ax.set_ylabel('Actual count in 15 decoded token forwards')
    ax.set_title('HEAVY event audit: P256/N1. Each arm executes 60 recurrent row-passes.');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/submission_vs_graph_counts.png',dpi=160);plt.close(fig)
    pairs=list(csv.DictReader((ROOT/'paired_interventions.csv').open()))
    first=next(r for r in pairs if r['P']=='256' and r['old']=='1' and r['new']=='1')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    old=[float(first[k]) for k in ('KEEP_old_ms','YIELD_old_ms','NO_REFORM1_old_ms')]
    all_rows=[float(first[k]) for k in ('KEEP_all_ms','YIELD_all_ms','NO_REFORM1_all_ms')]
    ax.bar(x-.15,old,.3,label='Old row output complete');ax.bar(x+.15,all_rows,.3,label='Both rows output complete')
    ax.set_xticks(x,['KEEP4','YIELD1 / REFORM','D1 / NO REFORM']);ax.set_ylabel('Measured ms from identical source state')
    ax.set_title('Controlled SYNC component: P256, 1 old + 1 new row. One paired state, not population CI.');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/rebatch_priority_tradeoff.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
