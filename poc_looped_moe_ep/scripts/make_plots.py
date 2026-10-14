"""Plots keep physical time and structural load proxies in separate axes."""
import csv
import gzip
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_looped_moe_ep.contract import *


def main():
    timing=json.loads((ROOT/'results/physical_timing.json').read_text())
    migration=json.loads((ROOT/'results/migration_roi.json').read_text())
    validity=json.loads((ROOT/'results/validity.json').read_text())
    model=json.loads((ROOT/'results/model_manifest.json').read_text())
    with gzip.open(ROOT/'results/per_loop_rank_metrics.csv.gz','rt') as inp:rows=list(csv.DictReader(inp))
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    def select(arm,loop):return [r for r in rows if r['placement']==arm and int(r['loop'])==loop and int(r['forward'])==0]
    def save(name):
        plt.tight_layout();plt.savefig(figures/name,dpi=160);plt.close()
    heat=[]
    for loop in range(16):
        vectors=[np.asarray(json.loads(r['expert_histogram']),float) for r in select('S0',loop)]
        heat.append(np.mean([v/v.sum() for v in vectors],axis=0))
    plt.figure(figsize=(12,4));plt.imshow(heat,aspect='auto');plt.colorbar(label='Mean request-normalized assignment fraction')
    plt.xlabel('Expert ID (one physical shared core pool)');plt.ylabel('Native loop index (0-based)')
    plt.title('Foil-1 H16; 24 held-out B1 native cache-free episodes');save('expert_routing_by_loop.png')
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for arm,ax in zip(['S0','S1'],axes):
        heat=np.asarray([np.mean([np.asarray(json.loads(r['rank_loads']),float)/int(r['paths'])
                                 for r in select(arm,h)],axis=0) for h in range(16)])
        im=ax.imshow(heat,aspect='auto',vmin=.1,vmax=.4)
        ax.scatter(heat.argmax(1),np.arange(16),color='red',s=10,label='Mean hottest rank')
        ax.set(xlabel='Logical EP rank (physical 4/5/6/7)',ylabel='Loop index',title=arm,xticks=range(4));ax.legend(fontsize=7)
    fig.colorbar(im,ax=axes,label='Assignment fraction',shrink=.7)
    fig.suptitle('Native H16, initial prefix; N=24 independent request episodes per loop')
    fig.subplots_adjust(right=.85,wspace=.3,top=.86)
    fig.savefig(figures/'rank_load_by_loop.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm,style in [('S0','-'),('S1','--')]:
        for metric,label in [('dispatch_ms','Dispatch'),('expert_ms','Expert'),('combine_ms','Combine')]:
            axes[0].plot(range(16),[np.mean([float(r[metric]) for r in select(arm,h)]) for h in range(16)],
                         style,label=f'{arm} {label}')
        axes[1].plot(range(16),[np.mean([float(r['rank_max_mean']) for r in select(arm,h)]) for h in range(16)],style,label=arm)
    axes[0].set(xlabel='Loop',ylabel='Light-event ms / core invocation',title='One cumulative-critical rank/request')
    axes[1].set(xlabel='Loop',ylabel='Rank assignment max/mean',title='Load improvement is not time improvement')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    save('physical_loop_cost.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    s=timing['summary']
    x=np.arange(2)
    axes[0].bar(x-.16,[s[a]['clean_ms']['mean'] for a in ['S0','S1']],.32,label='Clean 3-forward episode')
    axes[0].bar(x+.16,[s[a]['sparse_EP_ms']['mean'] for a in ['S0','S1']],.32,label='Measured routed D+E+C')
    axes[0].set(xticks=x,xticklabels=['S0','S1 proxy static'],ylabel='Actual measured ms',title='S2 timing: UNVALIDATED');axes[0].legend(fontsize=8)
    obs=timing['requests'];native=np.mean([r['static_sum_max'] for r in obs if r['placement']=='S0'])
    values=[native,np.mean([r['static_sum_max'] for r in obs if r['placement']=='S1']),
            np.mean([r['candidate_sum_max'] for r in obs if r['placement']=='S1'])]
    axes[1].bar(range(3),np.asarray(values)/native,color=['C0','C1','gray'])
    axes[1].set(xticks=range(3),xticklabels=['S0','S1','S2 candidate'],ylabel='Normalized assignment-max sum',title='STRUCTURAL ONLY — not latency')
    save('best_static_vs_oracle.png')
    c=timing['comparison'];keys=['S1_to_S2_candidate_structural_headroom','S1_to_ideal_structural_upper']
    values=[100*c[k]['mean'] for k in keys];errs=np.asarray([[100*(c[k]['mean']-c[k]['ci95'][0]),100*(c[k]['ci95'][1]-c[k]['mean'])] for k in keys]).T
    plt.figure(figsize=(7,4));plt.bar(range(2),values,yerr=errs,color=['gray','lightgray'],capsize=4)
    plt.xticks(range(2),['Future-truth load heuristic','Perfect-balance load bound'])
    plt.ylabel('STRUCTURAL assignment headroom (%)');plt.title('No validated physical S1→S2 latency oracle');save('oracle_headroom.png')
    plt.figure(figsize=(8,4));reuse=np.asarray([1,2,4,8,16]);m=migration['summary']
    for key,label in [('envelope_over_serial','Serial expert copies'),('envelope_over_parallel','Perfect edge-parallel expert copies'),
                      ('envelope_over_zero_launch_bulk','Zero-launch bulk/edge-parallel')]:
        plt.plot(reuse,reuse*m[key]['mean'],'o-',label=label)
    plt.axhline(1,color='black',linestyle=':');plt.axvline(1,color='red',linestyle=':')
    plt.xlabel('Hypothetical compatible reuse count (NOT automatically native H)')
    plt.ylabel('Zero-expert-time envelope / modeled copy cost')
    plt.title('Generous fixed-other-stages envelope; not measured benefit');plt.legend(fontsize=8);plt.yscale('log')
    save('movement_roi.png')
    plt.figure(figsize=(7,4));plt.bar(['Foil file incl. buffers','One selected H100','Four selected H100s'],
        [model['actual_weight_file_bytes']/1024**3,81559/1024,4*81559/1024])
    plt.ylabel('GiB (log scale)');plt.yscale('log');plt.title('Representativeness FAIL: tiny checkpoint fits one GPU');save('memory_footprint.png')
    verdict=dict(verdict='SMALL_MODEL_ONLY',tested_model_signal='NO_MATERIAL_PLACEMENT_LATENCY_GAIN_ON_TESTED_MODEL',
        method_decision='NO-GO: do not implement a placement/migration/replication method',
        true_EP4=validity['true_EP4'],GPU_mask=MASK,
        G0='PASS for authentic small-model physical EP4',
        G1='Routing/rank variation exists; material placement-sensitive latency not established',
        G2='UNVALIDATED: S2 is a load proxy, not a validated EP latency oracle',
        G3='Sparse EP is material only in a forced tiny-model eager EP4 setup',
        G4='NOT_ESTABLISHED: no measured net dynamic/E2E opportunity',
        G5='FAIL: 0.554B learned parameters; no representative EP4 checkpoint validated',
        measured_S0_S1=timing['comparison'],Stage_C='NOT_RUN; no promotion gates',
        S2_EP_latency_headroom='UNVALIDATED',full_generation_serving_E2E='NOT_RUN',
        quality='exact native/placement routing, sampled block outputs, last-token logits and greedy4; no accuracy benchmark',
        next_action='Close method implementation here. Reopen only with an accessible representative native-EP Looped MoE, not another oracle trick.',
        branch='flashvep/looped-moe-ep-poc')
    write(ROOT/'results/verdict.json',verdict)
    print(verdict['verdict'],'Figures:',len(list(figures.glob('*.png'))))


if __name__=='__main__':main()
