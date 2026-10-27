"""Editable figures from recorded data. No generated illustration replaces a trace."""
import csv
import gzip
import json
import re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT
from .affinity import coactivation,fit_transition

def read(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []

def save(fig,name):
    fig.tight_layout()
    fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=150,bbox_inches='tight')
    plt.close(fig)

def heatmaps():
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for column,model in enumerate(['ifm_ouro','ifm_huginn','foil1']):
        metadata=json.loads((ROOT/'raw'/f'{model}_native_fa3_smoke_v1/metadata.rank0.json').read_text())
        sites=[v['site'] for v in metadata['physical_moe_sites'] if v['recurrent']]
        site=min(sites,key=lambda s:tuple(map(int,re.findall(r'\d+',s))))
        manifest=json.loads((ROOT/'raw'/f'{model}_route_arrays_manifest.json').read_text())
        lookup={tuple(v['key']):v['index'] for v in manifest}
        arrays=np.load(ROOT/'raw'/f'{model}_route_arrays.npz')
        e=64 if model=='foil1' else 25
        key=(4,512,'general',site,0,'heldout')
        ids=arrays[f'ids_{lookup[key]}'];count=coactivation(ids,e)
        image=axes[0,column].imshow((count+count.T)/len(ids),vmin=0,cmap='magma')
        axes[0,column].set(title=f'{model}: {site}, pass 0\nHeld-out pairs, n={len(ids)} token samples',xlabel='Expert ID',ylabel='Expert ID')
        fig.colorbar(image,ax=axes[0,column],label='Pair frequency / token invocation',fraction=.046)
        key=(*key[:5],'calibration')
        previous=arrays[f'ids_{lookup[key]}'];nxt=arrays[f'ids_{lookup[(*key[:4],1,"calibration")]}']
        conditional,_=fit_transition(previous,nxt,e)
        image=axes[1,column].imshow(conditional,vmin=0,vmax=1,cmap='viridis')
        axes[1,column].set(title=f'Pass 0 -> 1, calibration n={len(previous)}\nMulti-label probabilities: each row sums to 2',xlabel='Next selected expert ID',ylabel='Previous selected expert ID')
        fig.colorbar(image,ax=axes[1,column],label='P(next expert is selected)',fraction=.046)
    fig.suptitle('Native router observations: B4, prefix 512, general text. Not a latency or speedup measurement.',y=1.02)
    save(fig,'coactivation_and_transition')

def recall():
    data=read('transition_affinity.csv');fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,model in zip(axes,['ifm_ouro','ifm_huginn','foil1']):
        rows=[v for v in data if v['model']==model and int(v['batch'])==4 and int(v['context'])==512 and int(v['lag'])==1]
        loops=sorted(set(int(v['previous_loop']) for v in rows))
        for metric,label in [('conditional_recall2','Previous-loop conditional'),('popularity_recall2','Pass/domain popularity')]:
            values=[np.mean([float(v[metric]) for v in rows if int(v['previous_loop'])==r]) for r in loops]
            ax.plot(loops,values,'o-',label=label)
        ax.set(title=model,xlabel='Previous pass index (0-based)',ylabel='Held-out Recall@2',ylim=(0,1));ax.legend(fontsize=8)
    fig.suptitle('B4, prefix 512. Mean across physical recurrent sites and four domains. Calibration/held-out prompts are disjoint.')
    save(fig,'heldout_recall_by_loop')

def placement():
    data=read('placement_structural.csv');fig,axes=plt.subplots(2,3,figsize=(13,8))
    methods=['static','popularity_balanced','generic_coactivation','recurrent_transition_graph']
    for col,model in enumerate(['ifm_ouro','ifm_huginn','foil1']):
        for row,ep in enumerate([4,8]):
            ax=axes[row,col]
            for name in methods:
                selected=[v for v in data if v['model']==model and int(v['ep'])==ep and v['method']==name]
                if not selected:continue
                x=np.mean([float(v['unique_destinations_mean']) for v in selected]);y=np.mean([float(v['rank_load_cv']) for v in selected])
                ax.scatter([x],[y],label=name,s=60)
            ax.set(title=f'{model} EP{ep}',xlabel='Mean destination ranks / token',ylabel='Mean per-call rank-load CV');ax.legend(fontsize=7)
    fig.suptitle('HELD-OUT FIXED-PLACEMENT TRACE MODEL. Same expert counts and HBM capacity. Not measured E2E speedup.')
    save(fig,'placement_locality_vs_load')

def timings():
    data=[v for v in read('baseline_e2e.csv') if v['status']=='PASS']
    generic={(v['model'],v['ep'],v['batch'],v['prefix_tokens']):v for v in read('generic_control.csv')}
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,model in zip(axes,['ifm_ouro','ifm_huginn','foil1']):
        for ep in [4,8]:
            selected=sorted([v for v in data if v['model']==model and int(v['ep'])==ep and int(v['prefix_tokens'])==128],key=lambda v:int(v['batch']))
            if not selected:continue
            ax.plot([int(v['batch']) for v in selected],[float(v['output_tokens_s']) for v in selected],'o-',label=f'Static EP{ep}')
        ax.set(title=model,xlabel='Global request batch',ylabel='Generated output tokens / s',xticks=[1,2,4,8,16]);ax.legend()
    fig.suptitle('LIVE TRUE EP: prefix 128, 32 greedy tokens, EOS disabled. Full-prefix recomputation. No cached-decode claim.')
    save(fig,'throughput_vs_batch')

def timelines():
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [4,8]:
            base=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1'
            traces=[]
            for rank in range(ep):
                path=base/f'kineto_b1_p128.rank{rank}.json.gz'
                if not path.exists():break
                with gzip.open(path,'rt') as f:traces.append(json.load(f))
            if len(traces)!=ep:continue
            common=min(d['baseTimeNanoseconds'] for d in traces)
            def stamp(d,e):return e['ts']+(d['baseTimeNanoseconds']-common)/1000
            start=min(stamp(d,e) for d in traces for e in d['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X')
            fig,ax=plt.subplots(figsize=(14,max(3,ep*.55)))
            for rank,d in enumerate(traces):
                events=[e for e in d['traceEvents'] if e.get('ph')=='X']
                for label,color in [('compute','#2878b5'),('NCCL','#e07b39'),('copy','#65a765')]:
                    selected=[e for e in events if (e.get('cat')=='kernel' and ('nccl' in e['name'].lower())==(label=='NCCL') and label!='copy')
                        or (label=='copy' and e.get('cat') in ['gpu_memcpy','gpu_memset'])]
                    intervals=[((stamp(d,e)-start)/1000,e['dur']/1000) for e in selected]
                    ax.broken_barh(intervals,(rank-.35,.7),facecolors=color,label=label if rank==0 else None)
            ax.set(xlabel='Same-host CUPTI timeline (ms)',ylabel='EP rank',yticks=list(range(ep)),
                title=f'{model} EP{ep}, B1, prefix 128: separate heavy observer\nBlue: non-NCCL kernels. Orange: NCCL including waits. Green: copies/memset. Not clean E2E.')
            ax.legend(loc='upper right');save(fig,f'{model}_ep{ep}_gpu_timeline')

def main():
    heatmaps();recall();placement();timings();timelines()

if __name__=='__main__':main()
