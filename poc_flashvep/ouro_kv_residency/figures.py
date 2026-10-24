"""Plot measured operators separately from modeled capacity and clean serving."""
import csv
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT

def save(fig,name):
    directory=ROOT/'figures';directory.mkdir(exist_ok=True)
    fig.tight_layout();fig.savefig(directory/f'{name}.png',dpi=160);fig.savefig(directory/f'{name}.svg');plt.close(fig)

def main():
    baseline=list(csv.DictReader((ROOT/'baseline_summary.csv').open()))
    fig,ax=plt.subplots(figsize=(8,4))
    for n in (1,2,4,8):
        context=np.array([1024,2048,4096,8192,16384,32768,65408])
        ax.plot(context/1024,context*n*786432/2**30,'--',label=f'B{n} logical BF16 (MODEL)')
    ax.axhline(54,color='black',label='54 GiB native physical pool (LIVE allocation)')
    ax.set(xlabel='Prompt context, Ki tokens',ylabel='GiB',title='A. Original R4 KV. Payload is not HBM traffic.')
    ax.set_ylim(0,110);ax.legend(fontsize=8);save(fig,'A_KV_footprint_vs_context_batch')
    windows=json.loads((ROOT/'profile_complete_token_windows.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4))
    for n in (1,4):
        points=[]
        for run in sorted({r['run'] for r in windows if r['run'].endswith(f'_N{n}')}):
            rs=[r for r in windows if r['run']==run and r['complete_R4_geometry']]
            p=int(run.split('_P')[1].split('_')[0]);points.append((p,np.mean([r['attention_exclusive_GPU_service_ms'] for r in rs]),np.mean([r['output_completion_interval_ms'] for r in rs])))
        points.sort()
        if points:
            ax.plot([p/1024 for p,a,t in points],[a for p,a,t in points],'o-',label=f'B{n} attention exclusive GPU service')
            ax.plot([p/1024 for p,a,t in points],[t for p,a,t in points],'s--',label=f'B{n} sampling-to-sampling interval')
    ax.set(xlabel='Prompt context, Ki tokens',ylabel='ms',title='B. LIVE heavy trace. 14 complete R4 windows/shape. Not clean E2E shares.')
    ax.legend(fontsize=8);save(fig,'B_decode_critical_path_vs_context')
    summary=json.loads((ROOT/'operator_summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for batch in (1,8):
        ax=axes[0 if batch==1 else 1]
        for world in (2,4,8):
            rs=[r for r in summary if r['P']==world and r['B']==batch and r['placement']=='balanced' and not r.get('cache_pressure')]
            rs.sort(key=lambda r:r['L'])
            ax.plot([r['L']/1024 for r in rs],[r['speedup_wall'] for r in rs],'o-',label=f'P{world}')
        ax.axhline(1,color='gray',linestyle='--');ax.set(xlabel='Visible context, Ki tokens',ylabel='Local / distributed wall latency',title=f'B{batch} real model-derived operator')
        ax.legend()
    fig.suptitle('E. LIVE 2/4/8 GPU attention. NCCL and merge included. Not serving speedup.')
    save(fig,'E_2_4_8gpu_KV_scaling')
    fig,ax=plt.subplots(figsize=(8,4))
    for world in (2,4,8):
        points=[]
        for p in sorted({r['L'] for r in summary}):
            for r in summary:
                if r['P']==world and r['L']==p and r['B']==1 and r['placement']=='balanced' and not r.get('cache_pressure'):points.append(r)
        ax.plot([r['L']*8192/world/2**20 for r in points],[r['distributed_wall_ms'] for r in points],'o-',label=f'P{world}')
    ax.set(xlabel='Mean valid KV per rank, MiB per physical-layer/depth namespace',ylabel='Operator wall latency, ms',
        title='G. LIVE operator memory–latency trade-off. Full-cache serving capacity NOT_RUN.')
    ax.legend();save(fig,'G_memory_capacity_vs_serving_latency_pareto')
    operators=list(csv.DictReader((ROOT/'attention_operator_pairs.csv').open()))
    if operators:
        fig,ax=plt.subplots(figsize=(8,4))
        errors=[float(r['max_relative_L2']) for r in operators]
        ax.hist(errors,bins=35);ax.axvline(.03,color='red',linestyle='--',label='Registered relative-L2 limit')
        ax.set(xlabel='Relative L2 error',ylabel='Operator comparisons',title=f'I. LIVE exact BF16 KV. {len(errors)} paired operator comparisons.')
        ax.legend();save(fig,'I_numerical_accuracy')
        fig,ax=plt.subplots(figsize=(8,4))
        for world in (2,4,8):
            points=[]
            for d in range(4):
                rs=[r for r in operators if int(r['P'])==world and int(r['depth'])==d and int(r['B'])==1 and r['placement']=='balanced' and int(r['L'])==32769 and r.get('cache_pressure')!='True']
                if rs:points.append((d+1,np.mean([float(r['speedup_wall']) for r in rs])))
            ax.plot([d for d,t in points],[t for d,t in points],'o-',label=f'Generic CP P{world}')
        ax.set(xlabel='Recurrent loop',ylabel='Local / distributed wall latency',title='F. Same generic CP at each depth. Loop-aware method NOT_RUN.')
        ax.legend();save(fig,'F_loop_aware_vs_generic')
    pairs=list(csv.DictReader((ROOT/'paired_comparisons.csv').open()))
    if pairs:
        fig,ax=plt.subplots(figsize=(8,4));groups=defaultdict(list)
        for r in pairs:groups[(int(r['P']),int(r['N']),r['lane'])].append(float(r['speedup']))
        labels=[];means=[]
        for key,values in sorted(groups.items()):labels.append(f'P{key[0]}/B{key[1]}\n{key[2]} n={len(values)}');means.append(np.mean(values))
        ax.bar(range(len(means)),means);ax.axhline(1,color='black',linestyle='--');ax.set_xticks(range(len(labels)),labels,fontsize=7)
        ax.set(ylabel='Native / layer-major mean ITL',title='D. LIVE exact generic layout. Independent-start pairs, bounded generation.')
        save(fig,'D_single_gpu_layout_and_tiering')
    fig,ax=plt.subplots(figsize=(8,4))
    peerfiles=list((ROOT/'distributed').glob('scaling_v2_P2_s*/rank0_peer_transfers.json'))
    if peerfiles:
        data=json.loads(peerfiles[0].read_text());ax.plot([r['bytes']/2**20 for r in data],[r['median_ms'] for r in data],'o-',label='Measured NCCL P2P + barriers')
    ax.set(xlabel='Original BF16 K+V transfer, MiB',ylabel='Wall latency, ms',title='C. LIVE movement cost. No full-model HBM roofline claim.')
    ax.legend();save(fig,'C_attn_roofline_and_shard_cost')
    strong=list(csv.DictReader((ROOT/'best_local_vs_CP.csv').open()))
    fig,ax=plt.subplots(figsize=(8,4))
    for world in (2,4,8):
        rs=[r for r in strong if int(r['P'])==world and r['B']=='1' and r['cache_pressure']=='True']
        rs.sort(key=lambda r:int(r['L']))
        values=np.array([float(r['speedup_over_best_local']) for r in rs])
        lower=np.array([float(r['CI_low']) for r in rs]);upper=np.array([float(r['CI_high']) for r in rs])
        ax.errorbar([int(r['L'])/1024 for r in rs],values,yerr=[values-lower,upper-values],fmt='o-',capsize=3,label=f'CP{world}')
    ax.axhline(1,color='black',linestyle='--',label='Best measured local S1/2/4/8/16')
    ax.set(xlabel='Visible context, Ki tokens',ylabel='Best local / distributed wall latency',
        title='J. Strong B1 baseline. LIVE operator, 128 MiB pressure, 3 local + 3 group starts.')
    ax.legend(fontsize=8);save(fig,'J_strong_local_vs_distributed')

if __name__=='__main__':main()
