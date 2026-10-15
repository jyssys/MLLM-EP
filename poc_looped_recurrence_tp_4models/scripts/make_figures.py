"""Render measured matrices and separately labeled diagnostic service charts."""
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_looped_recurrence_tp_4models.contract import ROOT
from poc_looped_recurrence_tp_4models.analysis import read,bootstrap


def load(name):
    path=ROOT/'results'/name
    if not path.exists():return []
    with path.open() as stream:return list(csv.DictReader(stream))


def save(fig,name):
    target=ROOT/'figures';target.mkdir(exist_ok=True)
    fig.savefig(target/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    summary=load('config_summary.csv');pairs=load('paired_comparisons.csv')
    index={(r['model'],r['family'],int(r['R']),int(r['TP']),int(r['prefix'])):r for r in summary}
    plt.rcParams.update({'font.size':9,'axes.grid':False})
    for prefix in [1024,16384,32704]:
        fig,axes=plt.subplots(1,4,figsize=(14,3.2))
        for ax,metric,label in zip(axes,['prefill_ms','TPOT_ms','E2E_ms','efficiency'],['Scheduled-to-first ms','Cached TPOT ms','Client E2E ms','E2E parallel efficiency']):
            values=np.full((3,3),np.nan)
            for i,R in enumerate([1,2,4]):
                for j,TP in enumerate([1,2,4]):
                    row=index.get(('ouro','native',R,TP,prefix))
                    if row:
                        if metric=='efficiency':
                            paired=next((r for r in pairs if r['model']=='ouro' and r['family']=='native' and
                                int(r['R'])==R and int(r['TP'])==TP and int(r['prefix'])==prefix and
                                r['metric']=='E2E_ms' and r['comparison']=='TP_speedup'),None)
                            if paired:values[i,j]=float(paired['mean'])/TP
                        else:values[i,j]=float(row[metric])
            ax.imshow(values,aspect='auto',cmap='viridis');ax.set(xticks=[0,1,2],xticklabels=[1,2,4],yticks=[0,1,2],yticklabels=[1,2,4],xlabel='TP',ylabel='R',title=label)
            for i,j in np.ndindex(values.shape):
                if np.isfinite(values[i,j]):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white')
        fig.suptitle(f'Native Ouro: P={prefix}, 64 identical forced outputs, n={3 if prefix==32704 else 8} requests');fig.tight_layout();save(fig,f'ouro_matrix_P{prefix}')
    for prefix in [1024,16384]:
        fig,axes=plt.subplots(1,3,figsize=(12,3.7))
        for ax,metric in zip(axes,['prefill_ms','TPOT_ms','E2E_ms']):
            for model,family,R,label in [('ouro','native',1,'Ouro R1'),('ouro','native',2,'Ouro R2'),('ouro','native',4,'Ouro R4'),('qwen3','native',1,'Qwen3-1.7B'),('qwen25','native',1,'Qwen2.5-3B')]:
                rows=sorted([r for r in pairs if r['model']==model and r['family']==family and int(r['R'])==R and int(r['prefix'])==prefix and r['metric']==metric and r['comparison']=='TP_speedup'],key=lambda r:int(r['TP']))
                if not rows:continue
                y=np.array([float(r['mean']) for r in rows]);low=np.array([float(r['low']) for r in rows]);high=np.array([float(r['high']) for r in rows])
                ax.errorbar([int(r['TP']) for r in rows],y,yerr=[y-low,high-y],marker='o',capsize=2,label=label)
            ax.plot([1,2,4],[1,2,4],'k--',alpha=.25,label='Ideal (not headroom)');ax.set(xlabel='TP',ylabel='Within-model TP speedup',title=metric,xticks=[1,2,4]);ax.legend(fontsize=7)
        fig.suptitle(f'Normalized scaling, P={prefix}. Paired request bootstrap 95% CI.');fig.tight_layout();save(fig,f'within_model_scaling_P{prefix}')
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for i,prefix in enumerate([1024,16384]):
        for ax,metric in zip(axes[i],['prefill_ms','TPOT_ms','E2E_ms']):
            for R in [1,2,4]:
                rows=sorted([r for r in pairs if r['comparison']=='NativeVsUnrolledRatio' and r['metric']==metric and int(r['R'])==R and int(r['prefix'])==prefix],key=lambda r:int(r['TP']))
                if rows:
                    y=np.array([float(r['mean']) for r in rows]);ax.errorbar([int(r['TP']) for r in rows],y,yerr=[y-[float(r['low']) for r in rows],[float(r['high']) for r in rows]-y],marker='o',capsize=3,label=f'R{R}')
            ax.axhline(1,color='k',linestyle='--');ax.set(title=f'P{prefix}: {metric}',xlabel='TP',ylabel='Native / Tied',xticks=[1,2,4]);ax.legend()
            lo,hi=ax.get_ylim();ax.set_ylim(min(.9,lo),max(1.1,hi))
    fig.suptitle('Same-storage timing contrast. Selected probes pass; a natural TP4 discrepancy remains.');fig.tight_layout();save(fig,'native_vs_tied')
    owners=load('physical_ownership.csv');fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    for model,R,label in [('ouro',4,'Ouro R4'),('qwen3',1,'Qwen3'),('qwen25',1,'Qwen2.5')]:
        rows=sorted([r for r in owners if r['model']==model and r['family']=='native' and int(r['R'])==R and int(r['rank'])==0],key=lambda r:int(r['TP']))
        if not rows:continue
        axes[0].plot([int(r['TP']) for r in rows],[int(r['KV_heads_local']) for r in rows],'-o',label=label)
        axes[1].plot([int(r['TP']) for r in rows],[int(r['cache_slots'])*16384*2*int(r['KV_heads_local'])*128*2/2**30 for r in rows],'-o',label=label)
    axes[0].set(title='Actual local KV heads',xlabel='TP',ylabel='Heads per rank',xticks=[1,2,4]);axes[1].set(title='Logical written 16K KV / rank (not allocated pool)',xlabel='TP',ylabel='GiB',xticks=[1,2,4])
    for ax in axes:ax.legend()
    fig.suptitle('Qwen2.5 TP4 replicates two global KV heads across four ranks');fig.tight_layout();save(fig,'kv_ownership')
    rows=load('forward_kernel_services.csv');fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,phase in zip(axes,['PREFILL','CACHED_DECODE']):
        for cat in ['ATTENTION_WITH_KV_READ','TP_COLLECTIVE','QKV_FFN_OUTPUT_GEMM']:
            x=[];y=[];labels=[]
            for model,R in [('ouro',1),('ouro',2),('ouro',4),('qwen3',1),('qwen25',1)]:
                chosen=[r for r in rows if r['model']==model and r['family']=='native' and int(r['R'])==R and int(r['TP'])==4 and int(r['rank'])==0 and r['phase']==phase and r['category']==cat and r['timing_validity']!='UNVALIDATED']
                x.append(len(x));y.append(np.mean([float(r['service_ms_per_layer']) for r in chosen]) if chosen else np.nan);labels.append(f'{model} R{R}')
            ax.plot(x,y,'-o',label=cat)
        ax.set(title=phase,xticks=x,xticklabels=labels,ylabel='Rank0 GPU service ms / logical layer');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=7)
    fig.suptitle('Diagnostic service only. Not critical-path shares. Invalid observer strata omitted.');fig.tight_layout();save(fig,'layer_normalized_service')
    attention=load('layer_attention_service.csv');fig,axes=plt.subplots(3,1,figsize=(12,6))
    for ax,(model,R) in zip(axes,[('ouro',4),('qwen3',1),('qwen25',1)]):
        chosen=[r for r in attention if r['model']==model and r['family']=='native' and int(r['R'])==R and int(r['TP'])==4 and int(r['rank'])==0 and int(r['forward'])==1]
        if chosen:ax.bar([int(r['logical_layer']) for r in chosen],[float(r['attention_service_ms']) for r in chosen],width=1)
        if model=='ouro':
            for boundary in [48,96,144]:ax.axvline(boundary,color='k',alpha=.3,linestyle='--')
        ax.set(title=f'{model} R{R}: first cached forward, TP4 rank0',ylabel='Attention service ms',xlabel='Logical layer')
    fig.suptitle('PARTIAL critical-path diagnostic: fused attention service, not complete layer latency');fig.tight_layout();save(fig,'layer_attention_diagnostic')
    outer=load('outer_events.csv');fig,axes=plt.subplots(4,1,figsize=(13,7))
    for rank,ax in enumerate(axes):
        selected=[r for r in outer if r['model']=='ouro' and r['family']=='native' and int(r['R'])==4 and int(r['TP'])==4 and int(r['rank'])==rank]
        selected=[r for r in selected if r['category']=='execute_model' and r['scheduled_rows']=='1'][:5]
        if selected:
            offset=float(selected[0]['start_ms'])
            ax.broken_barh([(float(r['start_ms'])-offset,float(r['duration_ms'])) for r in selected],(0,1),facecolors='tab:blue')
        ax.set(ylabel=f'GPU{4+rank}',yticks=[],xlabel='Local elapsed ms from own first cached forward')
    fig.suptitle('Actual outer events. Per-rank local origins. NOT globally aligned waiting evidence.');fig.tight_layout();save(fig,'outer_timeline_partial')
    requests=load('all_config_per_request.csv');idx={(r['family'],int(r['TP']),int(r['prefix']),r['request_id']):r for r in requests if r['model']=='ouro' and int(r['R'])==4}
    fig,axes=plt.subplots(1,2,figsize=(12,4));ax=axes[0]
    for prefix,offset in [(1024,-.08),(16384,.08)]:
        estimates=[]
        for TP in [1,2,4]:
            values=[]
            for key,row in idx.items():
                if key[:3]!=('native',TP,prefix):continue
                other=idx.get(('tied',TP,prefix,key[3]))
                if other:values.append((1-float(other['E2E_ms'])/float(row['E2E_ms']))*100)
            estimates.append(bootstrap(values))
        valid=[(i,s) for i,s in enumerate(estimates) if s['mean'] is not None]
        if valid:
            y=np.array([s['mean'] for _,s in valid]);ax.errorbar([i+offset for i,_ in valid],y,
                yerr=[y-[s['low'] for _,s in valid],[s['high'] for _,s in valid]-y],marker='o',capsize=3,label=f'P{prefix}')
    ax.axhline(0,color='k',linestyle='--');ax.set(xticks=[0,1,2],xticklabels=[1,2,4],xlabel='TP',ylabel='Observed tied E2E reduction vs native (%)',title='Measured contrast, NOT removable headroom');ax.legend()
    lo,hi=ax.get_ylim();ax.set_ylim(min(-10,lo),max(10,hi))
    axes[1].axis('off');axes[1].text(0,.95,'Physically targetable recurrence cost',weight='bold',va='top')
    axes[1].text(0,.8,'Python recurrence: already captured in GPU replay\nLayer reductions: ordinary TP work\nLoop KV slots: preserved by both exact arms\nCross-rank pure waiting: UNRESOLVED\n\nRemovable loop-specific ms: NOT IDENTIFIED\nFramework E2E gain: NOT ESTIMATED',va='top',linespacing=1.6)
    fig.tight_layout();save(fig,'headroom_and_confounds')
    execution=load('runtime_execution_counts.csv');signatures=load('graph_kernel_signatures.csv');fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for family,offset in [('native',-.15),('tied',.15)]:
        x=[];replays=[];counts=[]
        for i,TP in enumerate([1,2,4]):
            row=next((r for r in execution if r['model']=='ouro' and r['family']==family and int(r['R'])==4 and int(r['TP'])==TP and int(r['rank'])==0),None)
            signature=next((r for r in signatures if r['model']=='ouro' and r['family']==family and int(r['R'])==4 and int(r['TP'])==TP and int(r['rank'])==0 and int(r['forward'])==1),None)
            if row and signature:x.append(i+offset);replays.append(int(row['graph_replays']));counts.append(int(signature['kernel_count']))
        axes[0].bar(x,replays,width=.3,label=family);axes[1].bar(x,counts,width=.3,label=family)
    for ax in axes:ax.set(xticks=[0,1,2],xticklabels=[1,2,4],xlabel='TP');ax.legend()
    axes[0].set(title='Actual graph replays / 64-output request',ylabel='Replay count');axes[1].set(title='First cached-forward kernels (rank0)',ylabel='Physical kernel count')
    fig.suptitle('R4 decisive-control schedule. Raw node audit is a separate retained-graph lane.');fig.tight_layout();save(fig,'control_schedule_counts')


if __name__=='__main__':main()
