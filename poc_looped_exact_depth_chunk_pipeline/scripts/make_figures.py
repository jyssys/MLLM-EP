"""Plot measured observations and explicitly labeled non-executable forecasts."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_looped_exact_depth_chunk_pipeline.contract import *


def save(fig,name):
    directory=ROOT/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    fig,ax=plt.subplots(figsize=(11,4))
    for r in range(4):
        for j in range(8):
            ax.scatter(j,r,s=120,color='tab:blue');ax.text(j,r+.15,f'{r+1},{j+1}',ha='center',fontsize=8)
            if j:ax.annotate('',(j-.1,r),(j-.9,r),arrowprops=dict(arrowstyle='->',color='tab:orange'))
            if r:ax.annotate('',(j,r-.1),(j,r-.9),arrowprops=dict(arrowstyle='->',color='tab:green'))
    ax.set(xlabel='Contiguous causal chunk j',ylabel='Recurrent depth r',xticks=range(8),xticklabels=range(1,9),yticks=range(4),yticklabels=range(1,5))
    ax.invert_yaxis();ax.set_title('DEPENDENCY_ONLY: orange = same-depth prefix KV; green = preceding-depth hidden\nExactly the same graph as tied-unrolled generic PP')
    fig.tight_layout();save(fig,'exact_depth_chunk_dag')
    quality_path=ROOT/'results/native_quality_r4.json'
    if quality_path.exists():
        quality=read(quality_path);lanes=list(quality['final_lanes'].items())
        if lanes:
            fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(len(lanes))
            axes[0].bar(x,[100*r['score'] for _,r in lanes],color='tab:blue')
            means=np.array([100*r['score'] for _,r in lanes])
            axes[0].errorbar(x,means,yerr=[means-[100*r['problem_bootstrap_95CI'][0] for _,r in lanes],
                [100*r['problem_bootstrap_95CI'][1] for _,r in lanes]-means],fmt='none',ecolor='k',capsize=3)
            axes[1].bar(x,[100*r['natural_complete']/r['total'] for _,r in lanes],color='tab:orange')
            labels=[f'{key}\ncap {row["cap"]}, n={row["total"]}' for key,row in lanes]
            for ax,title in zip(axes,['Completed exact-answer accuracy (%)','Natural completion (%)']):
                ax.set(xticks=x,xticklabels=labels,title=title,ylim=(0,105));ax.tick_params(axis='x',labelsize=8)
            fig.suptitle('MEASURED_NATURAL_QUALITY. Capped generations count as incomplete, not correct.');fig.tight_layout();save(fig,'native_quality_r4')
    summary_path=ROOT/'results/baseline_summary.json'
    if summary_path.exists():
        data=[r for r in read(summary_path) if r['TP']==4 and r['R']==4]
        if data:
            fig,axes=plt.subplots(1,3,figsize=(13,4))
            for B in sorted({r['chunk'] for r in data}):
                rows=sorted([r for r in data if r['chunk']==B],key=lambda r:r['P'])
                for ax,metric in zip(axes,['prefill_ms','TPOT_ms','E2E_ms']):
                    y=np.array([r[metric]['mean'] for r in rows]);low=np.array([r[metric]['low'] for r in rows]);hi=np.array([r[metric]['high'] for r in rows])
                    ax.errorbar([r['P'] for r in rows],y,yerr=[y-low,hi-y],marker='o',capsize=3,label='FULL' if not B else f'CP{B}')
                    ax.set(xlabel='Prompt tokens',ylabel='ms',title=metric);ax.legend(fontsize=7)
            fig.suptitle('MEASURED_CLEAN: native R4 TP4, B1, 64 forced outputs. Not accuracy.');fig.tight_layout();save(fig,'native_baselines')
    own_path=ROOT/'results/physics/TP4_R4_chunk0_v1/ownership.json'
    ref_path=ROOT/'results/reference_components_v1/reference_component_correctness.json'
    if own_path.exists() and ref_path.exists():
        native=read(own_path);reference=read(ref_path)
        native_weight=np.mean([r['unique_parameter_storage_bytes'] for r in native])/2**30
        full_weight=reference['loaded_unique_parameter_bytes']/2**30
        logical_KV=(16384+63)*192*16*128*2*2/4/2**30
        fig,ax=plt.subplots(figsize=(8,4));weights=[native_weight,full_weight/2,full_weight]
        ax.bar(range(3),weights,label='Weights: loaded tensor accounting / projected partition')
        ax.bar(range(3),[logical_KV]*3,bottom=weights,label='Logical 16K written KV, not allocated cache pool')
        ax.set(xticks=range(3),xticklabels=['Native TP4','PP2×TP2 ownership','PP4 ownership'],ylabel='GiB / GPU')
        ax.legend(fontsize=8);ax.set_title('STRUCTURAL_TENSOR_ACCOUNTING. PP runtime and workspace not allocated.');fig.tight_layout();save(fig,'memory_ownership')
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for row in reference['component_results']:
            for ax,key in zip(axes,['K_relative_L2','V_relative_L2']):
                ax.plot([v['slot'] for v in row['KV']],[max(v[key],1e-7) for v in row['KV']],label=f'{row["request_id"]}, B{row["chunk"]}')
                ax.set(xlabel='Logical KV slot (48 per depth)',ylabel='Relative L2',title=key,yscale='log')
        for ax in axes:
            ax.axhline(reference['tolerance']['KV_relative_L2'],color='k',linestyle='--',label='Frozen limit')
            for boundary in [48,96,144]:ax.axvline(boundary,color='grey',alpha=.3)
            ax.legend(fontsize=7)
        fig.suptitle('MEASURED_COMPONENT_CORRECTNESS: full versus chunked HF BF16 reference. Not speedup.')
        fig.tight_layout();save(fig,'reference_KV_relative_error')
    model_path=ROOT/'results/calibrated_PP_requests.csv'
    if not model_path.exists():return
    with model_path.open() as stream:models=list(csv.DictReader(stream))
    if not models:return
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for P in sorted({int(r['P']) for r in models}):
        for TP in [1,2]:
            rows=[r for r in models if int(r['P'])==P and int(r['TP_per_stage'])==TP]
            groups={B:[r for r in rows if int(r['chunk'])==B] for B in sorted({int(r['chunk']) for r in rows})}
            for ax,field in zip(axes,['modeled_prefill_reduction_percent','optimistic_E2E_reduction_percent']):
                ax.plot(list(groups),[np.mean([float(r[field]) for r in records]) for records in groups.values()],'-o',label=f'P{P} PP{4//TP} TP{TP}')
                ax.axhline(0,color='k',linestyle='--');ax.set(xlabel='Chunk rows',ylabel='Reduction vs native TP4 (%)',title=field);ax.legend(fontsize=6)
    fig.suptitle('CALIBRATED, not measured pipeline speedup. E2E grants free native-decode cache transition.');fig.tight_layout();save(fig,'chunk_size_sensitivity')
    chosen=min(models,key=lambda r:abs(int(r['P'])-16384)+abs(int(r['chunk'])-2048))
    with (ROOT/'results/calibrated_timeline.csv').open() as stream:
        tape=[r for r in csv.DictReader(stream) if r['P']==chosen['P'] and r['TP_per_stage']==chosen['TP_per_stage'] and r['chunk_rows']==chosen['chunk'] and r['request_id']==chosen['request_id']]
    fig,axes=plt.subplots(4,1,figsize=(12,5),sharex=True);TP=int(chosen['TP_per_stage'])
    for rank,ax in enumerate(axes):
        selected=[r for r in tape if int(r['stage'])==rank//TP]
        ax.broken_barh([(float(r['start_ms']),float(r['end_ms'])-float(r['start_ms'])) for r in selected],(0,1),facecolors='tab:blue')
        ax.set(yticks=[],ylabel=f'GPU{4+rank}')
    axes[-1].set_xlabel('Modeled ms');fig.suptitle('CALIBRATED component schedule. Not an executed physical pipeline timeline.');fig.tight_layout();save(fig,'four_gpu_gantt')


if __name__=='__main__':main()
