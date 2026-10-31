"""Plot measured capacity, finite pressure, and host request intervals."""
import csv
import statistics
from .common import *
from .prepare import GRID


def read(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []


def numeric(value):
    try:return float(value)
    except (TypeError,ValueError):return None


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    setup();plt.rcParams.update({'font.size':9})
    def save(fig,name):
        fig.tight_layout();fig.savefig(ROOT/'figures'/(name+'.png'),dpi=150)
        fig.savefig(ROOT/'figures'/(name+'.svg'));plt.close(fig)
    rows=read('capacity_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for ax,model,pool in zip(axes,('ouro','huginn','qwen'),('8192','18432','native_vLLM')):
        selected=[r for r in rows if r['model']==model and r['cell']=='G2' and r['output_tokens']=='128' and
            (r['chunk']=='256' and r['KV_blocks']==pool if model!='qwen' else True)]
        selected=[r for r in selected if r['evidence']!='MEASURED_NO_CUDA_EVENT_INSTRUMENTATION']
        selected.sort(key=lambda r:int(r['batch']))
        for field,label in (('mu_P_finite_mean','P + first sample'),('mu_D_post_first_finite_mean','D after first sample')):
            values=[(int(r['batch']),numeric(r[field])) for r in selected if numeric(r[field]) is not None]
            if values:ax.plot(*zip(*values),marker='o',label=label)
        ax.set(title=model+'; 3072/128',xlabel='Active requests',ylabel='Finite service rate (requests/s)')
        ax.legend(fontsize=7);ax.grid(alpha=.25)
        if selected:ax.text(.02,.98,'cohorts/point: '+','.join(str(r['repeats']) for r in selected),transform=ax.transAxes,va='top',fontsize=7)
    fig.suptitle('Post-first-sample boundary; measured finite cohorts; not sustainable capacity',fontsize=10)
    save(fig,'independent_capacity_curves')
    screening=read('bottleneck_screening.csv')
    if screening:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        labels={'D_KV_ADMISSION_LIMITED':0,'D_SERVICE_PRESSURE_GPU_HOST_CAUSE_PENDING_PROFILE':1,
            'FINITE_P_PRESSURE_D_CAPACITY_CANDIDATE_PENDING_KERNEL_PROFILE':2,
            'INCONCLUSIVE_PRESSURE_WITHOUT_VERIFIED_D_SPARE':3,'BALANCED_OR_LOW_LOAD_FINITE_COHORT':4}
        for ax,model in zip(axes,('ouro','huginn')):
            inputs=sorted({GRID[k][0] for k in ('G1','G2','G3','G4','G5','G6','C1','C2')})
            outputs=sorted({GRID[k][1] for k in ('G1','G2','G3','G4','G5','G6','C1','C2')})
            image=np.full((len(inputs),len(outputs)),np.nan)
            for row in screening:
                if row['model']!=model or '_screen_' not in row['name']:continue
                length,cap=GRID[row['cell']];i=inputs.index(length);j=outputs.index(cap)
                image[i,j]=labels[row['category']]
                ax.text(j,i,row['cell']+'\n'+row['category'].split('_')[0]+'\n'+f'{float(row["realized_arrival_rate"]):.2f}/s',ha='center',va='center',fontsize=7)
            cmap=plt.get_cmap('Set2').copy();cmap.set_bad('lightgray')
            ax.imshow(image,vmin=0,vmax=4,cmap=cmap,aspect='auto')
            ax.set_xticks(range(len(outputs)),outputs);ax.set_yticks(range(len(inputs)),inputs)
            ax.set(title=model+'; C256, default pool, N24',xlabel='Controlled output tokens',ylabel='Input tokens')
        fig.suptitle('Finite pressure screening; gray cells are unmeasured; categories require profile qualification',fontsize=10)
        save(fig,'bottleneck_phase_diagram')
    serving=[r for r in read('serving_runs.csv') if r.get('eligible_for_serving_comparison')=='True']
    if serving:
        fig,axes=plt.subplots(1,3,figsize=(13,3.7))
        for ax,model in zip(axes,('ouro','huginn','qwen')):
            for kind,label in (('_rate','G2, default C256'),('_expanded_','Expanded generic engine')):
                selected=[r for r in serving if r['model']==model and kind in r['name'] and r['independent_start']=='0']
                selected.sort(key=lambda r:float(r['arrival_rate']))
                if selected:ax.plot([float(r['arrival_rate']) for r in selected],[100*float(r['joint_SLO_attainment']) for r in selected],marker='o',label=label)
            if model=='qwen':
                selected=[r for r in serving if r['model']==model]
                ax.scatter([float(r['arrival_rate']) for r in selected],[100*float(r['joint_SLO_attainment']) for r in selected],label='Native vLLM cells')
            ax.axhline(90,color='black',ls='--',lw=.8);ax.set_ylim(-3,103)
            ax.set(title=model,xlabel='Target Poisson arrival rate (requests/s)',ylabel='Joint TTFT + average TPOT attainment (%)')
            ax.grid(alpha=.25)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        fig.suptitle('Measured finite cohorts; model-specific SLOs frozen before held-out Serving',fontsize=10)
        save(fig,'slo_attainment_vs_arrival_rate')
    shares=read('e2e_component_shares_aggregate.csv')
    names=[f'phase_b_{m}_g2_screen_start0' for m in ('ouro','huginn')]
    if any(r['name'] in names for r in shares):
        fig,ax=plt.subplots(figsize=(9,3.8));bottom=[0.,0.]
        components=list(dict.fromkeys(r['component'] for r in shares if r['name'] in names))
        for component in components:
            values=[next((float(r['share_percent']) for r in shares if r['name']==name and r['component']==component),0.) for name in names]
            ax.bar(range(2),values,bottom=bottom,label=component.replace('_',' '))
            bottom=[a+b for a,b in zip(bottom,values)]
        ax.set_xticks(range(2),('Ouro','Huginn'));ax.set_ylabel('Share of summed request E2E (%)');ax.set_ylim(0,101)
        ax.set_title('G2 N24; exclusive host intervals, not GPU kernel or DMA shares')
        ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=7)
        save(fig,'host_e2e_component_shares')
    profiles=read('profiled_phase_kernel_busy.csv')
    if profiles:
        selected=[r for r in profiles if r['phase'] in ('bc_phase_prefill','bc_phase_decode')]
        fig,ax=plt.subplots(figsize=(8,3.5))
        ax.bar(range(len(selected)),[100*float(r['kernel_busy_fraction_of_profiled_host_phase']) for r in selected])
        ax.set_xticks(range(len(selected)),[r['run'].split('_')[1]+' '+r['phase'].removeprefix('bc_phase_') for r in selected])
        ax.set(ylabel='Recorded kernel-busy union / profiled phase (%)',ylim=(0,100),title='Separate CUPTI runs; heavy tracing changes latency; not clean Serving utilization')
        save(fig,'profiled_kernel_busy_not_clean_latency')
    print('FIGURES_UPDATED')


if __name__=='__main__':main()
