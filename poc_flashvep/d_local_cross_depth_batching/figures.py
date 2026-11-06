"""Plot measured data and explicit missing-evidence panels."""
import csv
import json
import math
from .common import ROOT, write


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.dpi':150,'savefig.bbox':'tight'})
    manifest=[]
    def save(fig,name,scope,sources):
        fig.savefig(ROOT/'figures'/(name+'.png'));fig.savefig(ROOT/'figures'/(name+'.svg'))
        plt.close(fig);manifest.append(dict(name=name,PNG=f'figures/{name}.png',SVG=f'figures/{name}.svg',
            scope=scope,sources=sources))
    def missing(axis,text):
        axis.set_facecolor('#f3f3f3');axis.text(.5,.5,text,ha='center',va='center',transform=axis.transAxes,wrap=True)
        axis.set_xticks([]);axis.set_yticks([])
    models=('ouro','huginn')
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())
    runs=list(csv.DictReader((ROOT/'serving_runs.csv').open()))
    # Only qualified activity windows are eligible for the primary timeline.
    fig,axes=plt.subplots(2,2,figsize=(11,6.5))
    profiles=[]
    for index,model in enumerate(models):
        path=ROOT/'raw'/f'local_{model}_native_thin_trace128/dlocal_profile_qualification.json'
        if not path.exists() or not json.loads(path.read_text()).get('qualified'):
            missing(axes[index,0],f'{model.title()}: NOT_MEASURED\nNo qualified kernel timeline')
            row=next((r for r in runs if r['name']==f'local_{model}_native_n128_s2026110701_off'),None)
            if row:
                limit=slos[f'{model}_L{4096 if model=="ouro" else 2048}']['ITL_ms']
                axes[index,1].bar([0,1],[float(row['ITL_p99_ms']),float(row['max_ITL_ms'])],color='#777')
                axes[index,1].axhline(limit,color='#d1495b',ls='--',label='Frozen ITL p99 limit')
                axes[index,1].set_xticks([0,1],['Native ITL p99','Native maximum ITL'])
                axes[index,1].set_ylabel('Marker-free decode latency (ms)');axes[index,1].legend(fontsize=8)
                axes[index,1].set_title('Statistical margin is not an insertion budget',fontsize=9)
            continue
        q=json.loads(path.read_text());folder=path.parent
        gaps=json.loads((folder/'kernel_gap_samples.json').read_text())['gaps_ms']
        positive=sorted(g for g in gaps if g>0)
        axes[index,0].plot(positive,np.arange(1,len(positive)+1)/len(positive),color='#247ba0')
        axes[index,0].set_xscale('log');axes[index,0].set_xlabel('Measured kernel-free interval (ms)')
        axes[index,0].set_ylabel('Empirical CDF');axes[index,0].set_title(f'{model.title()}: kernel union {100*q["kernel_union_fraction"]:.1f}%')
        a=q['local_clean'];b=q['local_profile']
        labels=('ITL p99','Maximum ITL');x=np.arange(2)
        axes[index,1].bar(x-.17,[a['ITL_p99_ms'],a['max_ITL_ms']],.34,label='Marker-free control',color='#777')
        axes[index,1].bar(x+.17,[b['ITL_p99_ms'],b['max_ITL_ms']],.34,label='Qualified trace',color='#247ba0')
        axes[index,1].set_xticks(x,labels);axes[index,1].set_ylabel('Local decode interval (ms)');axes[index,1].legend(fontsize=8)
        profiles.append(str(path.relative_to(ROOT)))
    fig.suptitle('Figure 1. Kernel gaps are not guaranteed decode-safe slack')
    fig.text(.5,.01,'Kernel gaps can contain DMA. A valid gap distribution does not prove that a full prefill slice fits.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95))
    save(fig,'figure_1_decode_safe_windows','Qualified activity-only observations. No simulated idle intervals.',profiles)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5));memory_sources=[]
    stacks=[];labels=[]
    for model in models:
        folder=ROOT/'raw'/f'local_{model}_native_n128_s2026110701_off'
        for role in ('prefill','decode'):
            matches=[p for p in folder.glob('memory_inventory_close_*.json') if json.loads(p.read_text())['role']==role]
            if not matches:continue
            p=matches[0];v=json.loads(p.read_text());memory_sources.append(str(p.relative_to(ROOT)))
            stacks.append([v['weight_and_model_buffer_storage_bytes']/2**30,v['KV_storage_bytes']/2**30,
                (v['exposed_runner_workspace_storage_bytes']+v['exposed_NIXL_hidden_staging_storage_bytes'])/2**30,
                (v['exposed_graph_tensor_storage_bytes']+v['opaque_or_unattributed_allocator_bytes'])/2**30])
            labels.append(model.title()+' '+('P' if role=='prefill' else 'D'))
    if stacks:
        bottom=np.zeros(len(stacks));colors=('#50514f','#247ba0','#70c1b3','#f3b61f')
        for j,label in enumerate(('Weights + model buffers','KV pool','Exposed workspace + staging','Graph / allocator residual')):
            values=np.array(stacks)[:,j];axes[0].bar(labels,values,bottom=bottom,label=label,color=colors[j]);bottom+=values
        axes[0].set_ylabel('Unique allocated storage (GiB)');axes[0].tick_params(axis='x',rotation=15);axes[0].legend(fontsize=8)
        inventory=list(csv.DictReader((ROOT/'gpu_memory_inventory.csv').open()))
        capacity=next((float(v['physical_HBM_bytes'])/2**30 for v in inventory if v.get('physical_HBM_bytes')),None)
        if capacity:
            axes[0].axhline(capacity,color='#b22222',ls='--',lw=1)
            axes[0].text(.98,.98,f'NVML physical HBM: {capacity:.3f} GiB',transform=axes[0].transAxes,
                ha='right',va='top',fontsize=8)
            axes[0].set_ylim(0,capacity*1.1)
    else:missing(axes[0],'NOT_MEASURED: HBM inventory')
    counts=[]
    for model in models:
        paths=sorted((ROOT/'raw').glob(f'qual32_{model}_full*/local_qualification.json'))
        if not paths:
            counts.append(None)
        else:
            q=json.loads(paths[-1].read_text());counts.append(q['count'] if q['passed'] else None)
    for i,value in enumerate(counts):
        if value is None:axes[1].text(i,.5,'N32 not qualified',ha='center',rotation=90)
        else:axes[1].bar(i,value,color='#70c1b3')
    axes[1].set_xticks(range(2),('Ouro','Huginn'));axes[1].set_ylim(0,36)
    axes[1].set_ylabel('Exact D-local full-prompt requests');axes[1].set_title('Correctness, not clean throughput')
    fig.suptitle('Figure 2. Whole-request D ownership uses the existing KV pool')
    fig.text(.5,.01,'Allocator totals overlap these components. Opaque graph memory is not fully attributed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_2_d_local_ownership_and_hbm','Measured storage inventory and N32 exactness. No added KV pool.',memory_sources)

    fig,axes=plt.subplots(1,2,figsize=(11,4.7));micro_sources=[]
    for index,model in enumerate(models):
        from .micro_analysis import eligible_cells
        folders=[ROOT/'raw'/f'micro_{model}_strict_grid_v5']
        items=[v for folder in folders for v in eligible_cells(folder) if v.get('control_version',0)>=5 and v['hidden_exact'] and v['KV_exact']]
        if not items:missing(axes[index],model.title()+': corrected V5 timing NOT_MEASURED');continue
        micro_sources.extend(str((p/'full_core_micro.json').relative_to(ROOT)) for p in folders)
        series={}
        for item in items:
            if item['decode_batch']!=2:continue
            vals=[np.median(r['batched_ms']) for r in item.get('paired_repetitions',[])]
            if vals:series[item['chunk'],item['depth_class']]=vals
            if item['depth_class']=='cross':series[item['chunk'],'separate']=[np.median(r['separate_ms']) for r in item.get('paired_repetitions',[])]
        for label,color,shift in (('separate','#bbb',-.25),('same','#777',0),('cross','#247ba0',.25)):
            x=[];y=[];err=[]
            for j,chunk in enumerate((16,32,64,128)):
                v=series.get((chunk,label))
                if not v:continue
                x.append(j+shift);y.append(np.median(v));err.append([np.median(v)-min(v),max(v)-np.median(v)])
            if x:axes[index].bar(x,y,.23,color=color,label=label if label=='separate' else label+' depth',yerr=np.array(err).T,capsize=3)
        axes[index].set_xticks(range(4),('16','32','64','128'));axes[index].set_xlabel('Prefill query tokens + 2 decode rows')
        axes[index].set_ylabel('Core + attention enclosure (ms)');axes[index].set_title(model.title());axes[index].legend(fontsize=8)
    fig.suptitle('Figure 3. Matched same-depth versus cross-depth batching')
    fig.text(.5,.01,'Three warm repetitions in one process. Whiskers show min–max, not an independent-start confidence interval.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_3_same_vs_cross_depth_micro','Measured legal ready-state operations. Timings include graph input staging. Not online serving.',micro_sources)

    fig,axes=plt.subplots(2,4,figsize=(13,6));sources=[]
    methods=('native','chunked','paced')
    for i,model in enumerate(models):
        selected={}
        for method in methods:
            candidates=[r for r in runs if r['model']==model and r['method']==method and r['marker_on']=='False'
                and r['complete']=='True' and int(r['observed_requests'])==128
                and int(r['arrival_seed'])==2026110701
                and abs(float(r['offered_QPS'])-(3.4569045 if model=='ouro' else 1.06129395274))<1e-6
                and (method=='native' or r['name']==f'serve_{model}_{method}_c128_fixed25_v5')]
            if candidates:selected[method]=candidates[-1]
        for j,(field,title) in enumerate((('TTFT_p95_ms','TTFT p95 (ms)'),('TPOT_p95_ms','TPOT p95 (ms)'),
                ('ITL_p99_ms','ITL p99 (ms)'),('joint_SLO_goodput','Joint SLO goodput (req/s)'))):
            ax=axes[i,j]
            for k,method in enumerate(methods):
                if method not in selected:
                    ax.text(k,.5,'NOT_MEASURED',ha='center',rotation=90,transform=ax.get_xaxis_transform(),fontsize=8);continue
                r=selected[method];sources.append(r['name'])
                ax.bar(k,float(r[field]),color='#247ba0' if r['decode_screen_pass']=='True' and r['serving_output_exact']=='True' else '#ed553b')
            length=4096 if model=='ouro' else 2048
            slo=slos[f'{model}_L{length}']
            if j in (0,1,2):
                limit=(slo['TTFT_ms'],.9*slo['TPOT_ms'],slo['ITL_ms'])[j]
                ax.axhline(limit,color='#b22222',ls='--',lw=1)
            ax.set_xticks(range(3),('Native','Chunked','Loop-paced'),rotation=20);ax.set_title(model.title()+': '+title,fontsize=10)
    fig.suptitle('Figure 4. Marker-free serving; red bars fail the decode screen')
    fig.text(.5,.01,'Fixed D placement is a controlled baseline, not the official Kairos policy. Cross-depth online expansion remains gated.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_4_serving_latency_and_goodput','Finite-cohort measurements. Missing arms are not zero.',sorted(set(sources)))

    fig,axes=plt.subplots(1,2,figsize=(11,4.5));op_sources=[]
    for i,model in enumerate(models):
        name='opportunity_ouro_graph_control_v2' if model=='ouro' else 'opportunity_huginn_s1'
        path=ROOT/'raw'/name/'opportunity_micro.json'
        if not path.exists():missing(axes[i],model.title()+': NOT_MEASURED');continue
        data=[r for r in json.loads(path.read_text()) if not r['warmup'] and 'graph_after' in r]
        for j,method in enumerate(('separate','same','cross')):
            values=[r['cohort_host_ms'] for r in data if r['method']==method]
            if values:axes[i].bar(j,np.median(values),color=('#aaa','#777','#247ba0')[j])
        axes[i].set_xticks(range(3),('Separate','Same-depth','Cross-depth'),rotation=15)
        axes[i].set_ylabel('Closed-loop cohort time (ms)');axes[i].set_title(model.title())
        op_sources.append(str(path.relative_to(ROOT)))
    fig.suptitle('Figure 5. Ready-work opportunity control with common graph policy')
    fig.text(.5,.01,'Two decode streams + one whole local prompt. Three warm repetitions; not a PD online-goodput measurement.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_5_ready_work_opportunity_control','Native priorities are common. Only recurrent compatibility changes. No modeled speedup.',op_sources)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    costs=list(csv.DictReader((ROOT/'slice_costs.csv').open())) if (ROOT/'slice_costs.csv').exists() else []
    for i,model in enumerate(models):
        for q,label,color in ((4 if model=='ouro' else 32,'Full depth per slice','#777'),(1,'One loop step','#247ba0')):
            selected=[r for r in costs if r['model']==model and int(r['quantum'])==q and int(r['loops'])==q]
            if selected:
                selected.sort(key=lambda r:int(r['chunk']))
                axes[i].plot([int(r['chunk']) for r in selected],[float(r['enclosure_p95_ms']) for r in selected],'-o',label=label,color=color)
        axes[i].set_xlabel('Prompt tokens per slice');axes[i].set_ylabel('Model-stream enclosure p95 (ms)')
        axes[i].set_title(model.title());axes[i].set_xticks([16,32,64,128]);axes[i].legend(fontsize=8)
    fig.suptitle('Figure 6. Smaller execution quanta reduce a serial blocking unit')
    fig.text(.5,.01,'Measured serial slices include causal attention. They do not prove that an online deadline-safe slot exists.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_6_slice_execution_quantum','CUDA-event enclosures, not exclusive kernel time or an online speedup.',['slice_costs.csv'])

    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for i,model in enumerate(models):
        data=[r for r in runs if r['model']==model and r['method']=='native' and r['marker_on']=='False'
            and r['complete']=='True' and int(r['observed_requests'])==128 and not (r['activity_profiler']=='True')]
        data=sorted(data,key=lambda r:float(r['offered_QPS']))
        for j,(field,label) in enumerate((('TTFT_p95_ms','TTFT p95 (ms)'),('ITL_p99_ms','ITL p99 (ms)'),('completed_QPS','Completed req/s'))):
            axes[i,j].plot([float(r['offered_QPS']) for r in data],[float(r[field]) for r in data],'-o',color='#777')
            axes[i,j].set_xlabel('Nominal offered QPS');axes[i,j].set_ylabel(label);axes[i,j].set_title(model.title())
    fig.suptitle('Figure 7. Native PD finite-cohort arrival-rate controls')
    fig.text(.5,.01,'One independent start per point. The overloaded cohort is not sustainable capacity.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_7_native_load_controls','Marker-free Native PD controls. No invented generic or cross-depth curve.',['serving_runs.csv'])

    fig,axes=plt.subplots(2,1,figsize=(11,4.5));diagnostic_sources=[]
    from .profile_analysis import union
    for i,model in enumerate(models):
        folder=ROOT/'raw'/f'local_{model}_native_thin_trace128'
        path=folder/'kernel_activity_decode.csv'
        if not path.exists():missing(axes[i],model.title()+': NOT_MEASURED');continue
        capture=json.loads((folder/'activity_capture_decode.json').read_text())
        origin=capture['start']['device_clock_ns']+250_000_000
        end=origin+20_000_000
        records=list(csv.DictReader(path.open()))
        mapping=json.loads((folder/'worker_map.json').read_text())
        device=next(v['visible_device'] for v in mapping.values() if v['role']=='decode')
        for graph,label,color,y in ((True,'Graph kernels','#247ba0',1),(False,'Non-graph kernels','#777',0)):
            intervals=union([(max(origin,int(v['start_ns'])),min(end,int(v['end_ns']))) for v in records
                if int(v['device'])==device and bool(int(v['graph_node']))==graph and int(v['end_ns'])>origin and int(v['start_ns'])<end])
            axes[i].broken_barh([((a-origin)/1e6,(b-a)/1e6) for a,b in intervals],(y-.3,.6),facecolors=color)
        axes[i].set_yticks([0,1],['Non-graph kernels','Graph kernels']);axes[i].set_xlim(0,20)
        axes[i].set_xlabel('Recorded device time (ms)');axes[i].set_title(model.title()+': diagnostic D-only capture',fontsize=10)
        diagnostic_sources.append(str(path.relative_to(ROOT)))
    fig.suptitle('Figure 8. UNQUALIFIED instrumented timeline — not clean decode-safe slack',color='#b22222',fontsize=12)
    fig.text(.5,.01,'Actual recorded intervals. Local-tail interference exceeds 5%. P activity, DMA, and safe idle capacity are not shown.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_8_unqualified_diagnostic_timeline','Measured under rejected instrumentation. Never use this figure as clean kernel activity or safe slack proof.',diagnostic_sources)

    fig,axes=plt.subplots(2,2,figsize=(11,6.5));large=list(csv.DictReader((ROOT/'readiness_opportunity.csv').open()))
    source_names=[]
    for i,model in enumerate(models):
        for j,chunk in enumerate((128,248)):
            preferred=f'opportunity_{model}_c{chunk}'+('_d_device_v1' if chunk==248 else '_alignment_v1')
            data=[v for v in large if v.get('name')==preferred]
            if not data:
                missing(axes[i,j],f'{model.title()} C{chunk}: NOT_MEASURED');continue
            names=('separate','same','same_align','cross')
            for k,method in enumerate(names):
                row=next((v for v in data if v['method']==method),None)
                if row:
                    height=float(row['local_first_ms'])
                    axes[i,j].bar(k,height,color=('#bbb','#777','#70c1b3','#247ba0')[k])
                    axes[i,j].text(k,height*1.015,f'{height:.0f}',ha='center',va='bottom',fontsize=8)
            axes[i,j].margins(y=.13)
            axes[i,j].set_xticks(range(4),['Separate','Same wait','Same align','Cross'],rotation=15)
            axes[i,j].set_ylabel('Local prompt to first token (ms)')
            axes[i,j].set_title(f'{model.title()} C{chunk}, physical GPU {data[0].get("physical_GPU")}')
            source_names.append(preferred)
    fig.suptitle('Figure 9. Charge the same-depth alignment step before claiming an extra depth benefit')
    fig.text(.5,.01,'One independent start per geometry. Three warm repetitions. Standalone microbenchmark, not online PD goodput.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_9_stronger_same_depth_control','Measured larger-chunk micro controls. Same align executes one real prefill step and reports its decode interference.',source_names)

    fig,axes=plt.subplots(2,2,figsize=(11,6.5));placement_sources=[]
    from .analyze import requests
    for i,model in enumerate(models):
        name=f'serve_{model}_paced_c128_fixed25_v5';folder=ROOT/'raw'/name
        data=requests(folder)
        if not data:
            for ax in axes[i]:missing(ax,model.title()+': NOT_MEASURED')
            continue
        for local,label,color in ((False,'Native route','#777'),(True,'D-local route','#247ba0')):
            selected=[v for v in data if (v['sample_id']%4==0)==local]
            axes[i,0].scatter([v['sample_id'] for v in selected],[v['ttft_ms']/1000 for v in selected],
                s=12,label=label,color=color)
        axes[i,0].set_xlabel('Offered request index');axes[i,0].set_ylabel('Arrival-to-first-token latency (s)')
        axes[i,0].set_title(model.title()+': fixed placement, local active limit = 1');axes[i,0].legend(fontsize=8)
        credit=folder/'D_coordinator_credits.csv'
        if credit.exists():
            values=list(csv.DictReader(credit.open()))
            axes[i,1].plot([float(v['time_s']) for v in values],
                [(float(v['total_blocks'])-float(v['reserved_blocks']))/float(v['total_blocks'])*100 for v in values],
                color='#247ba0')
            axes[i,1].set_xlabel('Time from measured window start (s)')
            axes[i,1].set_ylabel('Coordinator free KV credit (%)');axes[i,1].set_ylim(-2,102)
            axes[i,1].set_title('Host credit samples, not GPU execution')
        else:missing(axes[i,1],'NOT_MEASURED: coordinator credit samples')
        placement_sources.extend([f'raw/{name}/requests.jsonl',f'raw/{name}/D_coordinator_credits.csv'])
    fig.suptitle('Figure 10. Decode priority and a single local lease can accumulate local backlog')
    fig.text(.5,.01,'A growing local TTFT is not measured GPU compute time. This fixed control is not an optimized deflection policy.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    save(fig,'figure_10_fixed_placement_backlog','Marker-free per-request latency and sampled coordinator credit. No queue-to-kernel attribution.',placement_sources)
    write(ROOT/'figure_manifest.json',manifest)


if __name__=='__main__':main()
