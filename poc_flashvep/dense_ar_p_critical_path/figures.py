"""Create figures from measured tables. Keep missing cells missing."""
from collections import defaultdict
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,START,bind,write,digest
from .critical_analysis import csv_save,records

COLORS={'NativeStrong':'#555555','NativeCommon2':'#999999','StockToken':'#009e73',
    'OriginalLayer':'#0072b2','OptimizedLayer':'#56b4e9','OriginalHybrid':'#d55e00','OptimizedHybrid':'#e69f00',
    'Layer':'#0072b2','Hybrid':'#d55e00','Layer8':'#56b4e9','HybridIso8Approx':'#e69f00',
    'NativeCommon16':'#777777','StockCommon16':'#33bb88','HybridCommon16':'#aa6633',
    'HybridIsoCommon16':'#ccaa55','HybridGCControl':'#cc79a7',
    'NativeCommon4':'#888888','StockCommon4':'#33aa88'}
STAGES=('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms','P_prepare_to_ready_ms',
    'Ready_to_D_submit_ms','D_submit_to_accept_ms','D_accept_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms')
STAGE_NAMES=('A→P submit','P submit→accept','P accept→prepare','P prepare→ready','Ready→D submit',
    'D submit→accept','D accept→prepare','D prepare→first worker','Worker→client')


def load(name):
    path=ROOT/'analysis'/name
    return json.loads(path.read_text()) if path.exists() else None


def save(fig,stem,rows,caption,manifest):
    fig.text(.01,.01,caption,fontsize=8,va='bottom',wrap=True)
    fig.tight_layout(rect=(0,.1,1,.96))
    target=ROOT/'figures'/stem
    fig.savefig(target.with_suffix('.png'),dpi=180)
    fig.savefig(target.with_suffix('.pdf'))
    plt.close(fig);csv_save(target.with_suffix('.csv'),rows)
    manifest.append(dict(stem=stem,caption=caption,rows=len(rows),
        files={kind:dict(path=str(target.with_suffix('.'+kind).relative_to(ROOT)),
            SHA256=digest(target.with_suffix('.'+kind))) for kind in ('png','pdf','csv')}))


def errorbar(ax,x,value,**kwargs):
    center=value.get('mean',value.get('estimate'));ci=value.get('CI95')
    if center is None:return
    errors=None if ci is None else np.array([[max(0,center-ci[0])],[max(0,ci[1]-center)]])
    ax.errorbar(x,center,yerr=errors,capsize=3,marker='o',**kwargs)


def safety_unqualified(row):
    gate=row['decode_gate'];enough=row['independent_starts']>=3
    return (not row['quality_passed_all_starts'] or not row.get('runtime_policy_passed_all_starts',True)
        or gate.get('absolute_decode_attainment_95percent_pass') is False
        or (enough and (not gate.get('qualified',False)
            or any(not value.get('qualified',False) for value in row.get('additional_decode_gates',{}).values()))))


def serving_family(data,stem,manifest):
    from ..dense_ar_output_aware_v2.c3_analysis import means_ci
    if data is None:return
    condition_fields=('name','start','seed','QPS','GPU_pair','dataset_SHA256','source_SHA256','model_revision',
        'B','S','KV_GiB','KV_block','source_credit','D_running_admission_cap','axis_kind','axis_cut',
        'hybrid_suffix_fraction','safe_retirement_enabled','P_compile','D_compile','output_contract','actual_geometry','actual_runtime',
        'runtime_policy','D_request_registration_deferred')
    def conditions(row):
        labels={row[key] for key in ('label','left','right') if key in row}
        selected=[value for value in data['starts'] if value['label'] in labels and
            value['mode']==row['mode'] and ('start' not in row or value['start']==row['start'])]
        return dict(row,execution_conditions=[{key:value.get(key) for key in condition_fields} for value in selected])
    for mode in dict.fromkeys(row['mode'] for row in data['cells']):
        cells=[r for r in data['cells'] if r['mode']==mode]
        if not cells:continue
        labels=[r['label'] for r in cells]
        fig,axes=plt.subplots(1,2,figsize=(13,4.7))
        for index,row in enumerate(cells):
            for ax,field in zip(axes,('TTFT_p95_ms','SLO_qualified_output_tokens_per_second')):
                errorbar(ax,index,row['fields'][field],color=COLORS[row['label']])
                if safety_unqualified(row):
                    ax.scatter(index,row['fields'][field]['mean'],marker='x',s=90,color='crimson',zorder=5)
                ax.annotate('N'+str(row['independent_starts']),(index,row['fields'][field]['mean']),xytext=(0,9),
                    textcoords='offset points',ha='center',fontsize=8)
        for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=30,ha='right');ax.grid(axis='y',alpha=.25)
        axes[0].set_ylabel('TTFT p95 (ms)');axes[1].set_ylabel('SLO-qualified output tokens/s')
        fig.suptitle('C3 '+mode+' | LongBench400-v2 heldout192 | QPS2 | '+stem)
        save(fig,stem+'_'+mode.lower()+'_ttft_goodput',[conditions(row) for row in cells],
            'MEASURED_ONLINE. Independent server-start means and empirical95% CI when N≥3. '
            'No request/token CI. Red×=quality/runtime-policy failure,observed absolute failure,or N3 decode gate not qualified. Finite cohort. '
            'Full gate details remain in the CSV.',manifest)
        # Additive means,not p95 attribution or GPU-exclusive costs.
        rows=[];fig,ax=plt.subplots(figsize=(13,5.3));bottom=np.zeros(len(labels))
        for stage,title in zip(STAGES,STAGE_NAMES):
            values=[]
            for label in labels:
                starts=[r for r in data['starts'] if r['mode']==mode and r['label']==label]
                values.append(float(np.mean([r['mean_host_path_ms'][stage] for r in starts])))
                rows.extend(dict(mode=mode,label=label,start=r['start'],stage=stage,ms=r['mean_host_path_ms'][stage]) for r in starts)
            ax.bar(range(len(labels)),values,bottom=bottom,label=title);bottom+=values
        ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right');ax.set_ylabel('Mean host path (ms)')
        ax.legend(ncols=3,fontsize=8);fig.suptitle('Additive mean TTFT boundaries | '+mode+' | '+stem)
        save(fig,stem+'_'+mode.lower()+'_pd_host_stages',[conditions(row) for row in rows],
            'TRACE_ANALYSIS. Host Prepare is not GPU start. Shared requests and chunk gaps enter these intervals. '
            'These means do not decompose TTFT p95. All measured starts remain visible.',manifest)
        comparisons=[r for r in data['comparisons'] if r['mode']==mode and r['right']=='StockToken']
        if comparisons:
            fig,axes=plt.subplots(1,2,figsize=(12,4.7))
            for index,row in enumerate(comparisons):
                for ax,key in zip(axes,('TTFT_p95_ms','SLO_qualified_output_tokens_per_second')):
                    errorbar(ax,index,row['ratios'][key],color=COLORS[row['left']])
                    cell=next(value for value in cells if value['label']==row['left'])
                    if safety_unqualified(cell):ax.scatter(index,row['ratios'][key]['estimate'],marker='x',s=90,color='crimson',zorder=5)
            for ax in axes:
                ax.axhline(1,color='black',ls='--');ax.set_xticks(range(len(comparisons)),
                    [r['left'] for r in comparisons],rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
            axes[0].set_ylabel('TTFT p95 / strong Stock Token');axes[1].set_ylabel('SLO-goodput / strong Stock Token')
            fig.suptitle('Same-arrival paired starts | '+mode+' | '+stem)
            save(fig,stem+'_'+mode.lower()+'_stock_ratios',[conditions(row) for row in comparisons],
                'MEASURED_ONLINE. Pointwise paired95% bootstrap when N≥3. Quality and both Native decode gates are separate. '
                'Red×=unqualified safety/quality as in the raw-metric view. Strong Stock keeps its compiled path and best-tuned policy.',manifest)
        # Both Native references are retained. Do not hide failing gates behind a TTFT win.
        fig,axes=plt.subplots(1,3,figsize=(15,4.8));decode_rows=[]
        for index,row in enumerate(cells):
            for ax,key in zip(axes[:2],('ITL_p99_ms','TPOT_p95_ms')):
                errorbar(ax,index,row['fields'][key],color=COLORS[row['label']])
            errorbar(axes[2],index,row['fields']['decode_SLO_attainment'],color=COLORS[row['label']])
            decode_rows.append(dict(mode=mode,label=row['label'],independent_starts=row['independent_starts'],
                fields=row['fields'],both_native_decode_gate=row['decode_gate'],
                additional_decode_gates=row.get('additional_decode_gates',{}),quality=row['quality_passed_all_starts']))
        axes[0].axhline(34,color='red',ls=':');axes[1].axhline(33,color='red',ls=':');axes[2].axhline(.95,color='red',ls=':')
        for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
        for ax,label in zip(axes,('Client ITL p99 (ms)','Request TPOT p95 (ms/token)','Every-ITL + TPOT request attainment')):ax.set_ylabel(label)
        fig.suptitle('Decode protection | '+mode+' | '+stem)
        save(fig,stem+'_'+mode.lower()+'_decode_protection',[conditions(row) for row in decode_rows],
            'MEASURED_ONLINE. Frozen every-client-ITL≤34ms and TPOT≤33ms. Absolute request attainment≥95%. '
            'CSV includes max ITL and paired relative5% gates against NativeStrong and exact Common-D-policy Native.',manifest)
    interventions=[r for r in data['comparisons'] if
        (r['left'].startswith('Optimized') and r['right'].startswith('Original')) or r['right']=='HybridGCControl']
    if interventions:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for index,row in enumerate(interventions):
            errorbar(axes[0],index,row['P_critical_path_delta_ms'],color=COLORS[row['left']])
            errorbar(axes[1],index,row['ratios']['TTFT_p95_ms'],color=COLORS[row['left']])
        labels=[r['mode']+(' GC deferred / normal' if r['right']=='HybridGCControl' else ' '+r['left'])
            for r in interventions]
        for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
        axes[0].axhline(0,color='black',ls='--');axes[1].axhline(1,color='black',ls='--')
        axes[0].set_ylabel('Candidate − control P arrival→ready (ms)');axes[1].set_ylabel('Candidate / control TTFT p95')
        fig.suptitle('Intervention | '+stem)
        save(fig,stem+'_p_intervention',[conditions(row) for row in interventions],
            'MEASURED_ONLINE. Paired independent server starts. Earlier dispatch alone is not a shorter P critical path. '
            'Credit retirement and partition geometry are separate experiment families.',manifest)


def kernel_figures(data,manifest):
    if data is None:return
    from ..dense_ar_output_aware_v2.c3_analysis import means_ci
    values=[r for r in data['observations'] if r['family']=='PUBLISHED_TRACE_REPRODUCTION_SAME_PAIR']
    if not values:return
    labels=[label for label in ('Layer','StockToken','Hybrid') if any(r['label']==label for r in values)]
    ticks=[label+'\nN='+str(sum(r['label']==label and r['observer_passed'] for r in values))+'/3 CPC' for label in labels]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));rows=[]
    for index,label in enumerate(labels):
        selected=[r for r in values if r['label']==label and r['observer_passed']]
        if not selected:continue
        active=float(np.mean([r['P_kernel_overlapping_path_mean_ms'] for r in selected]))
        empty=float(np.mean([r['no_observed_P_kernel_path_mean_ms'] for r in selected]))
        axes[0].bar(index,active,color=COLORS[label],label='P kernels overlap the request path' if index==0 else None)
        axes[0].bar(index,empty,bottom=active,color='#cccccc',label='No observed P kernel' if index==0 else None)
        axes[1].bar(index,np.mean([r['P_kernel_resource_demand_ms_per_request'] for r in selected]),color=COLORS[label])
        path_ci=means_ci([r['Arrival_to_P_ready_mean_ms'] for r in selected])
        resource_ci=means_ci([r['P_kernel_resource_demand_ms_per_request'] for r in selected])
        errorbar(axes[0],index,path_ci,color='black',markersize=2)
        errorbar(axes[1],index,resource_ci,color='black',markersize=2)
        rows.extend(dict(row,Arrival_to_P_ready_start_mean_CI95=path_ci['CI95'],
            P_kernel_resource_demand_start_mean_CI95=resource_ci['CI95']) for row in selected)
    for ax in axes:ax.set_xticks(range(len(labels)),ticks);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean Arrival→P-ready path (ms)');axes[1].set_ylabel('Global P kernel union / requests (ms/request)')
    axes[0].legend(fontsize=8);fig.suptitle('Waiting-path overlap ≠ own GPU service | QPS2 | same GPU0/1')
    save(fig,'p_queue_kernel_overlap',rows,
        'MEASURED_ONLINE. CPC-qualified start means. Total error bars=empirical start-level95% CI when N≥3. CSV retains denominators. '
        'Preceding/shared kernels can overlap many waiting requests. No-kernel time is not physical all-engine idle. DMA not measured.',manifest)
    categories=('GEMM','ATTENTION','RMSNORM','FUSED_RMSNORM_RESIDUAL','ROPE','ACTIVATION','KV_CACHE','UNKNOWN')
    fig,ax=plt.subplots(figsize=(12,5));bottom=np.zeros(len(labels));table=[]
    for category in categories+('OTHER_NAMED_OR_OVERLAP',):
        bar=[]
        for label in labels:
            selected=[r for r in values if r['label']==label and r['observer_passed']]
            costs=[]
            for row in selected:
                source=row['category_resource_demand_ms_per_request']
                cost=source.get(category,0) if category!='OTHER_NAMED_OR_OVERLAP' else sum(v for k,v in source.items() if k not in categories)
                costs.append(cost);table.append(dict(row_type='PER_START_CATEGORY',label=label,start=row['start'],category=category,ms_per_request=cost,name=row['name']))
            bar.append(np.mean(costs) if costs else np.nan)
        ax.bar(range(len(labels)),bar,bottom=bottom,label=category);bottom+=bar
    for index,label in enumerate(labels):
        selected=[r for r in values if r['label']==label and r['observer_passed']]
        if not selected:continue
        total=means_ci([row['P_kernel_resource_demand_ms_per_request'] for row in selected])
        errorbar(ax,index,total,color='black',markersize=2)
        table.append(dict(row_type='START_MEAN_TOTAL',label=label,estimate=total['mean'],CI95=total['CI95'],starts=len(selected)))
    ax.set_xticks(range(len(labels)),ticks);ax.set_ylabel('Exclusive category union / requests (ms/request)');ax.legend(ncols=3,fontsize=8)
    fig.suptitle('Actual C3 P named-kernel resource demand | best-vs-best,not iso-work')
    save(fig,'p_gpu_operator_resource_demand',table,
        'MEASURED_ONLINE. Kernel-name overlap partition. Total error bars=empirical start-level95% CI when N≥3. Unknown stays unknown. '
        'Projection subtypes,fused Norm/residual internals,SM and HBM are NOT_MEASURED. N and CPC status appear in the source table.',manifest)
    # Fixed midpoint request IDs avoid selecting unusually favorable or unfavorable requests.
    captured=[r for r in values if r['start']==0 and r['observer_passed']]
    if captured:
        fig,axes=plt.subplots(len(captured),1,figsize=(13,2.8*len(captured)),squeeze=False);table=[]
        for ax,row in zip(axes[:,0],captured):
            folder=ROOT/'raw'/row['name'];host=json.loads((ROOT/'analysis'/(row['name']+'_service_accounting.json')).read_text())
            requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
            start=json.loads((folder/'window.json').read_text())['start']
            chosen=[r for r in host['rows'] if 96<=int(r['request_id'][1:])<=101]
            for index,request in enumerate(chosen):
                cursor=1000*(requests[request['request_id']]['arrival']-start)
                for stage_index,(stage,title) in enumerate(zip(STAGES,STAGE_NAMES)):
                    width=request[stage];ax.broken_barh([(cursor,width)],(index-.35,.7),
                        facecolors=plt.get_cmap('tab10')(stage_index),label=title if index==0 else None)
                    table.append(dict(name=row['name'],label=row['label'],request_id=request['request_id'],stage=stage,
                        left_ms_from_cohort_start=cursor,duration_ms=width));cursor+=width
            ax.set_yticks(range(len(chosen)),[r['request_id'] for r in chosen]);ax.set_title(row['label']);ax.set_xlabel('ms from scheduled cohort start')
        axes[0,0].legend(ncols=3,fontsize=7)
        left=min(row['left_ms_from_cohort_start'] for row in table)
        right=max(row['left_ms_from_cohort_start']+row['duration_ms'] for row in table)
        for ax in axes[:,0]:ax.set_xlim(left-.02*(right-left),right+.02*(right-left))
        fig.suptitle('Same-trace midpoint requests96–101 | host critical paths,not kernel-exclusive intervals')
        save(fig,'p_request_critical_path_gantt',table,
            'TRACE_ANALYSIS. Fixed request IDs96–101 in start0. All panels use the same x-axis limits. Every stage is a real host boundary. '
            'Prepare does not mean GPU start. Shared computation and chunks can enter a stage. Same request/arrival/output work.',manifest)
    buckets=[r for r in data['descriptive_input_output_buckets'] if r['evidence']=='MEASURED_ONLINE_KERNEL_ACTIVITY']
    if buckets:
        fig,axes=plt.subplots(1,len(labels),figsize=(5*len(labels),4.8),squeeze=False)
        inputs=('3K-6K','6K-10K','10K-20K','20K-36K');outputs=('1-16','17-64','65-128','129+')
        matrices=[]
        for label in labels:
            matrix=np.full((4,4),np.nan)
            for i,inp in enumerate(inputs):
                for j,out in enumerate(outputs):
                    group=[r for r in buckets if r['label']==label and r['input_bucket']==inp and r['output_bucket']==out]
                    if group:matrix[i,j]=np.mean([r['Arrival_to_P_ready_ms'] for r in group])
            matrices.append(matrix)
        low=min(float(np.nanmin(matrix)) for matrix in matrices)
        high=max(float(np.nanmax(matrix)) for matrix in matrices)
        for ax,label,matrix in zip(axes[0],labels,matrices):
            img=ax.imshow(matrix,aspect='auto',vmin=low,vmax=high);fig.colorbar(img,ax=ax,label='Mean P arrival→ready (ms)')
            ax.set_xticks(range(4),outputs);ax.set_yticks(range(4),inputs);ax.set_xlabel('Actual replay output tokens');ax.set_title(label)
        fig.suptitle('Descriptive input/output subgroups | no causal output-length claim')
        save(fig,'p_input_output_bottleneck_map',buckets,
            'TRACE_ANALYSIS. Mixed-replay subgroups. All panels share color limits. Missing cells remain blank. '
            'These are not fixed-output16/64/128/192 interventions or independent request-level confidence intervals.',manifest)


def fixed_output_paths(data,manifest):
    if not data:return
    selected=[row for row in data['starts'] if row['start']==0 and row['mode'] in ('Fixed16','Fixed64','Fixed128','Fixed192')]
    if not selected:return
    rows=[];fig,axes=plt.subplots(1,3,figsize=(15,5))
    for label in dict.fromkeys(row['label'] for row in selected):
        points=sorted([row for row in selected if row['label']==label],key=lambda row:int(row['mode'][5:]))
        assert len({(row['seed'],tuple(row['GPU_pair']),row['dataset_SHA256'],row['source_SHA256']) for row in points})==1
        output=[];p_cost=[];d_cost=[];residency=[]
        for row in points:
            length=int(row['mode'][5:]);requests=records(ROOT/'raw'/row['name']/'requests.jsonl')
            assert requests and all(value['output_tokens']==length for value in requests)
            span=float(np.mean([1000*(value['token_events'][-1]['worker_time']-value['token_events'][0]['worker_time']) for value in requests]))
            p=row['mean_P_arrival_to_ready_ms'];d=row['mean_host_path_ms']['TTFT_ms']-p
            output.append(length);p_cost.append(p);d_cost.append(d);residency.append(span)
            rows.append(dict(row,controlled_output=length,independent_starts_in_this_plot=1,
                mean_D_worker_first_to_last_emission_ms=span,mean_post_P_host_path_ms=d))
        for ax,values in zip(axes,(p_cost,d_cost,residency)):
            ax.plot(output,values,marker='o',color=COLORS[label],label=label)
    for ax,title in zip(axes,('Mean Arrival→P-ready host path (ms)','Mean P-ready→client-first host path (ms)',
        'Mean D first→last worker-emission span (ms)')):
        ax.set_title(title);ax.set_xlabel('Forced output tokens/request');ax.set_xticks((16,64,128,192));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Fixed-output intervention | each arm keeps prompts,arrival,pair and partition | N1 common start only')
    save(fig,'fixed_output_residency_host_paths',rows,
        'MEASURED_ONLINE / TRACE_ANALYSIS. Use the common start0 at every output length. No CI from requests. '
        'Output192 N3 is reported separately. Worker-emission span includes scheduling gaps,not GPU-exclusive decode time. '
        'Output changes reservation and residency together. This is not a residency-only causal intervention.',manifest)


def main():
    os.environ.setdefault('SOURCE_DATE_EPOCH',str(START))
    bind();plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    kernel_figures(load('p_critical_causal_summary.json'),manifest)
    serving_family(load('c3_credit_confirmation_analysis.json'),'credit_retirement',manifest)
    serving_family(load('p_geometry_confirmation_analysis.json'),'partition_geometry',manifest)
    serving_family(load('p_output_stress_analysis.json'),'fixed_output_stress',manifest)
    fixed_output_paths(load('p_output_stress_analysis.json'),manifest)
    serving_family(load('coordinator_hygiene_confirmation_analysis.json'),'coordinator_hygiene',manifest)
    serving_family(load('coordinator_hygiene_stable_stock_confirmation_analysis.json'),'coordinator_hygiene_stable_stock',manifest)
    serving_family(load('coordinator_hygiene_fixed_ports_confirmation_analysis.json'),'coordinator_hygiene_fixed_ports',manifest)
    serving_family(load('coordinator_hygiene_warm_regime_confirmation_analysis.json'),'coordinator_hygiene_warm_regime',manifest)
    serving_family(load('output192_D_cap_intervention_analysis.json'),'output192_d_cap_intervention',manifest)
    serving_family(load('p_geometry_D_cap4_confirmation_analysis.json'),'geometry_d_cap4',manifest)
    serial=load('serial_serving_operator_analysis.json')
    if serial:
        from ..dense_ar_output_aware_v2.c3_analysis import means_ci
        selected=[row for row in serial['results'] if row.get('CPC_passed')]
        families=list(dict.fromkeys(row['family'] for row in selected))
        for family in families:
            rows=[];values=[row for row in selected if row['family']==family]
            labels=list(dict.fromkeys(row['label'] for row in values))
            fig,axes=plt.subplots(1,3,figsize=(14,4.9))
            for ax,phase in zip(axes,('P_arrival_to_ready','D_submit_to_first_worker','D_subsequent_decode')):
                for index,label in enumerate(labels):
                    costs=[p for row in values if row['label']==label for p in row['phase_means'] if p['phase']==phase]
                    active=float(np.mean([row['mean_GPU_kernel_union_ms'] for row in costs]))
                    empty=float(np.mean([row['mean_no_observed_kernel_ms'] for row in costs]))
                    ax.bar(index,active,color=COLORS[label]);ax.bar(index,empty,bottom=active,color='#cccccc')
                    errorbar(ax,index,means_ci([row['mean_host_ms'] for row in costs]),color='black',markersize=2)
                    rows.extend(dict(family=family,label=label,start=row['start'],
                        **next(p for p in row['phase_means'] if p['phase']==phase)) for row in values if row['label']==label)
                ticks=[label+'\nN='+str(sum(row['label']==label for row in values)) for label in labels]
                ax.set_xticks(range(len(labels)),ticks,rotation=25,ha='right');ax.set_ylabel('Mean enclosure (ms)')
                ax.set_title(phase);ax.grid(axis='y',alpha=.2)
            fig.suptitle('Serial actual serving API | Batch1/1/1 | 32 prompts | Output16 | '+family)
            save(fig,'serial_pd_service_'+family.lower(),rows,
                'ISOLATED_KERNEL_MEASUREMENT. Colored=observed kernel union. Grey=no observed kernel,not physical all-engine idle. '
                'Only CPC-qualified starts enter bars. Total error bars=empirical start-level95% CI when N≥3. '
                'D-first includes suffix and first output,not suffix-only kernels.',manifest)
    write(ROOT/'analysis/figure_manifest.json',dict(time=time.time(),figures=manifest,
        scope='PNG/PDF figures and exact CSV rows. No interpolated measurement or inferred CI. '
            'A partial N is labelled N. These figures do not claim that the campaign is complete.'))
    print('P_CRITICAL_FIGURES',len(manifest),flush=True)


if __name__=='__main__':main()
