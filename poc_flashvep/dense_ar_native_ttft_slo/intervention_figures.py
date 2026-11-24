"""Plot policy effects separately from serial kernel measurements."""
import json
import csv
import itertools
import os
import time
import numpy as np
from .common import ROOT,write,digest,preserve
from .failure_analysis import save_csv


def build():
    os.environ.setdefault('SOURCE_DATE_EPOCH','1795449700')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..dense_ar_p_critical_path.kernel_analysis import analyze_activity
    from .profile_analysis import records,clock_offset
    preserve();manifest=[]
    def save(fig,stem,rows,scope,evidence):
        fig.text(.01,.01,scope,fontsize=7)
        fig.tight_layout(rect=[0,.075,1,1])
        save_csv(ROOT/'figures'/(stem+'.csv'),[dict(model_revision='b968826d9c46dd6066d109eabc6255188de91218',
            metadata_reference='controlled_intervention_per_start.csv or serial_native_relative_per_start.csv',**row) for row in rows])
        for ext in ('png','pdf'):
            fig.savefig(ROOT/'figures'/(stem+'.'+ext),**(dict(dpi=170) if ext=='png'
                else dict(metadata={'CreationDate':None,'ModDate':None})))
        plt.close(fig)
        manifest.append(dict(stem=stem,evidence=evidence,scope=scope,
            files={ext:digest(ROOT/'figures'/(stem+'.'+ext)) for ext in ('png','pdf','csv')}))
    profile_path=ROOT/'analysis/serial_native_relative_kernel_analysis_v2.json'
    if profile_path.exists():
        profile=json.loads(profile_path.read_text())
        labels=('NativeStrong','StockToken','Layer','Hybrid')
        phases=('P_arrival_to_ready','D_submit_to_first_worker','D_subsequent_decode')
        fig,axes=plt.subplots(1,3,figsize=(13,4.6));table=[]
        for ax,phase in zip(axes,phases):
            for index,label in enumerate(labels):
                rows=[next(p for p in r['phase_means'] if p['phase']==phase) for r in profile['results']
                    if r['label']==label and r.get('CPC',{}).get('passed')]
                if not rows:
                    diagnostic=any(r['label']==label and r['evidence']=='DIAGNOSTIC' for r in profile['results'])
                    status='DIAGNOSTIC_ONLY' if diagnostic else 'NO_QUALIFIED_CAPTURE'
                    ax.text(index,.02,status,rotation=90,ha='center',va='bottom',fontsize=7,
                        transform=ax.get_xaxis_transform())
                    table.append(dict(method=label,phase=phase,qualified_starts=0,evidence=status,
                        scope='Do not promote the failed CPC into a primary kernel bar. Its raw diagnostic cost remains in the profile tables.'))
                    continue
                host=np.asarray([r['host_enclosure_ms'] for r in rows])
                kernel=np.asarray([r['GPU_kernel_union_ms'] for r in rows])
                ax.bar(index,kernel.mean(),color='#4c78a8')
                ax.bar(index,(host-kernel).mean(),bottom=kernel.mean(),color='#d9d9d9')
                ax.text(index,host.mean(),f'N={len(rows)}',ha='center',va='bottom',fontsize=8)
                for field,values in (('host_enclosure_ms',host),('GPU_kernel_union_ms',kernel)):
                    from .figures import means_ci
                    mean,ci=means_ci(values)
                    table.append(dict(method=label,phase=phase,metric=field,qualified_starts=len(rows),mean_ms=mean,CI95=ci,
                        evidence='ISOLATED_KERNEL_MEASUREMENT',GPU_pair=[0,1],count_per_start=32,output_tokens_per_request=16))
            ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Mean ms/request',title=phase.replace('_',' '))
            ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
        save(fig,'serial_native_split_kernel_and_host',table,
            'Serial32 | Actual Batch1/1/1 | Output16 | qualified CPC only. Blue: kernel union. Grey: no observed kernel.\n'
            'Grey is not physical idle or DMA. D-first includes continuation and first-output work. This is not Main400 Natural TTFT.',
            'ISOLATED_KERNEL_MEASUREMENT')
        # Select the first qualified triplet,not the fastest triplet.
        fig,axes=plt.subplots(4,1,figsize=(12,7));table=[]
        for ax,label in zip(axes,labels):
            selected=next((r for r in profile['results'] if r['label']==label and r.get('CPC',{}).get('passed')),None)
            if selected is None:
                diagnostic=any(r['label']==label and r['evidence']=='DIAGNOSTIC' for r in profile['results'])
                status='DIAGNOSTIC_ONLY' if diagnostic else 'NO_QUALIFIED_CAPTURE'
                ax.text(.5,.5,label+': '+status,ha='center',transform=ax.transAxes)
                table.append(dict(method=label,evidence=status,scope='No qualified timeline is plotted.'))
                continue
            folder=ROOT/'raw'/selected['jobs'][1]
            request=next(r for r in records(folder/'requests.jsonl') if r['sample_id']==0)
            origin=request['arrival'];end=request['token_events'][-1]['worker_time']
            for role,y,color in (('prefill',1,'#4c78a8'),('decode',0,'#b279a2')):
                activity=analyze_activity(folder/('activity_'+role));roi=activity['ROI']
                offset=clock_offset(roi['left'])
                spans=[]
                for left,right in activity['intervals']:
                    a=max(origin,(left-offset)/1e9);b=min(end,(right-offset)/1e9)
                    if a>=b:continue
                    spans.append((1000*(a-origin),1000*(b-a)))
                    table.append(dict(method=label,start=selected['start'],request_id='q0',prompt_id=request['prompt_id'],
                        role=role,left_ms=1000*(a-origin),right_ms=1000*(b-origin),evidence='ISOLATED_KERNEL_MEASUREMENT'))
                ax.broken_barh(spans,(y-.25,.5),facecolors=color)
            ax.set(yticks=[0,1],yticklabels=['D','P'],title=label,xlabel='Time since q0 arrival (ms)')
        save(fig,'serial_native_split_actual_kernel_timeline',table,
            'First request of the first qualified prospective CPC per arm. Source,ROI and clock uncertainty remain in the profile analysis.\n'
            'CUDA kernel activity only. White space does not prove no DMA or a safe future decode window.',
            'ISOLATED_KERNEL_MEASUREMENT')
    path=ROOT/'analysis/controlled_intervention_analysis.json'
    if path.exists():
        analysis=json.loads(path.read_text());contrasts=[r for r in analysis['contrasts'] if r['one_signature_factor']]
        for family in sorted({r['family'] for r in contrasts}):
            selected=[r for r in contrasts if r['family']==family]
            fig,axes=plt.subplots(1,2,figsize=(13,5));table=[]
            for ax,metric in zip(axes,('TTFT_p95_ms','SLO_qualified_output_tokens_per_second')):
                for index,row in enumerate(selected):
                    test=row['ratios'][metric];point=test['estimate'];ci=test['CI95']
                    ax.scatter(point,index,color='#1768ac' if row['quality_passed'] else '#c44e52',marker='o')
                    if ci is not None:ax.plot(ci,[index,index],color='#1768ac')
                    table.append(dict(family=family,candidate=row['candidate'],control=row['control'],metric=metric,
                        ratio=point,CI95=ci,paired_starts=row['paired_starts'],quality_passed=row['quality_passed'],
                        changed_signature_fields=row['changed_signature_fields'],evidence=row['evidence']))
                ax.axvline(1,color='black',linestyle='--',linewidth=1)
                ax.set(yticks=range(len(selected)),yticklabels=[r['candidate']+' / '+r['control'] for r in selected],
                    xlabel='Ratio of paired start means',title=metric);ax.grid(axis='x',alpha=.2)
            save(fig,'policy_effects_'+family.lower(),table,
                family+' | QPS2 | GPUs0/1 | warm compiled paths | exact paired arrivals.\n'
                'Pointwise start-level CI only for N>=3. Red: quality gate failed or pending. Decode gates remain separate.',
                'MEASURED_ONLINE')
            gates=[r for r in analysis['decode_gates'] if r['family']==family]
            fig,axes=plt.subplots(1,2,figsize=(13,5));table=[]
            for ax,metric in zip(axes,('ITL_p99_ms','TPOT_p95_ms')):
                for index,row in enumerate(gates):
                    for offset,reference,color in ((-.13,'NativeStrong','#1768ac'),(.13,'NativeCommon','#ef8a17')):
                        test=row['decode_gate']['tests'][reference][metric]
                        ax.scatter(test['estimate'],index+offset,color=color,s=20)
                        if test['CI95'] is not None:ax.plot(test['CI95'],[index+offset]*2,color=color)
                        table.append(dict(family=family,method=row['label'],reference=reference,
                            actual_common_reference=row['common_D_policy_reference'],metric=metric,ratio=test['estimate'],
                            CI95=test['CI95'],decode_gate=row['decode_gate'],quality_passed=row['quality_passed']))
                ax.axvline(1.05,color='red',linestyle='--',linewidth=1)
                ax.set(yticks=range(len(gates)),yticklabels=[r['label'] for r in gates],xlabel='Native-relative ratio',title=metric)
                ax.grid(axis='x',alpha=.2)
            save(fig,'policy_decode_gates_'+family.lower(),table,
                family+' | Blue: NativeStrong. Orange: exact common-D Native. Source windows can remain different.\n'
                'Relative CI upper<=1.05 and absolute request decode attainment>=95% are separate requirements.',
                'MEASURED_ONLINE')
    compact=ROOT/'figures/main400_request_slo_failures.csv'
    if compact.exists():
        with compact.open(newline='') as stream:
            rows=[row for row in csv.DictReader(stream) if row['family']=='Current_Main400_Natural']
        if rows:
            fig,ax=plt.subplots(figsize=(9,5));table=[]
            for label in ('NativeStrong','NativeCommon0','NativeCommon2','StockToken','StockCap2','Layer','Hybrid'):
                part=[row for row in rows if row['label']==label]
                starts=sorted({int(row['start']) for row in part})
                if not starts:continue
                grid=np.unique([0.]+[float(row['TTFT_ms']) for row in part])
                curves=np.asarray([np.searchsorted(sorted(float(row['TTFT_ms']) for row in part
                    if int(row['start'])==index),grid,side='right')/sum(int(row['start'])==index for row in part)
                    for index in starts])
                mean=curves.mean(axis=0)
                line,=ax.step(grid/1000,mean,where='post',label=label+' N='+str(len(starts)))
                if len(starts)>=3:
                    boot=np.asarray(list(itertools.product(range(len(starts)),repeat=len(starts))))
                    low,high=np.percentile(curves[boot].mean(axis=1),[2.5,97.5],axis=0)
                    ax.fill_between(grid/1000,low,high,step='post',color=line.get_color(),alpha=.05)
                else:low=high=[None]*len(grid)
                table.extend(dict(family='Current_Main400_Natural',method=label,TTFT_ms=float(x),CDF=float(y),
                    CI95_low=None if a is None else float(a),CI95_high=None if b is None else float(b),
                    starts=len(starts),requests=len(part),QPS=2.,GPU_pair=[0,1],
                    seeds=sorted({int(row['seed']) for row in part}),
                    dataset_SHA256=sorted({row['dataset_SHA256'] for row in part}),
                    worker_source_SHA256=sorted({row['source_SHA256'] for row in part}),
                    evidence='FRESH_RAW_TRACE_DISTRIBUTION') for x,y,a,b in zip(grid,mean,low,high))
            ax.axvline(5,color='black',linestyle='--',label='5s TTFT SLO')
            ax.set(xlabel='TTFT (s)',ylabel='Request fraction at or below',title='Fresh Main400 Natural: TTFT distribution')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
            save(fig,'fresh_main400_native_split_ttft_CDF',table,
                'Fresh warm-runtime family only | QPS2 | same400 prompts and paired arrivals. No pooling with the published v2 family.\n'
                'CSV preserves pointwise start-level95% CI. Natural output work and realized batches can differ. No fresh Agg rerun.',
                'MEASURED_ONLINE')
    write(ROOT/'analysis/intervention_figure_manifest.json',dict(time=time.time(),passed=True,figures=manifest,
        analysis_inputs_SHA256={str(path.relative_to(ROOT)):digest(path) for path in
            (profile_path,ROOT/'analysis/controlled_intervention_analysis.json',compact) if path.exists()},
        scope='Serial activity,online policy effects and decode gates use separate figures and source families.'))
    preserve();return manifest


if __name__=='__main__':print('INTERVENTION_FIGURES',len(build()),flush=True)
