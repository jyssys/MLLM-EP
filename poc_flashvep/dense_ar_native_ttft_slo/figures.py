"""Render measured distributions and explicitly offline metric accounting."""
import collections
import csv
import itertools
import json
import os
import time
import numpy as np
from .common import ROOT,write,digest,preserve
from .failure_analysis import save_csv
from .causal_analysis import read_csv

METHODS=('Agg','NativeStrong','NativeCommon0','NativeCommon2','StockToken','Layer','Hybrid')
COLORS=dict(zip(METHODS,('#9e9e9e','#1768ac','#4eb3d3','#53a2be','#ef8a17','#bd3786','#00a087')))
MAIN='Historical_Main400_Natural'


def means_ci(values):
    a=np.asarray(values,float);mean=float(a.mean())
    if len(a)<3:return mean,None
    indices=np.asarray(list(itertools.product(range(len(a)),repeat=len(a))))
    return mean,np.percentile(a[indices].mean(axis=1),[2.5,97.5]).tolist()


def build():
    os.environ.setdefault('SOURCE_DATE_EPOCH','1795449700')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    preserve()
    failure=json.loads((ROOT/'analysis/request_failure_attribution.json').read_text())
    causal=json.loads((ROOT/'analysis/native_relative_causal_comparisons.json').read_text())
    paths=json.loads((ROOT/'analysis/request_critical_paths.json').read_text())
    requests=[row for row in read_csv(ROOT/'figures/request_slo_failures.csv') if row['family']==MAIN]
    starts=[row for row in failure['starts'] if row['family']==MAIN]
    by_method=collections.defaultdict(list)
    for row in requests:by_method[row['label']].append(row)
    manifest=[]
    condition=dict(family=MAIN,QPS=2.0,requests_per_start=400,independent_starts=3,
        model_revision='b968826d9c46dd6066d109eabc6255188de91218',
        dataset_SHA256='0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9',
        numerical_mode='BI0',B=8192,S=1024,KV_GiB_per_GPU=42,KV_block_tokens=128,
        client_scope='Coordinator delivery,not HTTP',throughput_scope='Output-only finite cohort,not sustainable capacity',
        source_and_GPU_metadata='metrics_per_start.csv and request_slo_failures.csv',
        evidence='HISTORICAL_RAW_RECALCULATION')
    def save(fig,stem,table,scope,evidence='HISTORICAL_RAW_RECALCULATION'):
        fig.text(.01,.01,'Historical v2 raw recalculation | Qwen3-8B | H100x2 | Main400 Natural | QPS2 | BI0 | N3\n'+scope,fontsize=7)
        fig.tight_layout(rect=[0,.065,1,1]);rows=[dict(condition,**row) for row in table]
        save_csv(ROOT/'figures'/(stem+'.csv'),rows)
        for ext in ('png','pdf'):
            kwargs=dict(dpi=170) if ext=='png' else dict(metadata={'CreationDate':None,'ModDate':None})
            fig.savefig(ROOT/'figures'/(stem+'.'+ext),**kwargs)
        plt.close(fig)
        manifest.append(dict(stem=stem,evidence=evidence,scope=scope,conditions=condition,
            files={ext:digest(ROOT/'figures'/(stem+'.'+ext)) for ext in ('png','pdf','csv')}))
    # Empirical CDF: each independent start contributes equal weight.
    fig,ax=plt.subplots(figsize=(8,4.5));table=[]
    for label in METHODS:
        rows=by_method[label];grid=np.unique([0.]+[float(row['TTFT_ms']) for row in rows])
        curves=np.asarray([np.searchsorted(sorted(float(r['TTFT_ms']) for r in rows if int(r['start'])==s),grid,side='right')/400 for s in range(3)])
        indices=np.asarray(list(itertools.product(range(3),repeat=3)));lo,hi=np.percentile(curves[indices].mean(axis=1),[2.5,97.5],axis=0)
        mean=curves.mean(axis=0);ax.step(grid/1000,mean,where='post',label=label,color=COLORS[label]);ax.fill_between(grid/1000,lo,hi,step='post',alpha=.08,color=COLORS[label])
        table.extend(dict(method=label,TTFT_ms=float(x),CDF=float(y),CI95_low=float(a),CI95_high=float(b)) for x,y,a,b in zip(grid,mean,lo,hi))
    ax.axvline(5,color='black',linestyle='--',linewidth=1,label='5s TTFT SLO');ax.set(xlabel='TTFT (s)',ylabel='Request fraction at or below',title='TTFT empirical CDF');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'method_ttft_CDF',table,'Step CDF; pointwise start-level 95% bootstrap CI. No request-level superiority CI.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));table=[]
    all_intervals=collections.defaultdict(list)
    with (ROOT/'figures/client_ITL_intervals.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['family']==MAIN and row['label'] in METHODS:all_intervals[row['label']].append(float(row['ITL_ms']))
    for label in METHODS:
        for ax,kind,values in ((axes[0],'ITL',all_intervals[label]),(axes[1],'TPOT',[float(row['TPOT_ms']) for row in by_method[label] if row['TPOT_ms']])):
            values=np.sort(values);grid=np.unique(np.percentile(values,np.r_[np.linspace(0,99,160),99.5,99.9,100]))
            cdf=np.searchsorted(values,grid,side='right')/len(values);ax.step(grid,cdf,where='post',label=label,color=COLORS[label])
            table.extend(dict(method=label,metric=kind,ms=float(x),CDF=float(y),observations=len(values)) for x,y in zip(grid,cdf))
    for ax,limit,title in zip(axes,(34,33),('Client-visible ITL','Request TPOT')):
        ax.axvline(limit,color='black',linestyle='--');ax.set_xscale('symlog',linthresh=1);ax.set(xlabel='ms',ylabel='Empirical fraction',title=title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    save(fig,'method_ITL_TPOT_distribution',table,'Descriptive pooled intervals/requests,not independent starts. Zero co-delivery ITLs stay zero; TPOT=None is excluded.')
    fig,axes=plt.subplots(2,4,figsize=(14,6));table=[]
    for ax,label in zip(axes.flat,METHODS):
        for start in range(3):
            rows=sorted((r for r in by_method[label] if int(r['start'])==start),key=lambda r:int(r['sample_id']));origin=min(float(r['arrival']) for r in rows)
            ax.scatter([float(r['arrival'])-origin for r in rows],[float(r['TTFT_ms'])/1000 for r in rows],s=4,alpha=.45,label='start'+str(start))
            table.extend(dict(method=label,start=start,sample_id=int(r['sample_id']),arrival_seconds=float(r['arrival'])-origin,TTFT_ms=float(r['TTFT_ms'])) for r in rows)
        ax.axhline(5,color='black',linestyle='--',linewidth=.8);ax.set(title=label,xlabel='Scheduled arrival (s)',ylabel='TTFT (s)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6)
    axes.flat[-1].set_visible(False)
    save(fig,'request_TTFT_vs_arrival',table,'Each point is an observed request. Repeated starts are shown separately. No burst-only causal intervention.')
    labels=('NativeStrong','NativeCommon0','NativeCommon2','StockToken','Layer','Hybrid');host=[r for r in paths['starts'] if r['family']==MAIN]
    stage_names=paths['stage_names'];fig,ax=plt.subplots(figsize=(11,5));bottom=np.zeros(len(labels));table=[]
    palette=plt.get_cmap('tab20')
    for k,field in enumerate(stage_names):
        heights=[]
        for label in labels:
            values=[r['means'][field] for r in host if r['label']==label];mean,ci=means_ci(values);heights.append(mean/1000)
            table.append(dict(method=label,host_stage=field,mean_ms=mean,CI95=ci,evidence='TRACE_ANALYSIS_HOST_ENCLOSURE'))
        ax.bar(labels,heights,bottom=bottom,label=field.removesuffix('_ms'),color=palette(k));bottom+=heights
    ax.set(ylabel='Mean client TTFT (s)',title='Additive host critical-path boundaries');ax.legend(fontsize=6,ncol=2);ax.grid(axis='y',alpha=.2)
    ax.tick_params(axis='x',rotation=20)
    save(fig,'native_split_host_stage_breakdown',table,'Additive means only; Prepare is not GPU start. Enclosures include queue and host work. Not a p95 or GPU-exclusive decomposition.','TRACE_ANALYSIS')
    fig,axes=plt.subplots(2,2,figsize=(11,6));table=[]
    with (ROOT/'figures/request_critical_paths.csv').open(newline='') as stream:
        traces=[r for r in csv.DictReader(stream) if r['family']==MAIN and r['label'] in ('NativeStrong','StockToken','Layer','Hybrid') and int(r['start'])==0]
    for ax,label in zip(axes.flat,('NativeStrong','StockToken','Layer','Hybrid')):
        part=sorted((r for r in traces if r['label']==label),key=lambda r:int(r['sample_id']));origin=min(json.loads(r['boundaries'])['arrival'] for r in part)
        for row in part:
            b=json.loads(row['boundaries']);sample=int(row['sample_id'])
            for phase,left,right,color in (('P prepare→ready',b['P_prepare'],b['P_ready'],'#4c78a8'),('D accept→prepare',b['D_accept'],b['D_prepare'],'#f2b134'),('D prepare→first worker',b['D_prepare'],b['first_worker'],'#b279a2')):
                ax.plot([left-origin,right-origin],[sample,sample],color=color,linewidth=.6)
                table.append(dict(method=label,start=0,sample_id=sample,phase=phase,left_seconds=left-origin,right_seconds=right-origin,evidence='HOST_ENCLOSURE_NOT_GPU_SERVICE'))
        ax.set(title=label,xlabel='Time since first scheduled arrival (s)',ylabel='Request order')
    save(fig,'P_D_service_queue_host_timeline',table,'Representative start0 only. Host envelopes,not GPU kernel activity. Kernel service controls are separate.','TRACE_ANALYSIS')
    categories=failure['categories'];fig,ax=plt.subplots(figsize=(10,4.5));bottom=np.zeros(len(METHODS));table=[]
    names=list(dict.fromkeys(r['category'] for r in categories));colors=plt.get_cmap('tab20')
    for k,category in enumerate(names):
        values=[]
        for label in METHODS:
            part=[r for r in categories if r['family']==MAIN and r['label']==label and r['category']==category]
            mean,ci=means_ci([r['request_fraction'] for r in part]);values.append(mean*100)
            table.append(dict(method=label,category=category,mean_request_fraction=mean,CI95=ci,request_observations=sum(r['request_count'] for r in part),output_tokens=sum(r['output_tokens'] for r in part)))
        ax.bar(METHODS,values,bottom=bottom,label=category,color=colors(k));bottom+=values
    ax.set(ylabel='Request observations (%)',title='All eight joint-SLO outcomes');ax.legend(fontsize=6,ncol=2)
    ax.tick_params(axis='x',rotation=20)
    save(fig,'joint_SLO_eight_category_breakdown',table,'400 requests x3 starts per method; 1200 observations are not 1200 unique requests or independent starts.')
    fig,ax=plt.subplots(figsize=(10,4.5));table=[];x=np.arange(len(METHODS));width=.19
    for offset,field in enumerate(('TTFT_SLO_attainment','every_ITL_SLO_attainment','TPOT_SLO_attainment','Joint_SLO_attainment')):
        values=[];errors=[]
        for label in METHODS:
            mean,ci=means_ci([r[field] for r in starts if r['label']==label]);values.append(mean*100);errors.append([mean-ci[0],ci[1]-mean]);table.append(dict(method=label,metric=field,mean=mean,CI95=ci))
        ax.bar(x+(offset-1.5)*width,values,width,label=field.replace('_SLO_attainment',''),yerr=np.array(errors).T*100,capsize=2)
    ax.set(xticks=x,xticklabels=METHODS,ylabel='Request attainment (%)',title='Individual metric passes and joint pass');ax.legend(fontsize=7)
    ax.tick_params(axis='x',rotation=20)
    save(fig,'individual_and_joint_SLO_attainment',table,'Frozen TTFT<=5s, every client ITL<=34ms, request TPOT<=33ms/token. Start-level pointwise 95% CI.')
    fig,ax=plt.subplots(figsize=(9,4.5));table=[]
    for offset,field in enumerate(('generated_output_tokens_per_second','SLO_qualified_output_tokens_per_second')):
        means=[];errors=[]
        for label in METHODS:
            mean,ci=means_ci([r[field] for r in starts if r['label']==label]);means.append(mean);errors.append([mean-ci[0],ci[1]-mean]);table.append(dict(method=label,metric=field,start_mean=mean,CI95=ci))
        ax.bar(np.arange(len(METHODS))+(offset-.5)*.35,means,.35,label='Raw output' if offset==0 else 'Joint-SLO qualified output',yerr=np.array(errors).T,capsize=2)
    ax.set(xticks=np.arange(len(METHODS)),xticklabels=METHODS,ylabel='Generated output tokens/s',title='Raw throughput and SLO-goodput');ax.legend(fontsize=8)
    ax.tick_params(axis='x',rotation=20)
    save(fig,'raw_output_throughput_vs_SLO_goodput',table,'Actual Natural EOS work and measured cohort duration per start. Finite-cohort rates,not sustainable capacity.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));table=[]
    for label in ('NativeStrong','StockToken','Layer','Hybrid'):
        rows=[r for r in by_method[label] if r['JOINT_PASS']=='False'];n=np.sort([int(r['output_tokens']) for r in rows]);grid=np.unique(n);cdf=np.searchsorted(n,grid,side='right')/len(n)
        axes[0].step(grid,cdf,where='post',label=label,color=COLORS[label]);axes[0].set_xscale('log')
        tt=[r for r in by_method[label] if r['TTFT_PASS']=='False'];it=[r for r in by_method[label] if r['ITL_PASS']=='False']
        table.append(dict(record_type='SUMMARY',method=label,joint_fail_count=len(rows),joint_fail_output_mean=float(n.mean()),
            TTFT_fail_count=len(tt),TTFT_fail_output_mean=float(np.mean([int(r['output_tokens']) for r in tt])) if tt else None,
            ITL_fail_count=len(it),ITL_fail_output_mean=float(np.mean([int(r['output_tokens']) for r in it])) if it else None))
        table.extend(dict(record_type='JOINT_FAILED_OUTPUT_CDF',method=label,output_tokens=int(x),CDF=float(y),
            observations=len(n)) for x,y in zip(grid,cdf))
    names=('NativeStrong','StockToken','Layer','Hybrid');axes[1].bar(names,[next(r['ITL_fail_output_mean'] for r in table
        if r['method']==label and r['record_type']=='SUMMARY') for label in names],color=[COLORS[label] for label in names])
    axes[0].set(xlabel='Actual output tokens in joint-failed requests',ylabel='Request fraction at or below',title='Failed-request length distribution');axes[0].legend(fontsize=7)
    axes[1].set(ylabel='Mean actual output tokens',title='All-ITL-failed request length')
    save(fig,'SLO_failed_output_token_distribution',table,'Descriptive repeated-request observations. Natural output lengths vary across methods; no quality-loss inference.')
    fig,ax=plt.subplots(figsize=(10,4.5));table=[];base=[]
    for label in METHODS:base.append(float(np.mean([r['SLO_qualified_output_tokens_per_second'] for r in starts if r['label']==label])))
    ax.bar(METHODS,base,label='Measured joint-SLO goodput',color='#277da1');bottom=np.asarray(base)
    for gate,color in zip(('TTFT','ITL','TPOT'),('#f8961e','#f94144','#90be6d')):
        values=[]
        for label in METHODS:
            part=[r for r in failure['shapley'] if r['family']==MAIN and r['label']==label and r['gate']==gate];mean=float(np.mean([r['attributed_goodput_loss'] for r in part]));values.append(mean)
            table.append(dict(method=label,gate=gate,metric_attributed_excluded_tokens_per_second=mean,evidence='OFFLINE_METRIC_FAILURE_SHAPLEY'))
        ax.bar(METHODS,values,bottom=bottom,label=gate+' metric-attributed exclusion',color=color);bottom+=values
    ax.set(ylabel='Output tokens/s at frozen measured duration',title='Offline SLO-loss waterfall to raw throughput');ax.legend(fontsize=7)
    ax.tick_params(axis='x',rotation=20)
    save(fig,'SLO_counterfactual_metric_waterfall',table,'Exact three-gate Shapley allocation of metric losses. No GPU,queue,admission or runtime speedup is simulated.','OFFLINE_METRIC_FAILURE_SHAPLEY')
    value=dict(time=time.time(),passed=True,figures=manifest,analysis_inputs_SHA256={name:digest(ROOT/'analysis'/name) for name in (
        'request_failure_attribution.json','request_critical_paths.json','native_relative_causal_comparisons.json')})
    write(ROOT/'analysis/figure_manifest.json',value);preserve();print('NATIVE_SLO_FIGURES',len(manifest),flush=True)
    return value


if __name__=='__main__':build()
