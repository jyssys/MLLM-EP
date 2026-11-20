"""Draw measured points. Keep discovery, confirmation and diagnostics separate."""
import csv
import json
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write

COLORS={'Layer':'#0072B2','Token':'#D55E00','Native':'#009E73','Agg':'#CC79A7'}


def read(name):
    path=ROOT/'analysis'/name
    return json.loads(path.read_text()) if path.exists() else {}


def save(fig,name,scope):
    fig.text(.01,.01,scope,fontsize=8,va='bottom',wrap=True)
    fig.tight_layout(rect=(0,.09,1,1))
    for extension in ('png','pdf'):
        fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
    plt.close(fig)
    return dict(name=name,evidence_scope=scope,files=[f'figures/{name}.{e}' for e in ('png','pdf')])


def pilot_curves():
    rows=read('pilot50_all_cuts_qps_v1.json').get('results',[]);rows=[r for r in rows if 'metrics' in r]
    result=[]
    for field,label,suffix,scale in [('TTFT_p95_ms','TTFT p95 (s)','ttft',.001),
        ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','goodput',1.),
        ('ITL_p99_ms','Client ITL p99 (ms)','itl',1.),('TPOT_p95_ms','Request TPOT p95 (ms/token)','tpot',1.),
        ('generated_output_tokens_per_second','Generated output tokens/s','throughput',1.)]:
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
        for ax,cut in zip(axes.flat,(2,4,8,12)):
            for axis in ('Layer','Token','Native'):
                values=sorted([r for r in rows if r['axis']==axis and (r['cut']==cut or axis=='Native')],key=lambda r:r['QPS'])
                if not values:continue
                measured={r['QPS']:r for r in values}
                x=[.4,.7,1.,1.2,1.65,2.,2.35]
                y=[measured[q]['metrics'][field]*scale if q in measured else np.nan for q in x]
                ax.plot(x,y,color=COLORS[axis],label=axis,marker='o',linewidth=1.5)
                bad=[r for r in values if not r['quality']['CI_passed']]
                ax.scatter([r['QPS'] for r in bad],[r['metrics'][field]*scale for r in bad],facecolors='white',edgecolors=COLORS[axis],zorder=5)
            ax.set(title=f'Last{cut} / matched token fraction {cut}/36',ylabel=label,xlabel='Offered QPS (requests/s)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        result.append(save(fig,'pilot50_cut_qps_'+suffix,
            'DIAGNOSTIC online discovery | N50,1 independent start/cell,no start-level CI. All measured cells fail the frozen task-quality CI gate and use hollow markers.\n'
            'BI0 | B8192/S1024 | C0 actual1/1/1 | c4,h4,KV128 | Natural EOS cap2048. Only measured QPS points. Finite cohorts, not sustainable capacity.'))
    return result


def confirmation():
    rows=read('full200_confirmation_statistics.json').get('comparisons',[])
    fig,axes=plt.subplots(1,2,figsize=(10,4));fields=['TTFT_p95_ms','SLO_qualified_output_tokens_per_second']
    for ax,field,title in zip(axes,fields,['Layer / Token TTFT p95','Layer / Token SLO-goodput']):
        for row in rows:
            v=row['metrics'][field];center=v['ratio'];lo,hi=v['CI95']
            ax.errorbar(row['cut'],center,yerr=[[center-lo],[hi-center]],fmt='o',capsize=5,color=COLORS['Layer'])
        ax.axhline(1,color='black',ls='--',lw=1);ax.set(xlabel='Fixed layer cut',ylabel=title,xticks=[2,4,8,12]);ax.grid(alpha=.2)
    return [save(fig,'full200_highload_paired_confirmation',
        'MEASURED_ONLINE | Full200 | QPS2.35 |3 independent starts/cell. Exact27 paired start bootstrap resamples,95% empirical CI.\n'
        'BI0 | C0 actual1/1/1 | c4,h4,KV128 | Natural EOS. Only completed3-start cuts appear. A TTFT interval alone is not a Layer-serving GO. Finite cohorts.')]


def credits():
    rows=read('credit_sweep_v1.json').get('results',[]);rows=[r for r in rows if 'metrics' in r]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,field,label,scale in zip(axes,['TTFT_p95_ms','P_queue_ms','SLO_qualified_output_tokens_per_second'],
        ['TTFT p95 (s)','Mean P queue to first prepare (s)','SLO-qualified output tokens/s'],[.001,.001,1.]):
        for axis in ('Native','Layer','Token'):
            values=sorted([r for r in rows if r['axis']==axis],key=lambda r:r['credit'])
            ys=[]
            for r in values:
                timeline=read(r['name']+'_axis_timeline.json')
                ys.append((timeline['summary'][field]['mean'] if field=='P_queue_ms' else r['metrics'][field])*scale)
            ax.plot([r['credit'] for r in values],ys,'o-',color=COLORS[axis],label=axis)
        ax.set(xlabel='Source outstanding credit',ylabel=label,xscale='log',xticks=[1,4,8,16,32]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    return [save(fig,'source_credit_intervention',
        'MEASURED_ONLINE one-factor discovery | Full200,1 start/cell | Last8 | QPS2.35,seed2083 | Actual1/1/1.\n'
        'Source credit is not actual batch or hidden capacity. Fixed Layer hidden pool32 was selected by memory/state feasibility. This c4 is not pooled with h4 confirmation.')]


def buckets():
    rows=read('input_bucket_online_controls_v1.json').get('results',[]);rows=[r for r in rows if 'metrics' in r]
    matrix=np.full((3,4),np.nan)
    for i,length in enumerate((8192,16384,32768)):
        for j,cut in enumerate((2,4,8,12)):
            arms={r['axis']:r for r in rows if r['center']==length and r['cut']==cut}
            if len(arms)==2:matrix[i,j]=arms['Layer']['metrics']['TTFT_p95_ms']/arms['Token']['metrics']['TTFT_p95_ms']
    fig,ax=plt.subplots(figsize=(7,4));masked=np.ma.masked_invalid(matrix)
    im=ax.imshow(masked,cmap='coolwarm',vmin=.8,vmax=1.2,aspect='auto')
    ax.set(xticks=range(4),xticklabels=[2,4,8,12],yticks=range(3),yticklabels=['8192 bucket','16384 bucket','32768 bucket'],xlabel='Fixed layer cut')
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:.3f}' if np.isfinite(v) else 'NOT_RUN',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Layer / Token TTFT p95')
    return [save(fig,'input_bucket_axis_ratio',
        'DIAGNOSTIC fixed-output control | Original unpadded inputs in frozen length buckets, not exact center lengths. N<=8,1 start/cell, no start-level CI.\n'
        'BI0 | B8192/S1024 | C0 actual1/1/1 | c4,h4,KV128 | Per-request C1 output counts. Nonzero aggregate row mismatch is reported separately.')]


def source_kernels():
    rows=read('P_kernel_profile_qualification_summary.json').get('results',[])
    qualified=[r for r in rows if r['qualified'] and r['mode'] in ('ONLINE','SERIAL_ADMISSION')]
    fig,ax=plt.subplots(figsize=(11,5))
    if not qualified:
        ax.axis('off');ax.text(.5,.5,'NOT_MEASURED_PRIMARY\nNo profile passes all observer and attribution gates.',ha='center',va='center',transform=ax.transAxes)
    else:
        keys=sorted({k for r in qualified for k in r['kernel_measurement']['P_named_category_means']});bottom=np.zeros(len(qualified))
        for key in keys:
            values=np.asarray([r['kernel_measurement']['P_named_category_means'].get(key,0) for r in qualified])
            ax.bar(range(len(qualified)),values,bottom=bottom,label=key);bottom+=values
        ax.set(xticks=range(len(qualified)),xticklabels=[r['axis']+'\n'+r['mode'] for r in qualified],ylabel='P request-attributed kernel union mean (ms)')
        ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    return [save(fig,'qualified_source_kernel_breakdown',
        'MEASURED_ONLINE only if every profile gate passes: <=5% observer change,>=99% API-correlated kernel-time coverage, matching queries/outputs, valid counters/bounds.\n'
        'Matched Last4/Fixed4723-output Full200 serial/online groups only. An unpaired qualified Natural Token subsample is excluded. Named categories are not SM/HBM measurements. D graph-exclusive time is not inferred.')]


def serial_online():
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,cut in zip(axes,(4,8)):
        filename='serial_online_service_vector_replay_v1.json' if cut==4 else 'Last8_serial_online_service_vector_replay_v1.json'
        summaries=read(filename).get('summaries',{})
        for axis in ('Layer','Token'):
            values=[summaries.get(f'{axis}_{mode}_c4') for mode in ('SERIAL_ADMISSION','ONLINE')]
            x=[0,1]
            ys=[v['P_plus_D_first_mean_ms'] if v else np.nan for v in values]
            ax.plot(x,ys,'o-',color=COLORS[axis],label=axis)
        ax.set(xticks=[0,1],xticklabels=['Serial admission','Online admission'],ylabel='Mean P host + D-first host service (ms)',title=f'Cut{cut},full200 same-output diagnostic')
        ax.grid(alpha=.2);ax.legend()
    return [save(fig,'matched_serial_online_service',
        'MEASURED_ONLINE DIAGNOSTIC | Full200,4723 fixed output tokens | Same cut,requests,seed2082/QPS2.35 |1 start/mode/axis | C0 actual1/1/1,c4,h4.\n'
        'These are host service envelopes, not TTFT or GPU-exclusive time. Serial admission preserves the same external Poisson trace. Cut4 and cut8 remain separate.')]


def winner_map():
    rows=read('pilot50_all_cuts_qps_v1.json').get('results',[]);rows=[r for r in rows if 'metrics' in r]
    rates=[.4,.7,1.,1.2,1.65,2.,2.35];matrix=np.full((4,len(rates)),np.nan)
    for i,cut in enumerate((2,4,8,12)):
        for j,rate in enumerate(rates):
            arms={r['axis']:r for r in rows if r['cut']==cut and r['QPS']==rate}
            if len(arms)==2:matrix[i,j]=arms['Layer']['metrics']['TTFT_p95_ms']/arms['Token']['metrics']['TTFT_p95_ms']
    fig,ax=plt.subplots(figsize=(10,4));im=ax.imshow(np.ma.masked_invalid(matrix),cmap='coolwarm',vmin=.8,vmax=1.2,aspect='auto')
    ax.set(xticks=range(len(rates)),xticklabels=rates,yticks=range(4),yticklabels=[2,4,8,12],xlabel='Offered QPS',ylabel='Fixed layer cut')
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'NOT_RUN',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Layer / Token TTFT p95')
    return [save(fig,'pilot50_cut_qps_ratio_heatmap',
        'DIAGNOSTIC DISCOVERY | N50,1 independent start/cell,no winner CI. All measured cells fail the frozen task-quality CI gate. Ratios are not qualified winner labels.\n'
        'BI0 | C0 actual1/1/1,c4,h4,KV128 | Natural EOS cap2048. Quality gates remain in the complete matrix. Missing cells remain NOT_RUN.')]


def decode_protection():
    rows=read('full200_confirmation_statistics.json').get('comparisons',[])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,field in zip(axes,['ITL_p99_ms','TPOT_p95_ms']):
        for axis,offset in [('Layer',-.15),('Token',.15)]:
            for index,row in enumerate(rows):
                value=row['decode_protection'][axis]['metrics'][field];center=value['ratio'];lo,hi=value['CI95']
                ax.errorbar(row['cut']+offset,center,yerr=[[center-lo],[hi-center]],fmt='o',capsize=4,
                    color=COLORS[axis],label=axis if index==0 else None)
        ax.axhline(1.05,color='red',ls='--',label='5% noninferiority bound');ax.axhline(1,color='black',lw=.7)
        ax.set(xticks=[2,4,8,12],xlabel='Fixed cut',ylabel=field+' / Native');ax.grid(alpha=.2);ax.legend(fontsize=8)
    return [save(fig,'native_relative_decode_noninferiority',
        'MEASURED_ONLINE | Full200,QPS2.35 |3 independent starts/cell | Exact27 paired start bootstrap resamples,95% empirical CI.\n'
        'ITL p99 and TPOT p95 must each have CI upper bound<=1.05. Relative safety and absolute TTFT/34ms-ITL/33ms-TPOT SLOs are different tests.')]


def continuous_bridge():
    rows=read('C3_continuous_bridge_results_v1.json').get('results',[])
    rows=[r for r in rows if r.get('status')=='MEASURED_ONLINE'];fig,axes=plt.subplots(1,2,figsize=(11,5))
    names=[];ttft=[];throughput=[]
    for row in rows:
        cfg=json.loads((ROOT/'raw'/row['name']/'configuration.json').read_text())
        names.append(row['axis']+'\n'+cfg['execution_family']);ttft.append(row['metrics']['TTFT_p95_ms']/1000)
        throughput.append(row['metrics']['generated_output_tokens_per_second'])
    for ax,values,label in zip(axes,[ttft,throughput],['TTFT p95 (s)','Generated output tokens/s']):
        ax.bar(range(len(values)),values,color=[COLORS[r['axis']] for r in rows]);ax.set(xticks=range(len(values)),xticklabels=names,ylabel=label)
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
        if not rows:ax.text(.5,.5,'NOT_RUN',ha='center',va='center',transform=ax.transAxes)
    return [save(fig,'continuous_baseline_bridge',
        'MEASURED_ONLINE baseline bridge | Full200,QPS2.35,seed2071 |1 start/configuration,no start-level CI | Common B8192/S1024,KV42GiB/GPU.\n'
        'Normal continuous batching is not actual1/1/1 matching. Compilation families remain separate. C3 axis qualification is reported independently. No missing axis performance is fabricated.')]


def destination_kernels():
    rows=read('D_suffix_profile_qualification_summary.json').get('results',[])
    rows+=read('D_kernelonly_suffix_profile_qualification_summary.json').get('results',[])
    good=[r for r in rows if r['qualified']];fig,ax=plt.subplots(figsize=(8,4))
    if not good:ax.axis('off');ax.text(.5,.5,'NOT_MEASURED_PRIMARY\nNo destination suffix profile passes all gates.',ha='center',va='center',transform=ax.transAxes)
    else:
        keys=sorted({k for r in good for k in r['kernel_measurement']['D_named_category_means']});bottom=np.zeros(len(good))
        for key in keys:
            values=np.asarray([r['kernel_measurement']['D_named_category_means'].get(key,0) for r in good]);ax.bar(range(len(good)),values,bottom=bottom,label=key);bottom+=values
        ax.set(xticks=range(len(good)),xticklabels=[r['axis'] for r in good],ylabel='D eager suffix kernel union mean (ms)');ax.legend(fontsize=7)
    return [save(fig,'qualified_destination_suffix_kernel_breakdown',
        'MEASURED_ONLINE only for profiles that pass all7 observer metrics,strict shapes/outputs and API attribution gates. Same Last4/Natural200/seed2061.\n'
        'Only prompt-query steps enter suffix costs. Decode-graph enclosures are not decode-exclusive kernel time. Transport/host gaps are not silently added to kernel categories.')]


def block_sensitivity():
    rows=read('KV_block_sensitivity_v1.json').get('results',[])
    rows=[r for r in rows if r.get('metrics')]
    fig,ax=plt.subplots(figsize=(8,4))
    for axis,offset in [('Layer',-.16),('Token',.16)]:
        values=[r for r in rows if r['axis']==axis]
        blocks={r.get('block',r.get('KV_block')):r for r in values}
        if not all(block in blocks for block in (16,128)):
            blocks={json.loads((ROOT/'raw'/r['name']/'configuration.json').read_text())['KV_block_tokens']:r for r in values}
        for index,block in enumerate((16,128)):
            if block in blocks:ax.bar(index+offset,blocks[block]['metrics']['TTFT_p95_ms']/1000,width=.3,color=COLORS[axis],label=axis if index==0 else None)
    ax.set(xticks=[0,1],xticklabels=['KV16','KV128'],ylabel='TTFT p95 (s)',title='Same Natural200 / Last8 / QPS2.35 / seed2089')
    ax.legend();ax.grid(axis='y',alpha=.2)
    return [save(fig,'KV_block_one_factor_sensitivity',
        'MEASURED_ONLINE one-factor discovery |1 independent start/block/axis,no start-level CI | BI0,B8192/S1024,C0 actual1/1/1,c4,h4.\n'
        'Both axes use the same block. Fresh KV16 state and lifetime gates precede performance. This is not the separate fixed-output Historical workload.')]


def attention_placement():
    rows=read('BI0_axis_structural_accounting.json').get('totals',[])
    rows=[r for r in rows if r['cut'] in (2,4,8,12)]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r['cut'] for r in rows],[r['Token_D_attention_pairs']/r['Layer_D_attention_pairs'] for r in rows],'o-',color=COLORS['Token'])
    ax.set(xticks=[2,4,8,12],xlabel='Fixed layer cut',ylabel='Token / Layer D-side causal attention pairs')
    ax.grid(alpha=.2)
    return [save(fig,'structural_D_attention_placement',
        'STRUCTURAL_ACCOUNTING | Frozen Natural200 input lengths and nearest16 prefix plan. Aggregate GEMM rows are near-matched,not exact.\n'
        'Logical causal pair counts are not measured FLOPs,kernel duration,HBM traffic or a serving speedup. Total P+D attention pairs are equal. Their placement differs.')]


def confirmed_host_demand():
    rows=read('full200_confirmation_host_service_demand.json').get('rows',[])
    if not rows:return []
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,field,label in zip(axes,['P_host_service_ms','D_whole_host_demand_ms'],
        ['P dispatch to ready host demand (ms)','D submit to completion host demand (ms)']):
        for axis in ('Layer','Token','Native'):
            part=sorted([r for r in rows if r['axis']==axis],key=lambda r:r['cut'])
            means=np.asarray([r['summary'][field]['mean'] for r in part])
            lower=np.asarray([r['summary'][field]['CI95'][0] for r in part])
            upper=np.asarray([r['summary'][field]['CI95'][1] for r in part])
            ax.errorbar([r['cut'] for r in part],means,yerr=[means-lower,upper-means],fmt='o-',capsize=4,color=COLORS[axis],label=axis)
        ax.set(xticks=[2,4,8,12],xlabel='Fixed layer cut',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    return [save(fig,'confirmed_P_D_host_service_demand',
        'TRACE_ANALYSIS | Natural200,QPS2.35,C0 actual1/1/1,c4,h4,KV128 | Three independent starts/cell,27 start-bootstrap draws,95% CI.\n'
        'These are host envelopes,not GPU-exclusive cost or physical idle. D includes decode residency. The same Native three starts are reused across cuts.')]


def main():
    result=[]
    for function in (pilot_curves,confirmation,credits,buckets,source_kernels,serial_online,winner_map,decode_protection,continuous_bridge,destination_kernels,block_sensitivity,attention_placement,confirmed_host_demand):result.extend(function())
    write(ROOT/'analysis/figure_manifest.json',dict(figures=result,
        scope='No missing measurement is interpolated as data. Confidence intervals use independent server starts only. Diagnostic and primary scopes are separate.'))
    print('FIGURES',len(result),flush=True)


if __name__=='__main__':main()
