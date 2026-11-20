"""Plot measured points and start-level intervals without filling missing arms."""
import collections
import csv
import itertools
import json
import shutil
import numpy as np
from .common import ROOT,REPO,write,digest


def interval(values):
    values=np.asarray(values,dtype=float)
    if len(values)<3:return None
    if len(values)<=6:draws=[float(values[list(i)].mean()) for i in itertools.product(range(len(values)),repeat=len(values))]
    else:draws=np.random.default_rng(2026112099).choice(values,(10000,len(values)),replace=True).mean(axis=1)
    return [float(np.percentile(draws,p)) for p in (2.5,97.5)]


def draw_ci_points(ax,x,low,high,color):
    """Show isolated intervals when neighboring one-start cells have no CI."""
    x=np.asarray(x,dtype=float);low=np.asarray(low,dtype=float);high=np.asarray(high,dtype=float)
    valid=np.isfinite(x)&np.isfinite(low)&np.isfinite(high)
    if not np.any(valid):return None
    assert np.all(high[valid]>=low[valid])
    center=(low[valid]+high[valid])/2;radius=(high[valid]-low[valid])/2
    return ax.errorbar(x[valid],center,yerr=radius,fmt='none',ecolor=color,capsize=3,
        elinewidth=1.2,zorder=4,label='_nolegend_')


def finish(fig,stem,manifest,scope,inputs):
    for suffix in ('png','pdf'):
        path=ROOT/'figures'/f'{stem}.{suffix}';fig.savefig(path,dpi=180,bbox_inches='tight')
        manifest.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path),scope=scope,inputs=inputs))


def measured_group(pattern):
    result=collections.defaultdict(list);excluded=[]
    for p in sorted((ROOT/'analysis').glob(pattern)):
        row=json.loads(p.read_text());name=row['name'];cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
        quality=ROOT/'analysis'/f'{name}_quality_gate.json'
        gate=json.loads(quality.read_text()) if quality.exists() else {}
        key=(cfg['method']+str(cfg.get('axis_cut','')),cfg['offered_QPS'])
        result[key].append(dict(metrics=row,configuration=cfg,name=name,quality=gate,source=p))
    for key,rows in list(result.items()):
        if not all(r['quality'].get('CI_passed') for r in rows):
            excluded.append(dict(cell=list(key),names=[r['name'] for r in rows],reason='AT_LEAST_ONE_START_FAILED_FULL200_TASK_QUALITY_CI'))
            del result[key]
    return result,excluded


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'pdf.fonttype':42})
    manifest=[]
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    reference=json.loads((ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json').read_text())['rows']
    by_id={r['prompt_id']:r for r in reference};tasks=sorted({r['task'] for r in data})
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for task in tasks:
        rows=[r for r in data if r['task']==task];lengths=[r['input_tokens'] for r in rows];outs=[by_id[r['prompt_id']]['output_tokens'] for r in rows]
        axes[0].hist(lengths,bins=np.linspace(0,40000,17),histtype='step',label=task)
        axes[1].hist(outs,bins=np.linspace(0,160,17),histtype='step',label=task)
        axes[2].scatter(lengths,outs,s=18,alpha=.7,label=task)
    for axis,label in zip(axes,('Actual input tokens','BI0 natural output tokens','Actual input tokens')):axis.set_xlabel(label)
    axes[0].set_ylabel('Requests');axes[1].set_ylabel('Requests');axes[2].set_ylabel('BI0 natural output tokens');axes[2].legend(fontsize=8)
    from matplotlib.ticker import FuncFormatter
    for axis in (axes[0],axes[2]):axis.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
    fig.suptitle('LongBench-ShortAnswer200-v2 | 40 requests/task | Reference concurrency1')
    fig.tight_layout();finish(fig,'bi0_shortanswer200_input_output_distribution',manifest,'MEASURED_NATURAL_OUTPUT_REFERENCE, not open-loop performance',['data/longbench_shortanswer200_v2.json','raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json']);plt.close(fig)
    opportunity=ROOT/'analysis/BI0_P_only_ContinuousNative_observer_v2.json'
    if opportunity.exists():
        audit=json.loads(opportunity.read_text());name=audit['names']['profile'];folder=ROOT/'raw'/name
        left,right=audit['kernel_activity']['common_host_window'];origin=json.loads((folder/'window.json').read_text())['start']
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
        bins=audit['kernel_activity']['roles']['prefill']['kernel_union_bins_1s']
        axes[0].bar([left-origin+b['start_seconds'] for b in bins],
            [100*b['kernel_union_seconds']/(b['end_seconds']-b['start_seconds']) for b in bins],width=1.,align='edge',color='#377eb8')
        axes[0].set_ylim(0,105);axes[0].set_ylabel('P kernel busy (%)')
        for role,key,label,color in (('prefill','waiting_total','P waiting','#d95f02'),('decode','active_decode','D active decode','#009e73'),
                                     ('decode','remote_KV_waiting','D remote-KV wait','#984ea3'),('decode','decode_ready_waiting','D decode-ready wait','#333333')):
            rows=json.loads((folder/f'scheduler_queues_{role}.json').read_text())['rows']
            rows=[r for r in rows if left<=r['time']<=right]
            axes[1].step([r['time']-origin for r in rows],[r[key] for r in rows],where='post',label=label,color=color)
        axes[1].set_ylabel('Requests');axes[1].legend(fontsize=8,ncol=2)
        quiet=json.loads((folder/'D_quiescent_witness.json').read_text())['rows'];durations=[]
        for row in quiet:
            a=max(left,row['start']);b=min(right,row['end'])
            if b<=a:continue
            axes[2].broken_barh([(a-origin,b-a)],(.25,.5),facecolors='#999999');durations.append((b-a)*1000)
        axes[2].set_ylim(0,1);axes[2].set_yticks([.5],['D host-empty']);axes[2].set_xlabel('Time since measurement start (s)')
        axes[2].set_xlim(left-origin,right-origin)
        fig.suptitle(f'BI0 continuous Native | QPS2.35 | P-only observer gate {"PASS" if audit["observer_gate_passed"] else "FAIL"}')
        fig.text(.01,-.015,'P: CUPTI kernel union. D: completed current-stream event and empty scheduler, not D kernel activity or a predicted safe budget.',fontsize=8)
        fig.tight_layout();finish(fig,'bi0_shortanswer200_P_kernel_D_host_state_timeline',manifest,
            'MEASURED_P_KERNEL_ACTIVITY_AND_D_HOST_STATE, not active-decode window proof',[str(opportunity.relative_to(ROOT))]);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4));ax.hist(durations,bins=30,color='#999999');ax.set_xlabel('Observed D host-empty interval (ms)');ax.set_ylabel('Intervals')
        ax.set_title('Stream-completed host quiescence | NOT a decode-safe-window distribution')
        fig.text(.01,-.02,f'n={len(durations)} intervals in one profiled server start. Future budget and all-stream D kernel gaps are not measured.',fontsize=8)
        finish(fig,'bi0_shortanswer200_D_host_quiescence_distribution',manifest,'TRACE_ANALYSIS, not safe-fit measurement',[str(opportunity.relative_to(ROOT))]);plt.close(fig)
    for filename,field,label,stem in (('BI0_isolated_axis_cost_summary.json','Token_over_Layer','Token / Layer full-forward CUDA enclosure','bi0_layer_token_forward_ratio'),
                                     ('BI0_GEMM_API_summary.json','Token_over_Layer','Token / Layer projection-only graph enclosure','bi0_layer_token_GEMM_ratio')):
        source=ROOT/'analysis'/filename
        if not source.exists():continue
        pairs=json.loads(source.read_text())['pairs'];fig,ax=plt.subplots(figsize=(8,4.5))
        for length in sorted({r['input_tokens'] for r in pairs}):
            rows=[r for r in pairs if r['input_tokens']==length];x=[r['cut'] for r in rows];y=np.array([r[field] for r in rows]);bounds=[r['paired_start_empirical_CI95'] for r in rows]
            line=ax.plot(x,y,'o-',label=f'Input{length}, paired starts n=3')[0]
            if all(v is not None for v in bounds):ax.fill_between(x,[v[0] for v in bounds],[v[1] for v in bounds],color=line.get_color(),alpha=.15)
        ax.axhline(1,color='black',linestyle='--',linewidth=1);ax.set_xticks([1,2,4,6,8,12]);ax.set_xlabel('Offloaded layers m; matched Token suffix L*m/36');ax.set_ylabel(label);ax.legend()
        ax.set_title('BI0 | Actual batch1 | Exact matched GEMM rows | Not TTFT')
        fig.text(.02,-.02,'Clean event enclosures. Kernel attribution requires a separate <=5% observer gate. Shading: empirical start bootstrap95% CI.',fontsize=8)
        finish(fig,stem,manifest,'ISOLATED_FORWARD_MEASUREMENT or ISOLATED_GEMM_API_MEASUREMENT, not serving speedup',[str(source.relative_to(ROOT))]);plt.close(fig)
    groups,excluded=measured_group('Main_*_ShortAnswer200_*_BI0_v3_metrics.json')
    fields=[('TTFT_p95_ms','TTFT p95 (ms)','TTFT_p95'),('ITL_p99_ms','Client-visible ITL p99 (ms)','ITL_p99'),
            ('TPOT_p95_ms','Request TPOT p95 (ms/token)','TPOT_p95'),
            ('generated_output_tokens_per_second','Generated output tokens/s','output_throughput'),
            ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','SLO_goodput'),
            ('Joint_SLO_attainment','Joint SLO attainment','SLO_attainment')]
    points=[]
    if groups:
        for field,ylabel,stem in fields:
            fig,ax=plt.subplots(figsize=(8,4.8))
            for method,color,marker in (('Agg','#d95f02','o'),('Native','#009e73','^')):
                cells=sorted((q,rows) for (m,q),rows in groups.items() if m==method)
                if not cells:continue
                x=[];means=[];low=[];high=[]
                for q,rows in cells:
                    values=[r['metrics'][field] for r in rows];mean=float(np.mean(values));ci=interval(values)
                    x.append(q);means.append(mean);low.append(ci[0] if ci else np.nan);high.append(ci[1] if ci else np.nan)
                    ax.annotate(f'n={len(rows)}',(q,mean),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
                    if field==fields[0][0]:points.append(dict(method=method,QPS=q,starts=len(rows),names=[r['name'] for r in rows]))
                ax.plot(x,means,marker=marker,color=color,label='Agg DP2' if method=='Agg' else 'Native PD1P1D')
                ax.fill_between(x,low,high,color=color,alpha=.15)
                draw_ci_points(ax,x,low,high,color)
            if field=='Joint_SLO_attainment':ax.set_ylim(0,1.05)
            ax.set_xlabel('Offered QPS (requests/s)');ax.set_ylabel(ylabel);ax.legend()
            ax.set_title('BI0 ShortAnswer200-v2 | 2 x H100 | B8192/S1024 | KV16')
            fig.text(.02,-.015,'Natural EOS. Arrival-to-drain finite cohorts, not sustainable capacity. CI only for >=3 independent starts.',fontsize=8)
            finish(fig,f'bi0_shortanswer200_{stem}_vs_qps',manifest,'MEASURED_ONLINE; ordinary continuous baselines',points)
            if stem in ('TTFT_p95','ITL_p99','TPOT_p95','output_throughput'):
                for suffix in ('png','pdf'):shutil.copyfile(ROOT/'figures'/f'bi0_shortanswer200_{stem}_vs_qps.{suffix}',REPO/'poc_flashvep'/f'bi0_shortanswer200_{stem}_vs_qps.{suffix}')
            plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(field,label,scale) in zip(axes.flat,(('Joint_SLO_attainment','Joint SLO attainment (%)',100),
                ('TTFT_mean_ms','Mean TTFT (ms)',1),('mean_TBT_ms','Mean TBT (ms)',1),
                ('generated_output_tokens_per_second','Generated output tokens/s',1))):
            for method,color,marker in (('Agg','#d95f02','o'),('Native','#009e73','^')):
                cells=sorted((q,rows) for (m,q),rows in groups.items() if m==method)
                x=[];y=[];lo=[];hi=[]
                for rate,rows in cells:
                    values=[r['metrics'][field]*scale for r in rows];bounds=interval(values)
                    x.append(rate);y.append(np.mean(values));lo.append(bounds[0] if bounds else np.nan);hi.append(bounds[1] if bounds else np.nan)
                ax.plot(x,y,color=color,marker=marker,label='Agg DP2' if method=='Agg' else 'Native PD1P1D');ax.fill_between(x,lo,hi,color=color,alpha=.15)
                draw_ci_points(ax,x,lo,hi,color)
            ax.set_xlabel('Offered QPS (requests/s)');ax.set_ylabel(label)
            if scale==100:ax.set_ylim(0,105)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.005))
        fig.suptitle('Qwen3-8B BI0 | 2 x H100 | LongBench-ShortAnswer200-v2 | Natural EOS | KV16')
        fig.text(.01,-.035,'n=1 at QPS0.4/1.2; n=3 at QPS2.35. Bars/shading: start bootstrap95% CI. SLO: TTFT5s/every ITL34ms/TPOT33ms.',fontsize=8)
        fig.text(.01,-.058,'B8192/S1024. Finite-cohort output-only throughput, not sustainable capacity. No Layer/Token or Dynamic result is implied.',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,.96));finish(fig,'bi0_shortanswer200_four_metrics_confirmed',manifest,'MEASURED_ONLINE; continuous baselines only',points)
        for suffix in ('png','pdf'):shutil.copyfile(ROOT/'figures'/f'bi0_shortanswer200_four_metrics_confirmed.{suffix}',REPO/'poc_flashvep'/f'bi0_shortanswer200_four_metrics_confirmed.{suffix}')
        plt.close(fig)
    write(ROOT/'figures/figure_manifest.json',dict(figures=manifest,excluded=excluded,
        evidence='Each figure records its measurement scope.',
        missing='No missing QPS, static or dynamic arm is filled. No one-start confidence interval is invented.'))
    return manifest


if __name__=='__main__':main()
