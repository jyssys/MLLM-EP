"""Show measured cells, historical mechanisms and explicit missing points."""
import json
import math
import shutil
import statistics
from .common import ROOT,PREVIOUS,REPO,write,digest

STYLE={'Agg':('#d95f02','o'),'Native':('#009e73','^'),
       'Layer':('#0072b2','s'),'Token':('#8c56a0','D')}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .resampling import mean_interval
    manifest=[]
    def save(fig,stem,evidence,description,sources,copy_to_top=False):
        paths=[]
        for ext in ('.png','.pdf'):
            path=ROOT/'figures'/(stem+ext)
            fig.savefig(path,dpi=175,bbox_inches='tight')
            paths.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
            if copy_to_top:
                shutil.copyfile(path,REPO/'poc_flashvep'/('dense_ar_axis_sweetspot_'+stem+ext))
        plt.close(fig)
        manifest.append(dict(stem=stem,evidence=evidence,description=description,files=paths,sources=sources))
    historical=json.loads((ROOT/'analysis/historical_guard_decomposition.json').read_text())
    raw=[r for r in historical['starts'] if r['stage']=='cohort2_confirmation' and r['method']!='Native']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ('Layer','Token'):
        color,marker=STYLE[method]
        for field,label,linestyle in (('P_submit_to_notification_mean_ms','P submit→ready','-'),
                ('D_cohort_host_occupancy_per_request_ms','D guard / request','--')):
            values=[];errors=[]
            for m in (4,8,12):
                samples=[r[field] for r in raw if r['method']==method and r['layers']==m]
                values.append(statistics.mean(samples));errors.append(mean_interval(samples))
            axes[0].plot((4,8,12),values,color=color,marker=marker,ls=linestyle,label=method+' '+label)
            for m,v,ci in zip((4,8,12),values,errors):
                axes[0].errorbar([m],[v],yerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],color=color,capsize=3)
        ratios=[]
        for m in (4,8,12):
            samples=[r['effective_stage_balance_D_over_P'] for r in raw if r['method']==method and r['layers']==m]
            ratios.append(statistics.mean(samples))
        axes[1].plot((4,8,12),ratios,color=color,marker=marker,label=method)
    axes[1].axhline(1,color='gray',ls='--',lw=1)
    for ax in axes:
        ax.set_xticks((4,8,12));ax.set_xlabel('Offloaded layers / matched Token work');ax.grid(alpha=.25)
    axes[0].set_ylabel('Measured host enclosure (ms/request)');axes[1].set_ylabel('D guard/request divided by P submit→ready')
    axes[0].legend(fontsize=8);axes[1].legend()
    fig.suptitle('HISTORICAL TRACE_ANALYSIS | Input8000/Output16/QPS6 | actual prep1/Decode2 | N144×3')
    fig.text(.01,.005,'Host demand is not exclusive GPU time. The cycle model also needs submission/negotiation gaps. Three independent starts, pointwise95% CI.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    save(fig,'historical_pipeline_balance','TRACE_ANALYSIS','Historical measured host stage demand and a derived balance ratio.',
        [r['name'] for r in raw])
    fig,ax=plt.subplots(figsize=(10,4))
    labels=[];guard=[];partner=[];residual=[]
    for m in (4,8,12):
        for method in ('Layer','Token'):
            values=[r for r in raw if r['method']==method and r['layers']==m]
            total=statistics.mean(r['mean_Ready_to_D_ms'] for r in values)
            g=statistics.mean(r['prior_guard_overlap_ms_per_request'] for r in values)
            p=statistics.mean(r['mean_wait_components_ms'].get('own_partner_only',0.) for r in values)
            labels.append(f'{method}\nLast{m}');guard.append(g);partner.append(p);residual.append(total-g-p)
    x=list(range(len(labels)))
    ax.bar(x,guard,label='Prior cohort guard overlap')
    ax.bar(x,partner,bottom=guard,label='Own partner only')
    ax.bar(x,residual,bottom=[a+b for a,b in zip(guard,partner)],label='Other / uninstrumented')
    ax.set_xticks(x,labels);ax.set_ylabel('Ready→D host wait (ms/request)');ax.legend(fontsize=8)
    ax.set_title('HISTORICAL TRACE_ANALYSIS | Disjoint interval accounting | N144×3')
    fig.text(.01,.01,'Guard spans include transfer, preparation, Decode and delivery. This is neither DMA duration nor physical GPU idle.',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    save(fig,'historical_wait_decomposition','TRACE_ANALYSIS','Exact host-wait identity. No physical GPU activity claim.',[r['name'] for r in raw])

    path=ROOT/'analysis/independent_start_statistics.json'
    data=json.loads(path.read_text()) if path.exists() else dict(aggregate=[],comparisons=[])
    clean=data['aggregate']
    for batch in sorted({r['batch'] for r in clean}):
        for output in sorted({r['output'] for r in clean if r['batch']==batch}):
            for qps in sorted({r['QPS'] for r in clean if r['batch']==batch and r['output']==output}):
                comparisons=[r for r in data['comparisons'] if r['reference']=='Token' and r['candidate']=='Layer'
                    and r['batch']==batch and r['output']==output and r['QPS']==qps and r['window']==32]
                if not comparisons:
                    continue
                points=[]
                for m in (2,4,8,12):
                    rows=[r for r in comparisons if r['layers']==m]
                    preferred=[r for r in rows if r['starts']>=3]
                    points.append((preferred or rows or [None])[0])
                fig,axes=plt.subplots(1,2,figsize=(10,4))
                for ax,key in zip(axes,('TTFT_p95_ms','GOODPUT')):
                    vals=[r['metrics'][key] if key!='GOODPUT' else r['common_Balanced_goodput'] for r in points if r]
                    ms=[m for m,r in zip((2,4,8,12),points) if r]
                    centers=[v['ratio'] if v['ratio'] is not None else math.nan for v in vals]
                    ax.plot(ms,centers,marker='o')
                    for m,v in zip(ms,vals):
                        if v['ratio'] is None:continue
                        ci=v['ratio_CI95']
                        if ci is not None:
                            ax.errorbar([m],[v['ratio']],yerr=[[max(0,v['ratio']-ci[0])],[max(0,ci[1]-v['ratio'])]],capsize=3)
                    ax.axhline(1,color='gray',ls='--');ax.set_xticks((2,4,8,12));ax.set_xlabel('Layer cut / exact matched Token work');ax.grid(alpha=.2)
                axes[0].set_ylabel('Layer / Token TTFT p95 (lower favors Layer)')
                axes[1].set_ylabel('Layer / Token SLO token/s (higher favors Layer)')
                fig.suptitle(f'MEASURED_ONLINE | O{output} / QPS{qps:g} / prep1 / Decode{batch}')
                fig.text(.01,.01,'Points use identical paired starts. n='+','.join(str(r['starts']) if r else 'NOT_RUN' for r in points)+
                    ' for Last2/4/8/12. CI uses server starts. Reused prompts, fresh arrival seeds. Finite cohorts.',fontsize=8)
                fig.tight_layout(rect=(0,.05,1,.94))
                save(fig,f'axis_ratio_O{output}_q{qps:g}_B{batch}','MEASURED_ONLINE','Per-cut matched-axis comparison. No claim for missing points.',
                    [dict(layers=r['layers'],starts=r['starts'],stage=r['stage'],seeds=r['seeds']) for r in points if r])

    prior=json.loads((PREVIOUS/'analysis/independent_start_statistics.json').read_text())
    def select(method,m,q):
        rows=[r for r in clean if r['method']==method and r['layers']==m and r['output']==200
            and r['QPS']==q and r['batch']==1 and r['window']==32 and r['stage']=='residency_discovery']
        return rows[0] if rows else None
    for m in (4,8,12):
        used=[]
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        panels=[('Joint_attainment','Joint SLO attainment (%)',100),('mean_TTFT_ms','Mean TTFT (s)',.001),
            ('mean_TBT_ms','Mean TBT (ms)',1),('output_tokens_s','Generated output tokens/s',1)]
        for method in ('Agg','Native','Layer','Token'):
            points=[]
            for q in range(2,11):
                if method=='Agg':
                    historical=[r for r in prior['aggregate'] if r['method']=='Agg' and r['cap']==200 and r['QPS']==q
                        and r['count']==144 and r['admission']==8 and r['stage']=='discovery']
                    row=historical[0] if historical else None
                else:
                    row=select(method,m if method in ('Layer','Token') else 0,q)
                points.append(row)
            color,marker=STYLE[method]
            label=('Agg DP2 HISTORICAL N144 (unpaired)' if method=='Agg' else method+(' Last'+str(m) if method in ('Layer','Token') else ' PD')+' new N36')
            for ax,(key,ylabel,scale) in zip(axes.flat,panels):
                def center(r):
                    if r is None:return math.nan
                    if key=='Joint_attainment':
                        return (r['regimes']['Common_Balanced_OutputSweep'][key]['mean'] if method=='Agg'
                            else r['common_Balanced'][key]['mean'])*scale
                    return r['metrics'][key]['mean']*scale
                ax.plot(range(2,11),[center(r) for r in points],color=color,marker=marker,
                    ls='--' if method=='Agg' else '-',label=label,ms=5)
                ax.set_xlabel('Offered QPS (requests/s)');ax.set_ylabel(ylabel);ax.set_xticks(range(2,11));ax.grid(alpha=.2)
                ax.set_xlim(1.8,10.2)
            used.extend(dict(method=method,QPS=q,historical=method=='Agg',starts=r['starts'])
                for q,r in zip(range(2,11),points) if r is not None)
        if not any(not r['historical'] for r in used):
            plt.close(fig);continue
        axes.flat[0].set_ylim(-2,102)
        for ax in list(axes.flat)[1:]:ax.set_ylim(bottom=0)
        fig.suptitle('Qwen3-8B | 2 × H100 | Input8000 / Output200 | Poisson Arrivals\n'
            f'B8192/S1024 | Last{m} matched-work pair | PD actual prep1/Decode1 mechanism control',fontsize=12)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,fontsize=8)
        fig.text(.015,.01,'Common frozen SLO: TTFT≤5s, every ITL≤34ms, TPOT≤33ms. New PD n=1, no CI.\n'
            'Missing QPS cells are NOT_RUN. Historical Agg uses a different seed/count and unrestricted continuous batching.\n'
            'This is not a paired four-method performance claim or a sustainable-capacity result.',fontsize=8)
        fig.tight_layout(rect=(0,.17,1,.92))
        save(fig,f'four_metrics_o200_last{m}','MEASURED_ONLINE_WITH_HISTORICAL_CONTEXT',
            'New matched PD controls. Historical Agg is explicitly unpaired and uses different actual batch semantics.',used,True)
    write(ROOT/'analysis/figure_manifest.json',dict(figures=manifest,
        scope='PNG and PDF. No historical artifact is overwritten. Missing cells remain absent. All timelines and host service costs retain their evidence class.'))
    print('SWEETSPOT_FIGURES',len(manifest),flush=True)


if __name__=='__main__':
    main()
