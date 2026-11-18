"""Draw only measured points. Preserve the historical four-panel layout."""
import json
import math
import shutil
from .common import ROOT,REPO,write

STYLE={'Agg':('#d95f02','o'),'Native':('#009e73','^'),
       'Layer':('#0072b2','s'),'Token':('#8c56a0','D')}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=ROOT/'analysis/independent_start_statistics.json'
    if not path.exists():
        print('NO_MEASURED_FIGURE_INPUTS',flush=True)
        return
    data=json.loads(path.read_text())
    aggregate=[r for r in data['aggregate'] if r['count']==144 and r['admission']==8]
    manifest=[]
    def select(method,layers,cap,qps):
        rows=[r for r in aggregate if r['method']==method and r['layers']==layers and r['cap']==cap and r['QPS']==qps]
        confirmation=[r for r in rows if r['stage']=='confirmation' and r['starts']>=3]
        discovery=[r for r in rows if r['stage']=='discovery']
        return (confirmation or discovery or rows or [None])[0]
    def metric(row,key):
        return row['regimes']['Balanced'][key] if key in ('Joint_attainment','Output_token_SLO_goodput') else row['metrics'][key]
    panels=[('Joint_attainment','SLO attainment','Joint SLO attainment (%)',100),
            ('mean_TTFT_ms','Mean TTFT','Mean TTFT (ms)',1),
            ('mean_TBT_ms','Mean TBT','Mean TBT (ms)',1),
            ('output_tokens_s','Generation throughput','Generated output tokens/s',1)]
    for layers in (4,8,12):
        fig,axes=plt.subplots(2,2,figsize=(14,8.3))
        used=[]
        for method in ('Agg','Native','Layer','Token'):
            values=[select(method,layers if method in ('Layer','Token') else 0,200,q) for q in range(2,11)]
            if not any(values):
                continue
            color,marker=STYLE[method]
            label=({'Agg':'Agg DP2','Native':'Native PD','Layer':f'Layer Last{layers}',
                    'Token':f'Matched Token { {2:"432/448",4:"880/896",8:"1776/1792",12:"2656/2672"}[layers]}'}[method])
            for ax,(key,title,ylabel,scale) in zip(axes.flat,panels):
                y=[metric(r,key)['mean']*scale if r is not None else math.nan for r in values]
                ax.plot(range(2,11),y,color=color,marker=marker,lw=1.7,ms=4,label=label)
                for q,row in zip(range(2,11),values):
                    if row is None:
                        continue
                    ci=metric(row,key)['CI95']
                    if ci is not None:
                        value=metric(row,key)['mean']*scale
                        ax.errorbar([q],[value],yerr=[[max(0,value-ci[0]*scale)],[max(0,ci[1]*scale-value)]],
                            color=color,capsize=2,lw=.8)
                ax.set(title=title,xlabel='Offered QPS (requests/s)',ylabel=ylabel)
                ax.set_xticks(range(2,11));ax.grid(alpha=.25)
            used.extend(r for r in values if r is not None)
        if not used:
            plt.close(fig)
            continue
        # Set bounds only after all methods. An early set_ylim disables
        # autoscaling and can hide the slower methods outside the first curve.
        for ax,(key,_,_,_) in zip(axes.flat,panels):
            ax.relim();ax.autoscale_view()
            ax.set_xlim(1.9,10.1);ax.set_xticks(range(2,11))
            ax.set_ylim(-2,102) if key=='Joint_attainment' else ax.set_ylim(bottom=0)
            finite=[float(y) for line in ax.lines for y in line.get_ydata() if math.isfinite(float(y))]
            assert not finite or max(finite)<=ax.get_ylim()[1],key
        fig.suptitle('Qwen3-8B | 2 × H100 | Input8000 / Output200 | Poisson Arrivals\n'
            f'All workers B8192/S1024; common PD admission cap8; Last{layers} matched-work pair',fontsize=13)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=4,fontsize=9)
        fig.text(.025,.02,'N144. Finite-cohort arrival-to-drain rates. Marker = measured point. Missing points are gaps.\n'
            'Discovery n=1. Prespecified QPS6 confirmation n=3, when complete. Pointwise independent-start 95% CI.\n'
            'Balanced SLO: TTFT≤5000ms, every ITL≤34ms, TPOT≤33ms. Evidence: MEASURED.',fontsize=8)
        fig.tight_layout(rect=(0,.16,1,.92))
        stem=f'unified_four_metrics_o200_last{layers}'
        for suffix in ('.png','.pdf'):
            target=ROOT/'figures'/(stem+suffix)
            fig.savefig(target,dpi=180,bbox_inches='tight')
            shutil.copyfile(target,REPO/'poc_flashvep'/('dense_ar_'+stem+suffix))
            if layers==12:
                shutil.copyfile(target,REPO/'poc_flashvep'/('dense_ar_unified_qps_four_metrics_o200'+suffix))
        plt.close(fig)
        manifest.append(dict(stem=stem,evidence='MEASURED',
            points=[dict(method=r['method'],QPS=r['QPS'],starts=r['starts'],seeds=r['seeds'],stage=r['stage']) for r in used]))
    for cap in (16,32,64,128,200):
        for layers in (4,8,12):
            fig,axes=plt.subplots(2,3,figsize=(13,7))
            keys=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s','Joint_attainment','Output_token_SLO_goodput')
            any_data=False
            for method in ('Agg','Native','Layer','Token'):
                points=[select(method,layers if method in ('Layer','Token') else 0,cap,q) for q in range(2,11)]
                if not any(points):
                    continue
                any_data=True
                color,marker=STYLE[method]
                for ax,key in zip(axes.flat,keys):
                    ax.plot(range(2,11),[metric(r,key)['mean'] if r else math.nan for r in points],
                        color=color,marker=marker,ms=3,label=method)
                    for q,row in zip(range(2,11),points):
                        if row is None:continue
                        point=metric(row,key);ci=point['CI95']
                        if ci is not None:
                            ax.errorbar([q],[point['mean']],yerr=[[max(0,point['mean']-ci[0])],
                                [max(0,ci[1]-point['mean'])]],color=color,capsize=2,lw=.7)
                    ax.set(title=key,xlabel='Offered QPS');ax.grid(alpha=.2)
            if any_data:
                axes.flat[0].legend(fontsize=8)
                fig.suptitle(f'Input8000 / Output{cap} / Last{layers} matched-work pair | N144 | MEASURED finite cohorts')
                fig.text(.02,.015,'All B8192/S1024; PD cap8. Discovery n=1. QPS6 confirmation n=3, when complete. Pointwise 95% start-level CI. Missing points are not interpolated.',fontsize=7)
                fig.tight_layout(rect=(0,.035,1,1))
                for suffix in ('.png','.pdf'):
                    fig.savefig(ROOT/'figures'/f'sweep_O{cap}_Last{layers}{suffix}',dpi=160,bbox_inches='tight')
            plt.close(fig)
    comparisons=[r for r in data['comparisons'] if r['reference']=='Token' and r['candidate']=='Layer' and r['admission']==8]
    for qps in sorted({r['QPS'] for r in comparisons}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for layers in (4,8,12):
            points=[]
            for cap in (16,32,64,128,200):
                candidates=[r for r in comparisons if r['layers']==layers and r['cap']==cap and r['QPS']==qps]
                confirmed=[r for r in candidates if r['stage']=='confirmation' and r['starts']>=3]
                chosen=(confirmed or [r for r in candidates if r['stage']=='discovery'] or [None])[0]
                points.append(chosen)
            for ax,kind in zip(axes,('TTFT','SLO')):
                def ratio(row):
                    if row is None:return None
                    return row['metrics']['TTFT_p95_ms'] if kind=='TTFT' else row['regimes']['Common_Balanced_OutputSweep']['Output_token_SLO_goodput']
                values=[ratio(r) for r in points]
                y=[r['ratio'] if r is not None and r['ratio'] is not None else math.nan for r in values]
                line,=ax.plot((16,32,64,128,200),y,marker='o',label=f'Last{layers}')
                for cap,row,value in zip((16,32,64,128,200),points,values):
                    if value is not None and value['ratio_CI95'] is not None:
                        ci=value['ratio_CI95'];center=value['ratio']
                        ax.errorbar([cap],[center],yerr=[[max(0,center-ci[0])],[max(0,ci[1]-center)]],capsize=2,color=line.get_color())
                ax.axhline(1,color='gray',ls='--',lw=.8);ax.set_xlabel('Output tokens');ax.grid(alpha=.2)
        axes[0].set_ylabel('Layer / Token TTFT p95; lower favors Layer')
        axes[1].set_ylabel('Layer / Token common-SLO token/s; higher favors Layer')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'Matched GEMM rows, common B/S and admission | QPS{qps:g} | N144 | MEASURED')
        fig.text(.02,.01,'Discovery n=1. QPS6 confirmation n=3, when complete. Pointwise paired-start 95% CI. Common secondary SLO: 5000/34/33ms.',fontsize=7)
        fig.tight_layout(rect=(0,.04,1,1))
        for suffix in ('.png','.pdf'):
            fig.savefig(ROOT/'figures'/f'output_crossover_q{qps:g}{suffix}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    write(ROOT/'analysis/figure_manifest.json',dict(figures=manifest,
        scope='Historical files are not overwritten. All primary worker B/S gates are required. Different output lengths are never connected as one QPS curve. Error bars use independent starts. Zero-denominator goodput ratios are missing, not fabricated.'))
    print('FIGURES',len(manifest),flush=True)
    from .root_figures import main as root_figures
    root_figures()


if __name__=='__main__':
    main()
