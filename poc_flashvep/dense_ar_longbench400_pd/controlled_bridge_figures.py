"""Plot controlled continuous bridge starts without replacing strong Token."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT
from .confirmed_figures import mean_ci
from .figures import save


def main(suffix='',prefix='controlled_C3_I3',source_note=''):
    data=json.loads((ROOT/'analysis'/('controlled_continuous_I3_bridge'+suffix+'_results_v1.json')).read_text())
    assert data['complete'],'Do not draw partial confirmations as full bridge results'
    cells=data['cells'];x=np.arange(len(cells))
    labels=[f"O{r['controlled_output']} / D-cap{r['D_cap']}" for r in cells]
    colors=dict(NativeStrong='#777777',NativeCommon='#009e73',Layer='#0072b2',StockToken='#cc79a7',
        TokenMode0Control='#8c564b',Hybrid='#e69f00')
    fields=[('TTFT_p95_ms','TTFT p95 (ms)','ttft'),('ITL_p99_ms','Client-visible ITL p99 (ms)','itl'),
        ('TPOT_p95_ms','Request TPOT p95 (ms/token)','tpot'),
        ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','goodput')]
    for field,ylabel,stem in fields:
        fig,ax=plt.subplots(figsize=(10,5.4));records=[]
        for index,arm in enumerate(colors):
            xs=x+(index-2.5)*.07;means=[];cis=[]
            for cell in cells:
                names=cell['arms'][arm]['names']
                values=[json.loads((ROOT/'analysis'/(n+'_metrics.json')).read_text())[field] for n in names]
                mean,ci=mean_ci(values);means.append(mean);cis.append(ci)
            ax.errorbar(xs,means,yerr=[[v-c[0] for v,c in zip(means,cis)],
                [c[1]-v for v,c in zip(means,cis)]],fmt='o',capsize=3,color=colors[arm],label=arm)
            records.append(dict(arm=arm,means=means,CI95=cis))
        ax.set(xticks=x,xticklabels=labels,xlim=(-.4,len(cells)-.6),ylabel=ylabel,
            title='Selected long-input cell | Actual continuous scheduler | QPS1.253873')
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
        save(fig,prefix+'_bridge_'+stem,
            'MEASURED ONLINE CONTROLLED OUTPUT | N16/start | Three paired independent starts/condition on4/5 | BI0,B8192/S1024,KV42GiB.\n'
            'No whole-request C0 barrier. D-cap is an admission limit,not a forced batch. Mode0 Token is a diagnostic control,not the strong compiled Token replacement.\n'
            'Error bars:pointwise27-resample start-level empirical95% CI. Logical iso-rows do not imply equal padding or actual batches.\n'
            'Same prompt set selected from a C0 candidate. Not natural400,sustainable capacity,global best-static performance,or dynamic selection.'+source_note,
            dict(evidence='MEASURED_ONLINE_CONTROLLED_OUTPUT',cells=labels,metric=field,points=records,
                 source_capsule_SHA256=data['source_capsule_SHA256']))
    fig,ax=plt.subplots(figsize=(10,5.2))
    for offset,ref,color in [(-.07,'StockToken','#cc79a7'),(.07,'TokenMode0Control','#8c564b')]:
        for i,cell in enumerate(cells):
            row=cell['comparisons']['Hybrid'][ref]['ratios']['TTFT_p95_ms'];v=row['estimate'];lo,hi=row['CI95']
            ax.errorbar(i+offset,v,yerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=color,
                label='Hybrid / '+ref if i==0 else None)
    ax.axhline(1,color='gray',ls='--');ax.set(xticks=x,xticklabels=labels,
        ylabel='Hybrid / Token TTFT p95 ratio',title='Keep the strong compiled Token and the Mode0 control separate')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,prefix+'_hybrid_token_runtime_bridge',
        'MEASURED ONLINE CONTROLLED OUTPUT | Three paired independent starts per cell | Pointwise95% empirical CI,27 resamples.\n'
        'Mode0/Stock comparison includes adapter and numerical-path differences. No single-compile-flag or GPU-exclusive attribution.\n'
        'Every serving claim also needs both Native decode gates. The older C0 result used different source,pair and seeds;no paired C0→C3 CI.'+source_note,
        dict(evidence='MEASURED_ONLINE_CONTROLLED_OUTPUT',cells=cells,
             source_capsule_SHA256=data['source_capsule_SHA256']))


if __name__=='__main__':
    main()
