"""Plot all eight exact-row hybrid confirmations, including non-wins."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT
from .figures import save


def main():
    data=json.loads((ROOT/'analysis/iso8_hybrid_heldout_confirmation_v1.json').read_text())
    assert data['completed_cells']==8
    cells=data['cells'];x=np.arange(8)
    labels=[f"I{r['input_bucket']} / O{r['controlled_output']}" for r in cells]
    specifications=[('TTFT_p95_ms','Hybrid / Pure-axis TTFT p95','ttft',('Layer','Token')),
        ('SLO_qualified_output_tokens_per_second','Hybrid / Reference SLO-goodput','goodput',('Layer','Token','Native'))]
    for field,ylabel,stem,refs in specifications:
        fig,ax=plt.subplots(figsize=(12,5))
        for offset,ref,color in zip(np.linspace(-.12,.12,len(refs)),refs,('#0072b2','#cc79a7','#009e73')):
            for i,cell in enumerate(cells):
                row=cell['comparisons']['Hybrid_over_'+ref][field]
                if row is None:continue
                v=row['estimate'];lo,hi=row['CI95']
                ax.errorbar(i+offset,v,yerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color=color,
                    label='Hybrid / '+ref if i==0 else None)
        ax.axhline(1,color='gray',ls='--');ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,
            title='All eight exact-row C0 Hybrid cells | Actual batch1/1/1 | n3 per arm')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,'hybrid_iso8_confirmed_'+stem,
            'MEASURED ONLINE C0 | N16/start | Three paired independent starts per cell;pointwise95% empirical CI,27 resamples.\n'
            'Layer8,Token8,Hybrid4/suffix≈L/8 have exactly equal aggregate Transformer rows. Attention,payload and native extra query differ.\n'
            'I3/O64 is an iso-work TTFT candidate,not a best-tuned C3 or dynamic win. All other cells remain. Throughput is finite-cohort.',
            dict(evidence='MEASURED_ONLINE',protocol='C0 actual1/1/1',metric=field,cells=cells))
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,field,title in zip(axes,('ITL_p99_ms','TPOT_p95_ms'),('Hybrid / Native ITL p99','Hybrid / Native TPOT p95')):
        for i,cell in enumerate(cells):
            row=cell['comparisons']['Hybrid_over_Native'][field]
            v=row['estimate'];lo,hi=row['CI95']
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,
                color='#009e73' if hi<=1.05 else '#d55e00')
        ax.axvline(1,color='gray');ax.axvline(1.05,color='red',ls='--')
        ax.set(xlabel=title);ax.grid(alpha=.2)
    axes[0].set_yticks(x,labels);axes[0].invert_yaxis()
    fig.suptitle('Exact-row C0 Hybrid decode protection | All eight cells')
    save(fig,'hybrid_iso8_confirmed_decode_gate',
        'MEASURED ONLINE C0 | Three paired independent starts/cell | Green:metric CI upper≤1.05. Orange:metric fails.\n'
        'Pointwise95% empirical CI,27 resamples. Absolute every-ITL34ms and request-TPOT33ms remain separate.\n'
        'Passing this control does not qualify C3 continuous batching or a future decode-safe insertion window.',
        dict(evidence='MEASURED_ONLINE',protocol='C0 actual1/1/1',cells=cells))
    print('EXACT_ISO8_FIGURES',3,'PNG/PDF pairs',flush=True)


if __name__=='__main__':
    main()
