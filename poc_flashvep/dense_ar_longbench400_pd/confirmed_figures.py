"""Plot measured start-level evidence without promoting C0 to continuous serving."""
import collections
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write,digest
from .figures import save


def mean_ci(values):
    a=np.asarray(values,float);n=len(a)
    if n<3:return float(a.mean()),None
    index=np.asarray(list(itertools.product(range(n),repeat=n)))
    return float(a.mean()),np.percentile(a[index].mean(axis=1),[2.5,97.5]).tolist()


def motivation():
    jobs=json.loads((ROOT/'jobs/motivation400_discovery_v1.json').read_text())['jobs']
    jobs += [r['name'] for r in json.loads((ROOT/'configs/motivation_independent_confirmation_v1.json').read_text())['rows']]
    groups=collections.defaultdict(list);provenance=[]
    for name in jobs:
        p=ROOT/'analysis'/(name+'_metrics.json');assert p.exists(),name
        d=json.loads(p.read_text());cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
        assert d['n']==400 and not cfg.get('axis_kind') and not cfg['batch_invariant']
        quality=json.loads((ROOT/'analysis'/(name+'_quality_gate.json')).read_text())
        assert quality['passed'],('Quality failed',name)
        groups[cfg['method'],cfg['qps']].append(d)
        provenance.append(dict(name=name,method=cfg['method'],QPS=cfg['qps'],seed=cfg['seed'],
            output_tokens=d['output_tokens'],dataset_SHA256=cfg['dataset_SHA256'],
            source_SHA256=d['runtime_source_archive_SHA256'],GPU_pair=cfg['gpus'],quality_passed=True))
    fields=[('TTFT_p95_ms','TTFT p95 (ms)','ttft_p95'),
        ('ITL_p99_ms','Client-visible ITL p99 (ms)','itl_p99'),
        ('TPOT_p95_ms','Request TPOT p95 (ms/token)','tpot_p95'),
        ('generated_output_tokens_per_second','Generated output tokens/s','output_tokens'),
        ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','slo_goodput')]
    manifest=[]
    for field,ylabel,stem in fields:
        fig,ax=plt.subplots(figsize=(8,5));points=[]
        for method,color,label in (('Agg','#d95f02','Agg DP2'),('Native','#009e73','Native PD 1P1D')):
            measured=sorted((q,rows) for (m,q),rows in groups.items() if m==method)
            xs=[];ys=[]
            for q,rows in measured:
                value,ci=mean_ci([r[field] for r in rows]);xs.append(q);ys.append(value)
                points.append(dict(method=method,QPS=q,mean=value,CI95=ci,independent_starts=len(rows)))
                if ci is not None:ax.errorbar(q,value,yerr=[[value-ci[0]],[ci[1]-value]],fmt='none',color=color,capsize=4)
                ax.annotate('n'+str(len(rows)),(q,value),xytext=(3,5),textcoords='offset points',fontsize=7,color=color)
            ax.plot(xs,ys,'o-',color=color,label=label)
        ax.set(xlabel='Offered QPS (requests/s)',ylabel=ylabel,
            title='LongBench-Balanced400-v1 | BI0 | 2 × H100 | B8192/S1024')
        if field=='TTFT_p95_ms':ax.axhline(5000,color='gray',ls='--',lw=1,label='TTFT SLO 5000ms')
        if field=='ITL_p99_ms':ax.axhline(34,color='gray',ls='--',lw=1,label='Every-ITL SLO 34ms (not a p99 SLO)')
        if field=='TPOT_p95_ms':ax.axhline(33,color='gray',ls='--',lw=1,label='Request TPOT SLO 33ms')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        caption=('MEASURED ONLINE | N400/start | Natural EOS cap2048 | n3 at QPS1.8/2.8/4.5; n1 elsewhere.\n'
            'Error bars: pointwise start-mean empirical bootstrap95% CI (27 resamples). No CI at n1.\n'
            'Finite-cohort drain throughput, not sustainable capacity. Lines connect measured points. BI0 actual output counts are recorded.')
        save(fig,'motivation400_confirmed_'+stem,caption,dict(evidence='MEASURED_ONLINE',metric=field,points=points,
            provenance=provenance,dataset_SHA256=digest(ROOT/'data/longbench_balanced400_v1.json')))
        manifest.append(dict(metric=field,stem='motivation400_confirmed_'+stem,points=points))
    write(ROOT/'analysis/motivation400_confirmed_metric_table_v1.json',dict(figures=manifest,provenance=provenance))
    print('Created five confirmed motivation figures.',flush=True)


def axis_matrix():
    result=json.loads((ROOT/'analysis/axis_matrix_heldout_confirmation_r2.json').read_text())
    fields=[('TTFT_p95_Layer_over_Token','Layer / Token TTFT p95','ttft_ratio'),
        ('goodput_Layer_over_Token','Layer / Token SLO-goodput','goodput_ratio')]
    for field,title,stem in fields:
        fig,axes=plt.subplots(1,4,figsize=(17,5));images=[]
        for ax,output in zip(axes,(8,16,64,128)):
            cells=[c for c in result['cells'] if c['output']==output]
            values=np.full((4,4),np.nan)
            for c in cells:
                if c.get(field):values[c['input_bucket'],(2,4,8,12).index(c['cut'])]=c[field]['estimate']
            image=ax.imshow(np.log2(values),cmap='coolwarm',vmin=-.35,vmax=.35);images.append(image)
            for c in cells:
                i,j=c['input_bucket'],(2,4,8,12).index(c['cut']);test=c.get(field)
                text='NOT_RUN' if not test else f"{test['estimate']:.3f}\nn{test['starts']}"
                if c['status']=='INVALID':text+='\nINVALID'
                elif c['status'] in ('LAYER_CONFIRMED','TOKEN_CONFIRMED'):text+='*'
                ax.text(j,i,text,ha='center',va='center',fontsize=8)
            ax.set_xticks(range(4),['Last2','Last4','Last8','Last12'])
            ax.set_yticks(range(4),['3K–6K','6K–10K','10K–20K','20K–36K'])
            ax.set(xlabel='Matched-work cut',title='Controlled output'+str(output))
        axes[0].set_ylabel('Original-prompt input bucket')
        fig.suptitle(title+' | C0 actual batch1/1/1 | Held-out prompts')
        # Keep a separate color legend because the ratio is transformed for symmetric coloring.
        fig.subplots_adjust(left=.08,right=.91,bottom=.24,top=.84,wspace=.42)
        color_axis=fig.add_axes([.935,.27,.012,.53])
        fig.colorbar(images[0],cax=color_axis,label='log2(Layer / Token)')
        caption=('MEASURED ONLINE, C0 mechanism control | N16/start | n3 paired independent starts; three predeclared core cells n5.\n'
            '*: pointwise TTFT direction plus winner decode gates. INVALID: latency direction fails decode protection.\n'
            '64 unadjusted CIs do not control familywise error. Not C3 continuous serving. Common QPS per input:0.910/0.871/0.822/0.763.')
        save(fig,'axis400_heldout_'+stem,caption,dict(evidence='MEASURED_ONLINE',protocol='C0 actual1/1/1',
            counts=result['counts'],metric=field,cells=result['cells']),manual_layout=True)
    print('Created two held-out axis ratio matrices.',flush=True)


def axis_decode_gates():
    result=json.loads((ROOT/'analysis/axis_matrix_heldout_confirmation_r2.json').read_text())
    from matplotlib.colors import TwoSlopeNorm
    for field,stem,title in (('ITL_p99_ms','itl_upper','ITL p99 CI upper / Native'),
            ('TPOT_p95_ms','tpot_upper','TPOT p95 CI upper / Native')):
        fig,axes=plt.subplots(2,4,figsize=(17,8));images=[]
        for row,axis in enumerate(('Layer','Token')):
            for ax,output in zip(axes[row],(8,16,64,128)):
                cells=[c for c in result['cells'] if c['output']==output]
                values=np.full((4,4),np.nan)
                for c in cells:
                    test=c.get('native_relative',{}).get(axis,{}).get(field)
                    if test:values[c['input_bucket'],(2,4,8,12).index(c['cut'])]=test['CI95'][1]
                image=ax.imshow(values,cmap='RdYlGn_r',norm=TwoSlopeNorm(vmin=.9,vcenter=1.05,vmax=1.2))
                images.append(image)
                for c in cells:
                    i,j=c['input_bucket'],(2,4,8,12).index(c['cut'])
                    test=c.get('native_relative',{}).get(axis,{})
                    value=test.get(field)
                    text='NOT_RUN' if value is None else f"{value['CI95'][1]:.3f}"
                    if value is not None and value['CI95'][1]>1.05:text+='\nNI FAIL'
                    if test and not test['absolute_decode_pass']:text+='\nABS FAIL'
                    ax.text(j,i,text,ha='center',va='center',fontsize=8)
                ax.set(xticks=range(4),xticklabels=['Last2','Last4','Last8','Last12'],
                    yticks=range(4),yticklabels=['3K–6K','6K–10K','10K–20K','20K–36K'],
                    title=axis+' / Output'+str(output),xlabel='Matched-work cut')
            axes[row,0].set_ylabel('Original-prompt input bucket')
        fig.suptitle(title+' | C0 actual batch1/1/1 | Threshold1.05')
        fig.subplots_adjust(left=.08,right=.91,bottom=.17,top=.9,wspace=.42,hspace=.48)
        color_axis=fig.add_axes([.935,.23,.012,.57]);fig.colorbar(images[0],cax=color_axis,label='Paired CI upper ratio')
        save(fig,'axis400_heldout_decode_'+stem,
            'MEASURED ONLINE C0 | N16/start | n3 paired independent starts;three core cells n5. Values are upper95% CI bounds,not mean ratios.\n'
            'NI FAIL:that metric upper bound>1.05. ABS FAIL:request decode-SLO attainment<95% in at least one start.\n'
            'Both metrics and the absolute gate are required. Pointwise empirical CIs have no familywise guarantee. Not C3 continuous safety.',
            dict(evidence='MEASURED_ONLINE',protocol='C0 actual1/1/1',metric=field,cells=result['cells']),manual_layout=True)
    print('Created two all-cell decode-gate matrices.',flush=True)


def main():
    motivation();axis_matrix();axis_decode_gates()


if __name__=='__main__':main()
