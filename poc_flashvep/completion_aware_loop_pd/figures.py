"""Draw measured serving results and explicitly labeled diagnostic bounds."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write,digest

OUT=ROOT/'figures'
COLORS=['#3c4654','#df9a3f','#148979','#8266b3']


def save(fig,name,scope,sources):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    return dict(name=name,png=str((OUT/(name+'.png')).relative_to(ROOT)),
        svg=str((OUT/(name+'.svg')).relative_to(ROOT)),scope=scope,sources=sources)


def serving():
    rows=json.loads((ROOT/'serving_results.json').read_text());fig,axes=plt.subplots(2,4,figsize=(15,7))
    keys=[('TTFT_p95_ms','TTFT p95 (s)',.001),('ITL_p99_ms','ITL p99 (ms)',1),
        ('completed_QPS','Completed requests/s',1),('SLO_goodput','Joint SLO-goodput (req/s)',1)]
    for i,model in enumerate(('ouro','huginn')):
        for j,(key,title,scale) in enumerate(keys):
            ax=axes[i,j]
            for x,arm in enumerate(('native','old','loop','token')):
                values=[r[key]*scale for r in rows if r.get('complete') and r['model']==model and f'_{arm}_seed' in r['name']]
                if not values:continue
                ax.bar(x,np.mean(values),color=COLORS[x],alpha=.78,width=.65)
                ax.scatter(np.linspace(x-.14,x+.14,len(values)),values,color='#1b2230',s=17,zorder=3)
                if len(values)<3:ax.text(x,np.mean(values),f'  n={len(values)}',fontsize=8,rotation=90,va='bottom')
            ax.set_xticks(range(4),['Native','Old Loop','New Loop','Token'],rotation=20)
            ax.set_title(f'{model.capitalize()} · {title}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if key=='ITL_p99_ms':
                limit=65.08287642151119 if model=='ouro' else 192.8212538175284
                ax.axhline(limit,color='#be3d4c',ls='--',lw=1,label='Frozen p99 SLO');ax.legend(fontsize=7)
    fig.suptitle('Primary v1, before recovery · marker-free N128 · finite-cohort rates',fontsize=14)
    fig.text(.5,-.02,'Bars: arithmetic means of start-level metrics. Points: independent starts. Ouro Token start 3 did not run. No Loop superiority claim.',ha='center',fontsize=10)
    fig.tight_layout()
    return save(fig,'primary_serving_comparison','Measured marker-free output-delivery metrics. Not sustainable capacity. Token cold graph captures confound the mechanism comparison.',['serving_results.json'])


def ratios():
    rows=json.loads((ROOT/'paired_start_ratios.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
    keys=['TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','SLO_goodput']
    for ax,model in zip(axes,('ouro','huginn')):
        for i,key in enumerate(keys):
            r=next(r for r in rows if r['model']==model and r['arm']=='loop' and r['reference']=='native' and r['metric']==key)
            point=r['geometric_ratio'];lo,hi=r['bootstrap_95_CI']
            ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',color='#148979',capsize=4)
        ax.axvline(1,color='gray',ls='--');ax.set_yticks(range(4),['TTFT p95','E2E p95','ITL p99','SLO-goodput'])
        ax.set_xlabel('New Loop / Native ratio');ax.set_title(model.capitalize());ax.grid(axis='x',alpha=.2)
    fig.suptitle('Three paired starts · geometric ratios and start-bootstrap 95% intervals')
    fig.text(.5,-.03,'Three starts provide limited uncertainty coverage. Lower latency is better. Higher goodput is better.',ha='center',fontsize=10)
    fig.tight_layout()
    return save(fig,'primary_paired_ratios','Measured paired start-level ratios. Confidence intervals do not treat tokens as independent samples.',['paired_start_ratios.json'])


def idle_diagnostics():
    fig,axes=plt.subplots(2,2,figsize=(12,7));bins=np.logspace(-5,4,55)
    for col,model in enumerate(('ouro','huginn')):
        name=f'oracle_decode_only_{model}_full128_calibration'
        data=np.load(ROOT/'idle_slack_oracle'/f'{name}_decode_intervals.npz');gaps=data['idle'];ms=(gaps[:,1]-gaps[:,0])/1e6
        axes[0,col].hist(ms,bins=bins,color='#467bac');axes[0,col].set_xscale('log');axes[0,col].set_yscale('log')
        axes[0,col].set_title(f'{model.capitalize()} · {len(ms):,} observed gaps');axes[0,col].set_ylabel('Gap count')
        axes[1,col].hist(ms,bins=bins,weights=ms/1000,color='#9d6b9e');axes[1,col].set_xscale('log')
        axes[1,col].set_ylabel('Idle time in bin (s)');axes[1,col].set_xlabel('Contiguous kernel gap (ms)')
        for ax in axes[:,col]:ax.grid(alpha=.2)
    fig.suptitle('DIAGNOSTIC ONLY · full D-stage kernel traces FAIL the 5% interference gate',color='#b63747',fontsize=14)
    fig.text(.5,-.02,'These are instrumented executions, not clean Native capacity. A kernel gap is not a verified Decode-safe opportunity.',ha='center',fontsize=10)
    fig.tight_layout()
    return save(fig,'instrumented_idle_gap_histogram','Measured D-stage kernel gaps in perturbed runs. Not qualified clean capacity.',['idle_slack_oracle/oracle_decode_only_ouro_full128_calibration.json','idle_slack_oracle/oracle_decode_only_huginn_full128_calibration.json'])


def costs():
    rows=json.loads((ROOT/'refreshed_continuation_enclosures.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,model in zip(axes,('ouro','huginn')):
        width=4096 if model=='ouro' else 2048;counts=(1,2,3) if model=='ouro' else (1,2,4,8)
        for field,label,color in (('operation_enclosure_sum_ms','Sum of operation enclosures','#9dbdb7'),
                ('first_to_last_prompt_GPU_clock_enclosure_ms','Complete recurrent-suffix enclosure','#148979')):
            med=[];lo=[];hi=[]
            for count in counts:
                values=[r[field] for r in rows if r['model']==model and r['role']=='decode' and r['token_tile']==width and r['offloaded_loops']==count and r[field] is not None]
                med.append(np.median(values));lo.append(np.min(values));hi.append(np.max(values))
            ax.plot(counts,med,'o-',label=label,color=color);ax.fill_between(counts,lo,hi,color=color,alpha=.12)
        ax.set_title(model.capitalize());ax.set_xlabel('Remaining full-prompt loops');ax.set_ylabel('CUDA-clock enclosure (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fresh serial calibration · N8 per cell · shaded range: observed min–max')
    fig.text(.5,-.02,'Enclosures include waits and host gaps. They are not kernel unions, WCET guarantees, or online Decode-safe costs.',ha='center',fontsize=10)
    fig.tight_layout()
    return save(fig,'full_prompt_continuation_costs','Measured CUDA Event enclosures. No clean online speedup claim.',['refreshed_continuation_enclosures.json'])


def main():
    manifest=[serving(),ratios(),idle_diagnostics(),costs()]
    write(ROOT/'figure_manifest.json',manifest);print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
