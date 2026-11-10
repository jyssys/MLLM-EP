"""Draw recovery measurements and actual boundary decisions with scope labels."""
from collections import Counter
import json
import numpy as np
from .figures import plt,save
from .common import ROOT,write
from .audit_old import events
from .timeline import HTML,run_data


def serving(rows):
    keys=[('TTFT_p95_ms','TTFT p95 (s)',.001),('ITL_p99_ms','ITL p99 (ms)',1),
        ('E2E_p95_ms','E2E p95 (s)',.001),('SLO_goodput','SLO-goodput (req/s)',1)]
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for i,model in enumerate(('ouro','huginn')):
        for j,(key,title,scale) in enumerate(keys):
            ax=axes[i,j]
            for x,arm in enumerate(('native','zero','loop','token')):
                values=[r[key]*scale for r in rows if r.get('complete') and r['model']==model
                    and r['secondary_arm']==arm and r['cohort']=='N128_CONFIRMATION']
                if not values:ax.text(x,.02,'NOT RUN',rotation=90,transform=ax.get_xaxis_transform());continue
                ax.bar(x,np.mean(values),color=('#3c4654','#6d95bb','#148979','#8266b3')[x],alpha=.8)
                ax.scatter(np.linspace(x-.12,x+.12,len(values)),values,c='#152537',s=18,zorder=3)
                ax.text(x,.98,f'n={len(values)}',ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=8)
            ax.set_xticks(range(4),['Native original','Native-path zero','Loop v2','Token v2'],rotation=25,fontsize=8)
            ax.set_title(f'{model.capitalize()} · {title}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if key=='ITL_p99_ms':ax.axhline(65.08287642151119 if model=='ouro' else 192.8212538175284,c='#bd4051',ls='--')
    fig.suptitle('Recovery v2 · marker-free N128 · held-out paired arrival seeds',fontsize=14)
    fig.text(.5,-.02,'Common KV pool and graph budget. Bars: start means. Points: independent starts. Zero-offload controls ran later. Finite-cohort rates only.',ha='center',fontsize=9)
    fig.tight_layout()
    return save(fig,'recovery_serving_comparison','Measured marker-free N128 recovery starts. Absolute SLO and Native-relative tail comparisons are distinct.',['recovery_serving_results.json'])


def decisions(rows):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    sources=[]
    for ax,model in zip(axes,('ouro','huginn')):
        total=0;committed=0;counts=Counter()
        for item in rows:
            if not item.get('complete') or item['model']!=model or item['secondary_arm']!='loop' or item['cohort']!='N128_CONFIRMATION':continue
            folder=ROOT/'raw'/item['name'];begin=json.loads((folder/'measurement_begin.json').read_text())['time'];ev=events(folder,begin)
            commits={(r['tid'],r['epoch']) for r in ev if r['event']=='SOURCE_COMMIT'}
            total+=item['count'];committed+=len(commits);sources.append(item['name'])
            for r in ev:
                if r['event']!='ADMISSION':continue
                real=(r['tid'],r['epoch']) in commits
                ax.scatter(r['budget_ms'],r['remaining'],marker='o' if real else 'x',
                    color='#148979' if real else '#a5aab2',alpha=.85 if real else .35,s=35)
                if real:counts[r['remaining']]+=1
        ax.set_title(f'{model.capitalize()} · actual commits {committed}/{total}')
        ax.set_xlabel('Predicted execution budget at admission (ms)');ax.set_ylabel('Remaining full-prompt Loops')
        ax.grid(alpha=.2);ax.text(.02,.98,'Green: committed\nGray: rejected or uncommitted\n'+str(dict(sorted(counts.items()))),
            transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle('Actual online boundary decisions · not forced cuts or an Oracle')
    fig.tight_layout()
    return save(fig,'recovery_budget_vs_actual_loop_count','Host-observed online budget and exact committed suffix depth. Not measured physical idle.',['recovery_serving_results.json',*sources])


def completion():
    rows=json.loads((ROOT/'recovery_completion_audit.json').read_text())
    confirmation_names=set(json.loads((ROOT/'recovery_plan.json').read_text())['jobs'])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,model in zip(axes,('ouro','huginn')):
        selected=[r for r in rows if r['model']==model and not r['token_split'] and r['name'] in confirmation_names]
        for r in selected:
            ax.scatter(r['predicted_completion_ms'],r['committed_to_first_output_host_ms'],
                c='#bd4051' if r['observed_completion_reservation_missed'] else '#148979',s=35)
        maximum=max([r['committed_to_first_output_host_ms'] for r in selected]+[100])
        ax.plot([0,maximum],[0,maximum],ls='--',c='gray',lw=1)
        ax.set_title(f'{model.capitalize()} · {len(selected)} committed requests')
        ax.set_xlabel('Predicted completion (ms)');ax.set_ylabel('Commit → first output (host ms)');ax.grid(alpha=.2)
    fig.suptitle('Completion predictions versus observations · red: reservation miss')
    fig.text(.5,-.02,'Observed intervals include transfer, import, scheduling, computation, and first output. They are not GPU kernel durations.',ha='center',fontsize=9)
    fig.tight_layout()
    return save(fig,'recovery_completion_prediction_error','Measured host completion and modeled prediction. No guaranteed completion or GPU occupancy claim.',['recovery_completion_audit.json'])


def timeline():
    data={}
    names=json.loads((ROOT/'recovery_plan.json').read_text())['jobs']
    for model in ('ouro','huginn'):
        data[model]=[]
        for arm,label in (('native','Native original'),('zero','Native-path zero'),('loop','Recovery Loop v2'),('token','Recovery Token v2')):
            name=(f'control_{model}_zero_seed2026110931' if arm=='zero' else
                next(n for n in names if f'_{model}_{arm}_seed2026110931' in n))
            data[model].append(run_data(ROOT,name,label))
    folder=ROOT/'figures';write(folder/'recovery_observed_timeline_data.json',data)
    text=HTML.replace('Native / old / new','Native / zero-offload / recovery Loop / recovery Token')
    (folder/'recovery_native_zero_loop_token_timeline.html').write_text(text.replace('__DATA__',json.dumps(data)))
    return dict(name='recovery_native_zero_loop_token_timeline',html='figures/recovery_native_zero_loop_token_timeline.html',
        scope='Independent same-arrival runs. Host observations only. No invented GPU kernel or idle bars.',
        sources=['figures/recovery_observed_timeline_data.json'])


def main():
    rows=json.loads((ROOT/'recovery_serving_results.json').read_text())
    rows += [dict(r,cohort='N128_CONFIRMATION',secondary_arm='zero')
        for r in json.loads((ROOT/'zero_control_results.json').read_text())]
    manifest=json.loads((ROOT/'figure_manifest.json').read_text())
    new=[serving(rows),decisions(rows),completion(),timeline()]
    keys={r['name'] for r in new}
    write(ROOT/'figure_manifest.json',[r for r in manifest if r['name'] not in keys]+new)
    print(json.dumps(new,indent=2))


if __name__=='__main__':
    main()
