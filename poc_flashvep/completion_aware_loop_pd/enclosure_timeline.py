"""Show GPU-clock stage enclosures without labeling them exclusive kernel bars."""
import json
from matplotlib.patches import Patch
from .common import ROOT,write
from .figures import plt,save


def main():
    values=json.loads((ROOT/'recovery_event_diagnostics.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(15,7))
    representatives=[]
    for ax,value in zip(axes,values):
        if value.get('status')=='NOT_COMPLETE':
            ax.text(.1,.5,'NOT_MEASURED',transform=ax.transAxes);continue
        traces=value['trace'];model=value['model']
        continuation=[r for r in traces['decode'] if r['prompt']]
        seed=next((x['seed'] for r in continuation for x in r['requests']),None)
        if seed is None:
            ax.text(.1,.5,'No actual continuation in this control',transform=ax.transAxes);continue
        selected=[r for r in traces['prefill'] if any(x['seed']==seed for x in r['requests'])]
        origin=min(r['begin']['time'] for r in selected)
        end=max(r['end']['time'] for r in continuation if any(x['seed']==seed for x in r['requests']))
        span=max(2.,end-origin+1.)
        source_requests=set()
        for role in ('prefill','decode'):
            for r in traces[role]:
                begin=r['begin']['time']-origin;stop=r['end']['time']-origin
                if stop<0 or begin>span:continue
                begin=max(0,begin);stop=min(span,stop)
                is_source=role=='prefill';is_continuation=role=='decode' and r['prompt']
                y=2 if is_source else 1 if is_continuation else 0
                color='#df9a3f' if is_source else '#148979' if is_continuation else '#4386bd'
                chosen=any(x['seed']==seed for x in r['requests'])
                ax.broken_barh([(begin,stop-begin)],(y-.28,.56),facecolors=color,
                    edgecolors='#bd4051' if chosen and (is_source or is_continuation) else 'none',
                    linewidth=1.2,alpha=.85)
                if is_source:source_requests.update(x['seed'] for x in r['requests'])
        ax.set_yticks([0,1,2],['D Decode enclosures','D continuation enclosures','P Prefill enclosures'])
        ax.set_xlim(0,span);ax.set_ylim(-.6,2.7);ax.set_xlabel('Time from selected request Prefill boundary (s)')
        ax.set_title(f'{model.capitalize()} · sampling seed {seed} · {len(source_requests)} source requests visible')
        ax.grid(axis='x',alpha=.2)
        representatives.append(dict(model=model,name=value['name'],seed=seed,origin=origin,end=end,
            event_interference_screen=value['five_percent_screen'],
            scope='Measured CUDA Event stage enclosures. A bar can include host gaps and waits. Blank intervals are not measured physical idle.'))
    fig.suptitle('DIAGNOSTIC GPU-CLOCK ENCLOSURES · not exclusive CUDA kernels',color='#bd4051',fontsize=14)
    fig.legend(handles=[Patch(color='#df9a3f',label='P Prefill'),Patch(color='#148979',label='D continued Prefill'),
        Patch(color='#4386bd',label='D Decode'),Patch(facecolor='white',edgecolor='#bd4051',label='Selected request')],
        ncol=4,loc='lower center',bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.04,1,.96))
    row=save(fig,'recovery_GPU_clock_enclosure_timeline',
        'Separate CUDA Event diagnostics. Enclosures can include idle time. This is not a physical kernel-idle trace.',
        ['recovery_event_diagnostics.json'])
    write(ROOT/'figures/recovery_GPU_clock_representatives.json',representatives)
    manifest=json.loads((ROOT/'figure_manifest.json').read_text())
    write(ROOT/'figure_manifest.json',[r for r in manifest if r['name']!=row['name']]+[row])
    print(json.dumps(representatives))


if __name__=='__main__':
    main()
