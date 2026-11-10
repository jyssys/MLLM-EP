"""Render an AS-IS-style comparison from host records, with explicit limits."""
from matplotlib.patches import Patch
import json
from .common import ROOT,write
from .figures import plt,save


def main():
    data=json.loads((ROOT/'figures/recovery_observed_timeline_data.json').read_text())
    fig,axes=plt.subplots(2,4,figsize=(20,7))
    chosen=[]
    for row,model in enumerate(('ouro','huginn')):
        arms=data[model]
        loop=next(a for a in arms if a['label']=='Recovery Loop v2')
        candidate=next((r for r in loop['requests'] if r['offloaded_loops']),None)
        index=candidate['index'] if candidate else 0
        selected=[next(r for r in a['requests'] if r['index']==index) for a in arms]
        extent=max(r['first']-r['arrival'] for r in selected)*1.15
        extent=max(.5,extent)
        chosen.append(dict(model=model,request_index=index,rule='First actual Loop offload in recovery seed 2026110931. Not selected for speedup.'))
        for col,(ax,arm,request) in enumerate(zip(axes[row],arms,selected)):
            origin=request['arrival']
            def bar(start,end,y,color,alpha=1.,edge='none'):
                start=max(0,start-origin);end=min(extent,end-origin)
                if end<start or start>extent:return
                ax.broken_barh([(start,max(0,end-start))],(y-.23,.46),facecolors=color,alpha=alpha,
                    edgecolors=edge,linewidth=1.2)
            for other in arm['requests']:
                for segment in other['p']:
                    bar(segment['start'],segment['end'],1,'#df9a3f',.28)
            for segment in request['p']:bar(segment['start'],segment['end'],1,'#df9a3f',.95,'#a5671e')
            for segment in request['d']:bar(segment['start'],segment['end'],0,'#148979',.95)
            if request['handoff_begin'] is not None and request['received'] is not None:
                bar(request['handoff_begin'],request['received'],.48,'#8b6eac',.8)
            for timestamp in arm['decode_outputs']:
                t=timestamp-origin
                if 0<=t<=extent:ax.vlines(t,-.42,-.31,color='#4386bd',alpha=.35,lw=.5)
            first=request['first']-origin
            ax.axvline(first,color='#bd4051',ls='--',lw=1.5)
            ax.text(.02,.97,f"TTFT {request['ttft_ms']:.1f} ms\nD: {request['offloaded_loops']} full-prompt Loops\nSuffix: {request['offloaded_suffix_tokens']} tokens",va='top',
                transform=ax.transAxes,fontsize=9)
            ax.set_xlim(0,extent);ax.set_ylim(-.55,1.8);ax.set_yticks([0,1],['D observed work','P host execution'])
            ax.set_title(f'{model.capitalize()} · {arm["label"]} · request {index}')
            ax.set_xlabel('Time from request arrival (s)');ax.grid(axis='x',alpha=.15)
    fig.suptitle('ACTUAL HOST OBSERVATIONS · original Native / zero-offload / adaptive Loop / Token',fontsize=15)
    fig.legend(handles=[Patch(color='#df9a3f',label='P host enclosure'),Patch(color='#148979',label='Selected D continuation enclosure'),
        Patch(color='#8b6eac',label='Observed handoff interval'),Patch(color='#4386bd',label='Output delivery marks')],
        loc='lower center',ncol=4,bbox_to_anchor=(.5,.015),fontsize=9)
    fig.text(.5,-.015,'These are not CUDA kernel bars. Blank space is not GPU idle. Independent runs share the same requests and arrival seed.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.95))
    item=save(fig,'recovery_actual_host_comparison',
        'Same selected request across matched arrival runs. Host records only. Other-source activity is pale orange. No invented physical idle.',
        ['figures/recovery_observed_timeline_data.json'])
    write(ROOT/'figures/host_comparison_representatives.json',chosen)
    manifest=json.loads((ROOT/'figure_manifest.json').read_text())
    write(ROOT/'figure_manifest.json',[r for r in manifest if r['name']!=item['name']]+[item])
    print(json.dumps(chosen))


if __name__=='__main__':
    main()
