"""Draw operation enclosures and separately labeled actual kernel intervals."""
import csv,json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def main():
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    data=json.loads((ROOT/'analysis/new_diagnostics.json').read_text());by={x['name']:x for x in data}
    replacements=json.loads((ROOT/'state/trace_versions.json').read_text())['replacements']
    arms=[('Zero','tail_zero_clean_n64'),('Loop28 full','tail_full_clean_n64'),
        ('Loop28 tile256','tail_tiled_clean_n64'),('Loop31 tile512','tail_loop31_clean_n64')]
    manifest=[]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,stat in zip(axes,('TTFT_ms','ITL_ms','TPOT_ms'),('p95','p99','p95')):
        ax.bar([a for a,n in arms],[by[n]['metrics'][key][stat] for a,n in arms],color=['#447e9f','#c65c52','#d1a34c','#148979'])
        ax.set_title(key+' '+stat);ax.tick_params(axis='x',rotation=25);ax.set_ylabel('ms')
    fig.suptitle('New matched N64 clean diagnostic. Not the frozen N128 results.')
    fig.tight_layout();fig.savefig(folder/'clean_diagnostic.png',dpi=150);fig.savefig(folder/'clean_diagnostic.svg');plt.close(fig)
    manifest.append(dict(name='clean_diagnostic',scope='Measured marker-free short controls. One Arrival trace, not independent-start confidence.'))
    full=replacements['tail_full_events_n64']
    with (ROOT/'analysis'/(full+'_cycles.csv')).open() as f:cycles=list(csv.DictReader(f))
    chosen=sorted(cycles,key=lambda x:float(x['ITL_ms']))[int(.99*(len(cycles)-1))]
    seed,g=int(chosen['seed']),int(chosen['generated'])
    panels=[('Zero',replacements['tail_zero_events_n64']),('Loop28 full',full),
        ('Loop28 tile256','tail_tiled_events_n64'),('Loop31 tile512','tail_loop31_events_n64'),
        ('Loop28 serialized counterfactual','tail_full_serial_events_n64')]
    fig,axes=plt.subplots(len(panels),1,figsize=(13,11),sharex=True)
    for ax,(label,name) in zip(axes,panels):
        trace=json.loads((ROOT/'analysis'/(name+'_event_timeline.json')).read_text())
        with (ROOT/'analysis'/(name+'_cycles.csv')).open() as f:
            interval=next(x for x in csv.DictReader(f) if int(x['seed'])==seed and int(x['generated'])==g)
        origin=float(interval['previous_output']);end=float(interval['next_output']);left=origin-.01;right=origin+.23
        for row in trace['rows']:
            a,b=row['begin']['time'],row['end']['time']
            if b<=left or a>=right:continue
            prompt=row['prompt'];y=1 if prompt else 0
            selected=any(x['seed']==seed for x in row['requests'])
            ax.broken_barh([((a-origin)*1000,1000*(b-a))],(y-.22,.44),
                facecolors='#148979' if prompt else '#447e9f',alpha=1 if selected or prompt else .2)
            if selected and not prompt and row['operation']=='recurrent' and a>=left:
                ax.text(max(1,(a-origin)*1000)+2,y+.25,f'R32 Decode, B{len(row["requests"])}',fontsize=8,clip_on=True)
        ax.axvline(0,color='black',lw=.7);ax.axvline(1000*(end-origin),color='#c65c52',label='Output delivery')
        ax.axvline(192.8212538,color='#c65c52',linestyle=':',lw=.8)
        ax.set_yticks([0,1],['Decode operations','Prefill quantum']);ax.set_title(label,loc='left',fontsize=10)
        ax.set_xlim(-10,230)
    axes[-1].set_xlabel('ms after this request\'s previous output. Independent runs aligned by that output.')
    fig.suptitle(f'CUDA Event operation enclosures. Seed {seed}, next token {g+1}.\nEnclosures are not exclusive kernels. Blank regions are not measured idle. Dotted line: frozen output threshold.')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(folder/'operation_timeline.png',dpi=150);fig.savefig(folder/'operation_timeline.svg');plt.close(fig)
    manifest.append(dict(name='operation_timeline',request_seed=seed,generated=g,
        selection='The interval nearest the observed Loop28 diagnostic ITL p99.',
        scope='CUDA Event enclosures aligned across independent runs. Same request and token index. Not a physical kernel timeline.'))
    attribution=json.loads((ROOT/'analysis/attribution.json').read_text())
    for result in attribution['kernel_tails']:
        if not result or result['status']!='OBSERVED_INSTRUMENTED_KERNEL_TAIL':continue
        name=result['name'];kernels=json.loads((ROOT/'analysis'/(name+'_kernel_intervals.json')).read_text())
        start=result['capture']['start']['host_midpoint'];limit=min(.3,result['window_s'])
        groups=defaultdict(list)
        for row in kernels:
            a,b=max(row['start'],start),min(row['end'],start+limit)
            if b>a:groups[row['stream']].append((1000*(a-start),1000*(b-a)))
        fig,ax=plt.subplots(figsize=(13,max(3,len(groups)*.6)))
        for y,(stream,spans) in enumerate(sorted(groups.items())):
            ax.broken_barh(spans,(y-.3,.6),facecolors='#447e9f')
        ax.set_yticks(range(len(groups)),[f'CUPTI stream {s}' for s in sorted(groups)])
        ax.set_xlabel('ms after capture anchor');ax.set_title(name+' — actual captured kernels, profiler sensitivity applies')
        fig.tight_layout();fig.savefig(folder/(name+'_kernels.png'),dpi=150);fig.savefig(folder/(name+'_kernels.svg'));plt.close(fig)
        manifest.append(dict(name=name+'_kernels',scope='Actual CUPTI kernel intervals in the first captured 300 ms. Not a clean full-cohort timeline. Instrumentation sensitivity is a separate gate.'))
    write(ROOT/'figure_manifest.json',manifest)
    print('FIGURES',len(manifest))


if __name__=='__main__':main()
