"""Editable figures from measured mechanism data. No predicted serving results."""
import csv
import json
import statistics
from .common import ROOT


def main():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader((ROOT/'full_results.csv').open()) if r['name'].startswith('quiet')]
    arms = ['PD_G1','A_sep','A_native_mixed']
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, model in zip(axes, ('ouro','huginn')):
        for index, arm in enumerate(arms):
            values = [float(r['makespan_ms'])/1000 for r in rows if r['model']==model and r['arm']==arm]
            if values:
                ax.bar(index, statistics.mean(values), color=['#76899b','#ce9773','#528f83'][index])
                ax.scatter([index]*len(values),values,color='black',s=18)
        ax.set_xticks(range(3),arms,rotation=15)
        ax.set_title(model.capitalize()+' | eight requests per start')
        ax.set_ylabel('Measured cohort makespan (s)')
    fig.suptitle('128-token prefixes, cap 32, independent quiet starts')
    fig.tight_layout()
    fig.savefig(ROOT/'figures/quiet_makespan.svg')
    plt.close(fig)

    share_rows = list(csv.DictReader((ROOT/'e2e_component_shares.csv').open()))
    phases = list(dict.fromkeys(r['phase'] for r in share_rows))
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax, model in zip(axes, ('ouro','huginn')):
        left = [0.]*3
        for phase in phases:
            values = []
            for arm in arms:
                group = [float(r['share_pct']) for r in share_rows
                         if f'_{model}_{arm}_' in r['run'] and r['phase']==phase]
                values.append(statistics.mean(group) if group else 0.)
            ax.barh(arms,values,left=left,label=phase.replace('_',' '))
            left = [x+y for x,y in zip(left,values)]
        ax.set_xlim(0,100)
        ax.set_xlabel('Share of summed request E2E (%)')
        ax.set_title(model.capitalize()+' | serial host observations')
    axes[1].legend(loc='upper center',bbox_to_anchor=(-.1,-.18),ncol=2,fontsize=8)
    fig.suptitle('Not a GPU-kernel or cohort critical-path partition')
    fig.tight_layout(rect=(0,.16,1,1))
    fig.savefig(ROOT/'figures/host_e2e_shares.svg')
    plt.close(fig)

    evidence=json.loads((ROOT/'operator_fusion_evidence.json').read_text())
    fig, axes=plt.subplots(1,2,figsize=(12,4))
    for ax, record in zip(axes,evidence['selected_kernel_traces']):
        path=ROOT/'raw'/record['run']/record['trace']
        events=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel']
        origin=min(e['ts'] for e in events)
        streams=sorted({e['args']['stream'] for e in events})
        for index,stream in enumerate(streams):
            intervals=[((e['ts']-origin)/1000,e['dur']/1000) for e in events if e['args']['stream']==stream]
            ax.broken_barh(intervals,(index-.3,.6),facecolors='#407988')
        ax.set_yticks(range(len(streams)),[f'Stream {s}' for s in streams])
        ax.set_xlabel('Time from first kernel (ms)')
        ax.set_title(record['run'].split('_')[2].capitalize()+f" | {len(events)} GPU kernels")
    fig.suptitle('One native mixed core per model | heavy diagnostic trace, not serving latency')
    fig.tight_layout()
    fig.savefig(ROOT/'figures/selected_gpu_kernel_timeline.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
