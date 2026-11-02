"""Show same-start kernel observations, host events, and live memory separately."""
import math
from .common import *
from .analyze import records,table
from .profile_analysis import union


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    manifest=json.loads((ROOT/'figure_manifest.json').read_text())
    extra=[]
    for model in ('ouro','huginn'):
        choices=[]
        paths=list((ROOT/'raw').glob(f'profile_warm_{model}_*/profile_analysis.json'))+list((ROOT/'raw').glob(f'long_profile_{model}_*/profile_analysis.json'))
        for path in paths:
            p=json.loads(path.read_text())
            if p.get('status')=='PARSED':choices.append((p.get('eligible_for_matched_clean',False),'n128' in path.parent.name,'both' in path.parent.name,path.parent))
        if not choices:continue
        folder=max(choices,key=lambda x:x[:3])[3];name=folder.name
        window=json.loads((folder/'window.json').read_text());begin=window['start'];end=window['end']
        from .queues import analyze
        q,series=analyze(name,window)
        fig,axes=plt.subplots(5,1,figsize=(12,10),sharex=True)
        sources=[folder/'window.json',folder/'queue_observations.jsonl',folder/'requests.jsonl']
        axes[0].step([r['time']-begin for r in series],[r['ready_backlog_lower_bound'] for r in series],where='post',label='D-clear P-ready lower bound')
        axes[0].step([r['time']-begin for r in series],[r['coordinator_D_credit_blocked'] for r in series],where='post',label='D-credit-blocked')
        axes[0].set_ylabel('Requests');axes[0].legend(loc='upper right',fontsize=8)
        for axis,role,color in ((axes[1],'prefill','C0'),(axes[2],'decode','C1')):
            trace=folder/f'aligned_kernels_{role}.json';proof=folder/f'profile_analysis_{role}.json'
            if not trace.exists():
                axis.text(.5,.5,f'{role.title()} kernel activity: NOT_MEASURED',transform=axis.transAxes,ha='center')
                axis.set_yticks([]);continue
            values=json.loads(trace.read_text());p=json.loads(proof.read_text());ks=values['kernels'];sources.extend((trace,proof))
            width=.1
            from .interval_bins import bins
            intervals=bins(ks,begin,end,width)
            edges=[a for a,b,v in intervals]
            occupancy=[v for a,b,v in intervals]
            # Record actual interval union. Missing collector output never
            # becomes zero or an invented idle interval.
            for at,stop,percentage in intervals:
                extra.append(dict(figure=model+'_same_start_kernel_timeline',name=name,role=role,
                    time_after_first_arrival_s=at-begin,bin_seconds=stop-at,kernel_union_percent=percentage,
                    eligible_profile=p['eligible_for_matched_clean'],evidence='MEASURED_DEVICE_KERNEL_INTERVAL_UNION'))
            axis.fill_between([v-begin for v in edges],occupancy,step='post',color=color,alpha=.7)
            axis.set_ylabel(role.title()+'\nkernel union (%)');axis.set_ylim(0,105)
            axis.set_title('Qualified matched profile' if p['eligible_for_matched_clean'] else 'DIAGNOSTIC: matched perturbation gate failed',fontsize=8)
        credit=records(folder/'queue_observations.jsonl')
        for role,color in (('prefill','C0'),('decode','C1')):
            data=[]
            for r in credit:
                if not begin<=r['time']<=end:continue
                for c in r['worker_credits'].values():
                    if c['role']==role:data.append((r['time']-begin,c['slots']))
            if data:axes[3].step(*zip(*data),where='post',label=role.title()+' reserved slots',color=color)
        axes[3].set_ylabel('Slots');axes[3].legend(fontsize=8)
        for path in folder.glob('memory_progress_*.jsonl'):
            data=[r for r in records(path) if begin<=r['time']<=end]
            if not data:continue
            role=data[0]['role'];sources.append(path)
            axes[4].step([r['time']-begin for r in data],[r['KV_free_blocks']*r['KV_block_bytes']/1024**3 for r in data],where='post',label=role.title()+' owner free KV')
        axes[4].set(xlabel='Seconds after first measured arrival',ylabel='Free KV (GiB)');axes[4].legend(fontsize=8)
        for axis in axes:
            axis.axvline(window['last_arrival']-begin,color='black',ls='--',lw=.8)
            axis.set_xlim(0,end-begin);axis.grid(alpha=.2)
        fig.suptitle(f'{model.title()} | {name}\nAll panels use this one profiled server start. Dashed line: last arrival. Kernel union is not SM occupancy.',fontsize=10)
        fig.tight_layout();outputs=[]
        for suffix in ('png','svg'):
            p=ROOT/'figures'/(model+'_same_start_kernel_timeline.'+suffix);fig.savefig(p,dpi=180);outputs.append(str(p.relative_to(ROOT)))
        plt.close(fig)
        sources=list(dict.fromkeys(s for s in sources if s.exists()))
        manifest['figures'].append(dict(name=model+'_same_start_kernel_timeline',outputs=outputs,
            exact_sources=[str(s.relative_to(ROOT)) for s in sources],source_hashes={str(s.relative_to(ROOT)):digest(s) for s in sources},
            scope='Graph-internal device intervals from the same run. Zero bins denote no recorded kernel under an active collector, not proof that arbitrary useful work fits.'))
    table('kernel_timeline_figure_data.csv',extra)
    write(ROOT/'figure_manifest.json',manifest)
    print(json.dumps(dict(kernel_figure_rows=len(extra))))


if __name__=='__main__':main()
