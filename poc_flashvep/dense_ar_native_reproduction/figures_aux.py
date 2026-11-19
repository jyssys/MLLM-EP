"""Separate sampled host queues from qualified kernel activity figures."""
import json
import numpy as np
from .common import ROOT
from .figures import start_interval


def save(fig,stem):
    for suffix in ('png','pdf'):fig.savefig(ROOT/'figures'/(stem+'.'+suffix),dpi=180)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=json.loads((ROOT/'analysis/Main_full_analysis.json').read_text())
    native=[r for r in table['per_start'] if r['method']=='Native']
    keys=[('prefill','waiting_total','P waiting requests'),
        ('decode','remote_KV_waiting','D remote-KV waiting'),
        ('decode','decode_ready_including_first_query','D ready waiting, including first query'),
        ('decode','active_decode','D active decode requests')]
    fig,axes=plt.subplots(2,2,figsize=(10,6.5))
    for ax,(role,key,label) in zip(axes.flat,keys):
        xs=[];ys=[];low=[];high=[]
        for rate in sorted({r['offered_QPS'] for r in native}):
            values=[r['timeline']['time_weighted_queues'][role][key]['mean'] for r in native
                if r['offered_QPS']==rate and 'mean' in r['timeline']['time_weighted_queues'][role][key]]
            if not values:continue
            ci=start_interval(values);xs.append(rate);ys.append(float(np.mean(values)))
            low.append(ci[0] if ci else np.nan);high.append(ci[1] if ci else np.nan)
        ax.plot(xs,ys,'o-',color='#009e73')
        if np.isfinite(low).any():ax.errorbar(xs,ys,yerr=[np.array(ys)-low,np.array(high)-ys],fmt='none',capsize=3,color='#009e73')
        ax.set(xlabel='Offered QPS',ylabel=label);ax.grid(alpha=.2)
    fig.suptitle('Native PD | Frozen LongBench v1 Mixed200 | B8192/S1024')
    fig.text(.02,.015,'TRACE_ANALYSIS: time-weighted scheduler snapshots. Not GPU busy or physical idle. Missing fields are omitted. n=1 has no CI.',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.96));save(fig,'longbench200_native_scheduler_queues');plt.close(fig)
    profiles=[]
    for file in sorted((ROOT/'analysis').glob('Native_online_kernel_activity_q*_v1.json')):
        profiles.append(json.loads(file.read_text()))
    if not profiles:return
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    for role,color,marker in (('prefill','#0072b2','o'),('decode','#cc79a7','^')):
        good=[p for p in profiles if p['observer_gate_passed']]
        bad=[p for p in profiles if not p['observer_gate_passed']]
        if good:ax.plot([p['offered_QPS'] for p in good],[100*p['kernel_activity']['roles'][role]['kernel_busy_fraction'] for p in good],marker=marker,color=color,label=role+' qualified')
        if bad:ax.scatter([p['offered_QPS'] for p in bad],[100*p['kernel_activity']['roles'][role]['kernel_busy_fraction'] for p in bad],marker=marker,edgecolors=color,facecolors='none',label=role+' diagnostic: observer gate failed')
    ax.set(xlabel='Offered QPS',ylabel='Kernel union / common ROI (%)',ylim=(0,105))
    ax.set_title('Native kernel activity | Prespecified initial-window capture\nOne profiled start per point, no start-level CI',fontsize=10)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.text(.02,.015,'Kernel inactivity can include DMA and control waits. It is not a decode-safe window.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'longbench200_native_kernel_busy');plt.close(fig)
    for p in profiles:
        fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
        for ax,role in zip(axes,('prefill','decode')):
            row=p['kernel_activity']['roles'][role];duration=row['window_seconds']
            bins=row['kernel_union_bins_1s'];starts=np.array([b['start_seconds'] for b in bins])
            width=np.array([b['end_seconds']-b['start_seconds'] for b in bins])
            busy=np.array([b['kernel_union_seconds'] for b in bins])
            ax.bar(starts,100*busy/width,width=width,align='edge',color='#0072b2' if role=='prefill' else '#cc79a7')
            ax.set(ylabel=role+' kernel busy (%)',ylim=(0,105));ax.grid(alpha=.2)
        axes[-1].set_xlabel('Seconds from the common GPU observation window')
        fig.suptitle(f'Native PD, QPS{p["offered_QPS"]:g} | '+('Observer point-estimate gate PASS' if p['observer_gate_passed'] else 'DIAGNOSTIC: observer gate failed'))
        fig.text(.02,.015,'Separate GPUs on separate rows. One-second kernel unions are not SM utilization or safe executable budgets.',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.95));save(fig,f'longbench200_native_kernel_timeline_q{p["offered_QPS"]:g}');plt.close(fig)


if __name__=='__main__':main()
