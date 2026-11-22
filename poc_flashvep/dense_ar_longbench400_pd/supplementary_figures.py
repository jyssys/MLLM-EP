"""Show causal controls and qualified shared-ROI activity with explicit limits."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT
from .figures import save
from .shared_activity_analysis import map_cupti
from .window_analysis import merge


def counterfactuals():
    output=json.loads((ROOT/'analysis/iso8_I3_exact_output_counterfactual_v1.json').read_text())
    load=json.loads((ROOT/'analysis/load_only_I0_O8_Cut2_counterfactual_v1.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for i,cell in enumerate(output['cells']):
        row=cell['Hybrid_over']['Token']['TTFT_p95_ms']
        v=row['estimate'];lo,hi=row['CI95']
        axes[0].errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color='#0072b2')
        axes[0].annotate(f'{v:.3f}',(i,v),xytext=(8,0),textcoords='offset points')
    axes[0].set(xticks=[0,1],xticklabels=['Output16','Output64'],
        ylabel='Hybrid / Token TTFT p95',title='Change only output contract | I3 | q1.253873')
    for i,label in enumerate(('low','high')):
        row=load['results'][label]['Layer_over_Token']['TTFT_p95_ms']
        v=row['estimate'];lo,hi=row['CI95']
        axes[1].errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color='#d55e00')
        axes[1].annotate(f'{v:.3f}',(i,v),xytext=(8,0),textcoords='offset points')
    axes[1].set(xticks=[0,1],xticklabels=['QPS0.910229','QPS8.292124'],
        ylabel='Layer / Token TTFT p95',title='Change only offered load | I0 | Cut2 | Output8')
    for ax in axes:
        ax.axhline(1,color='gray',ls='--');ax.set_xlim(-.4,1.5);ax.grid(alpha=.2)
    save(fig,'controlled_output_and_load_counterfactuals',
        'MEASURED ONLINE C0 | N16/start | Actual1/1/1 | Three paired independent starts,27-resample pointwise95% CI.\n'
        'Each panel keeps source,pair,requests,partition and normalized trace fixed. Output panel is an iso-row Hybrid control.\n'
        'Output reversal is confirmed on this contract. Load-only reversal is NOT confirmed. Neither panel establishes C3 natural superiority.',
        dict(evidence='MEASURED_ONLINE',output=output,load=load))


def activity():
    source=json.loads((ROOT/'analysis/recovered_native_shared_window_activity_v1.json').read_text())
    cells=[r for r in source['cells'] if r['status']=='MEASURED_KERNEL_ACTIVITY']
    assert len(cells)==2
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    x=np.arange(len(cells))
    for role,color,offset,label in [('prefill','#0072b2',-.18,'P'),('decode','#009e73',.18,'D')]:
        vals=[100*r['summaries'][role]['kernel_busy_fraction']['mean'] for r in cells]
        cis=[r['summaries'][role]['kernel_busy_fraction']['CI95'] for r in cells]
        axes[0].bar(x+offset,vals,.36,color=color,label=label)
        axes[0].errorbar(x+offset,vals,yerr=[[v-100*c[0] for v,c in zip(vals,cis)],
            [100*c[1]-v for v,c in zip(vals,cis)]],fmt='none',ecolor='black',capsize=3)
    axes[0].set(xticks=x,xticklabels=[str(r['QPS']) for r in cells],xlabel='Offered QPS',
        ylabel='Observed kernel union / shared ROI (%)',ylim=(0,105),title='Qualified common P/D host-time window')
    axes[0].legend()
    for field,color,label in [('P_waiting','#0072b2','P waiting'),('D_active','#009e73','D active'),
            ('D_remote','#e69f00','D remote-KV waiting'),('D_ready','#cc79a7','D decode-ready waiting')]:
        role='prefill' if field=='P_waiting' else 'decode'
        key={'P_waiting':'waiting_total','D_active':'active_decode','D_remote':'remote_KV_waiting','D_ready':'decode_ready_waiting'}[field]
        axes[1].plot(x,[r['summaries'][role]['queue_start_means'][key] for r in cells],'o-',label=label,color=color)
    axes[1].set(xticks=x,xticklabels=[str(r['QPS']) for r in cells],xlabel='Offered QPS',
        ylabel='Time-weighted requests in shared ROI',title='Queue states are not GPU occupancy')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'recovered_native_shared_ROI_activity_and_queues',
        'MEASURED KERNEL ACTIVITY + TRACE ANALYSIS | Natural400 | Three independent clean/profile/clean triplets per QPS.\n'
        'Both-role observer5% gates pass. Busy CI uses3 profile starts,not kernel records. Common READ/compute Native,not original motivation runtime.\n'
        'Short shared ROIs do not establish full-run saturation,all-engine physical idle,a future executable window,or sustainable capacity.',source)
    item=cells[1]['results'][0]
    folder=ROOT/'raw'/item['name'];left=item['shared_host_left_ns'];right=min(left+1_000_000_000,item['shared_host_right_ns'])
    fig,ax=plt.subplots(figsize=(12,4))
    for index,role in enumerate(('prefill','decode')):
        roi=json.loads((folder/f'activity_{role}/actual_ROI.json').read_text())
        a,b=map_cupti(roi,left),map_cupti(roi,right)
        sets={'kernel':[],'copy':[]}
        with (folder/f'activity_{role}/kernels.csv').open() as stream:
            for row in csv.DictReader(stream):
                l,r=max(a,int(row['start_ns'])),min(b,int(row['end_ns']))
                if r<=l:continue
                key='kernel' if row['kind']=='kernel' else 'copy' if row['kind'] in ('memcpy','memcpy2') else None
                if key:sets[key].append((l,r))
        for key,dy,color in [('kernel',.02,'#0072b2'),('copy',-.17,'#e69f00')]:
            spans=[((l-a)/(b-a),(r-l)/(b-a)) for l,r in merge(sets[key])]
            ax.broken_barh(spans,(1-index+dy,.13),facecolors=color,label=key if index==0 else None)
    ax.set(xlim=(0,1),xlabel='Fraction of the displayed1-second shared host window',
        yticks=[.95,-.05],yticklabels=['P','D'],title='Observed kernels and copies | Recovered Native | QPS2.8')
    ax.legend();ax.grid(axis='x',alpha=.2)
    save(fig,'recovered_native_observed_kernel_copy_timeline',
        'MEASURED KERNEL ACTIVITY | One illustrative1-second segment of a qualified triplet.\n'
        'Intervals are clipped with the recorded two-anchor host/CUPTI clock mapping. Blue:kernel union. Orange:observed copy union.\n'
        'Blank regions are not a certification of all-engine idle or a decode-safe executable window. See shared-ROI analysis for n3 summaries.',
        dict(evidence='MEASURED_KERNEL_ACTIVITY',source=item['name'],display_host_interval_ns=[left,right]))


def demand_rules():
    source=json.loads((ROOT/'analysis/minimal_calibrated_demand_rules_v1.json').read_text())
    keys=['serial_sum','bottleneck','load_blend_at_0_5']
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    x=np.arange(3)
    axes[0].bar(x,[100*source['summary'][k]['accuracy'] for k in keys],color=['#f58518','#4c78a8','#72b7b2'])
    axes[0].set(ylabel='Correct clear-winner classification (%)',ylim=(0,100),title='38 clear cells; no prospective rule confirmation')
    axes[1].bar(x,[source['summary'][k]['regret_mean_percent'] for k in keys],color=['#f58518','#4c78a8','#72b7b2'])
    axes[1].set(ylabel='Mean measured TTFT regret (%)',title='59 safety-valid cells; not goodput regret')
    for ax in axes:ax.set_xticks(x,['Serial sum','Bottleneck','Load blend']);ax.grid(axis='y',alpha=.2)
    save(fig,'exploratory_minimal_demand_rule_comparison',
        'OFFLINE MODEL | Serial-calibrated host-service costs. Rule forms are assessed after existing matrix results.\n'
        'No label refit. The signed bottleneck score compresses static costs and known controlled output,not one direct online observable.\n'
        'Natural EOS,C3 packing and decode-safe online admission remain unvalidated. No serving-speedup claim.',source)


def kernel_gaps():
    source=json.loads((ROOT/'analysis/recovered_native_shared_window_activity_v1.json').read_text())
    fig,ax=plt.subplots(figsize=(9,5))
    records=[]
    for cell,color in zip(source['cells'],('#009e73','#0072b2')):
        if cell['status']!='MEASURED_KERNEL_ACTIVITY':continue
        for index,start in enumerate(cell['results']):
            gaps=np.sort(start['roles']['decode']['interior_observed_kernel_gaps_ms'])
            assert len(gaps)
            ax.step(gaps,np.arange(1,len(gaps)+1)/len(gaps),where='post',color=color,alpha=.6,
                label=f'QPS{cell["QPS"]},3 independent profiles' if index==0 else None)
            records.append(dict(name=start['name'],QPS=cell['QPS'],gap_count=len(gaps),
                distribution=start['roles']['decode']['interior_kernel_gap_distribution_ms']))
    ax.set(xscale='log',xlabel='Interior observed D kernel gap (ms)',ylabel='Within-profile empirical CDF',
        title='Kernel inactivity distribution is not a decode-safe executable budget')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'recovered_native_observed_kernel_gap_distribution',
        'MEASURED KERNEL ACTIVITY | Three qualified native profiles/QPS | Each line is one profile,not an independent gap sample CI.\n'
        'Conservative shared P/D host ROI. Exclude boundary-censored gaps. Copy or control activity can occur inside these kernel gaps.\n'
        'No ready partial state or future deadline predictor was validated. Decode-safe budget and offload fit-rate remain NOT_IDENTIFIED/NOT_RUN.',
        dict(evidence='MEASURED_KERNEL_ACTIVITY',profiles=records,safe_window='NOT_IDENTIFIED'))


if __name__=='__main__':
    counterfactuals()
    activity()
    demand_rules()
    kernel_gaps()
