"""Plot measured historical/current families without a causal bridge claim."""
import json
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write
from .figures import save,COLORS,read


def ratio():
    old=read('historical_winner_source_guard_audit_v1.json')
    current=read('historical_bridge_independent_start_statistics.json')
    data=[dict(label='Historical BI1\nInput8000 / Output16',**old['TTFT_p95_ratio'])] if old else []
    for key,label in [('current_BI0_block128_all3','Current BI0 / KV128\nInput8000 / Output16'),
                      ('current_BI0_block16_heldout3','Current BI0 / KV16\nInput8000 / Output16')]:
        if current.get(key):data.append(dict(label=label,**current[key]['Layer_over_Token']['TTFT_p95_ms']))
    main=read('full200_confirmation_statistics.json')
    for row in main.get('comparisons',[]):
        if row['cut']==8:data.append(dict(label='Current BI0\nLongBench200 Natural',**row['metrics']['TTFT_p95_ms']))
    fig,ax=plt.subplots(figsize=(11,4))
    for i,r in enumerate(data):
        ax.plot(i,r['ratio'],'o',color=COLORS['Layer'])
        if r.get('CI95'):
            lo,hi=r['CI95'];ax.errorbar(i,r['ratio'],yerr=[[r['ratio']-lo],[hi-r['ratio']]],fmt='none',capsize=5,color=COLORS['Layer'])
        ax.annotate('n='+str(r['starts']),(i,r['ratio']),xytext=(5,7),textcoords='offset points',fontsize=8)
    ax.axhline(1,color='black',ls='--');ax.set(xticks=range(len(data)),xticklabels=[r['label'] for r in data],ylabel='Layer / Token TTFT p95')
    ax.grid(axis='y',alpha=.2)
    return save(fig,'historical_current_axis_ttft_ratio',
        'MEASURED families and reused Historical starts | Last8,actual1/1/1. n is independent server starts, not requests. N1 has no start-level CI.\n'
        'KV16 uses three new held-out seeds. Historical/current source,block,transport and credit semantics differ. Across-family changes are not a numerical-flag-only or workload-only causal effect.')


def exact_chains():
    source=read('exact_source_chain_decomposition_v1.json')
    destination=read('exact_destination_chain_decomposition_v1.json')
    left=source['Layer_minus_Token_exact_queue_parts_ms']
    right=destination['historical_Layer_minus_Token_exact_weighted_parts_ms']
    labels_left={'initial_P_dispatch':'Initial dispatch',
        'weighted_prior_source_host_service':'Prior P host service',
        'weighted_arrival_gaps':'Arrival gaps','weighted_source_credit_extension':'Credit extension',
        'weighted_eligible_host_admission_gaps':'Eligible host gaps',
        'mean_current_dispatch_to_prepare':'Current prepare offset','negative_mean_arrival_offset':'Mean arrival offset'}
    labels_right={'initial_D_submit':'Initial D submit',
        'weighted_prior_D_preparation':'Prior D preparation',
        'weighted_prior_decode_residency':'Prior decode residency',
        'weighted_prior_delivery':'Prior delivery',
        'weighted_D_waiting_for_ready_supply':'Wait for P-ready supply',
        'weighted_eligible_destination_host_gap':'Eligible host gaps'}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,parts,labels,title in [(axes[0],left,labels_left,'Natural LongBench200 / Last4 / seed2061'),
                                (axes[1],right,labels_right,'Historical Input8000 / Output16 / Last8 / n3')]:
        keys=list(parts);values=[parts[k] for k in keys]
        ax.barh(range(len(keys)),values,color=[COLORS['Layer'] if v>0 else COLORS['Token'] for v in values])
        ax.set(yticks=range(len(keys)),yticklabels=[labels[k] for k in keys],xlabel='Layer minus Token weighted contribution (ms)',title=title)
        ax.axvline(0,color='black',lw=.8);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        for i,v in enumerate(values):ax.annotate(f'{v:+.1f}',(v,i),xytext=(3 if v>=0 else -3,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=8)
        ax.margins(x=.18)
    return save(fig,'exact_source_destination_queue_chains',
        'TRACE_ANALYSIS | Exact unrolling of measured host event chains. Left sums to the mean P queue difference +2711.089ms. Right sums to D-submit time difference -480.601ms.\n'
        'These are separate workload/cut families. Terms are accounting contributions,not independently randomized effects,exclusive GPU times or physical-idle windows.')


def transport_factorial():
    data=read('historical_bridge_independent_start_statistics.json').get('current_BI1_coalescing_factorial')
    if not data:return None
    metric=data['metrics']['TTFT_p95_ms'];fig,ax=plt.subplots(figsize=(8,4))
    for i,(key,label) in enumerate([('no_coalescing','No coalescing'),('coalescing','Exact-byte coalescing')]):
        r=metric[key];lo,hi=r['CI95']
        ax.errorbar(i,r['ratio'],yerr=[[r['ratio']-lo],[hi-r['ratio']]],fmt='o',capsize=5,color=COLORS['Layer'])
    ax.set(xticks=[0,1],xticklabels=['Coalescing flag OFF','Coalescing flag ON'],ylabel='Layer / Token TTFT p95',title='BI1 host-flag diagnostic | Input8000 / Output16 / Last8')
    ax.axhline(1,color='black',ls='--');ax.grid(axis='y',alpha=.2)
    return save(fig,'BI1_exact_byte_coalescing_axis_factorial',
        'DIAGNOSTIC | Three paired starts per factor/axis,27 start-bootstrap resamples. Actual descriptors stay1764 before/after. Physical coalescing intervention NOT_REALIZED.\n'
        f'Ratio-of-axis-ratios {metric["relative_axis_ratio_change"]:.4f},95% CI [{metric["CI95"][0]:.4f},{metric["CI95"][1]:.4f}]. Includes one discovery plus two new seeds. Native-relative safety NOT_MEASURED.')


def historical_guard():
    old=read('historical_winner_source_guard_audit_v1.json');values=old['guard_wait_by_axis_ms']
    fig,ax=plt.subplots(figsize=(9,4));axes=['Native','Layer','Token'];bottom=np.zeros(3)
    for field,label in [('prior_D_preparation_guard_overlap_ms','Prior D preparation guard'),
        ('prior_D_decode_residency_guard_overlap_ms','Prior decode-residency guard'),
        ('prior_delivery_guard_overlap_ms','Prior delivery guard'),('other_host_wait_ms','Other host wait')]:
        y=np.asarray([values[a][field]/1000 for a in axes]);ax.bar(axes,y,bottom=bottom,label=label);bottom+=y
    for index,axis in enumerate(axes):
        raw=np.asarray([r['mean_guard_wait_parts_ms']['ready_to_D_submit_ms']/1000 for r in old['starts'] if r['axis']==axis])
        draws=[float(raw[list(ids)].mean()) for ids in itertools.product(range(3),repeat=3)]
        lo,hi=np.percentile(draws,[2.5,97.5]);center=float(raw.mean())
        ax.errorbar(index,center,yerr=[[center-lo],[hi-center]],fmt='none',ecolor='black',capsize=5)
    ax.set(ylabel='Mean Ready to D-submit host wait (s)',title='Historical Input8000 / Output16 | Last8 | QPS6 | N72 x3')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    return save(fig,'historical_ready_D_guard_overlap',
        'HISTORICAL TRACE_ANALYSIS | Disjoint overlap with prior host-cohort states. These bars are not GPU time, physical idle or an ACK/guard-removal intervention.\n'
        'Error bars:95% start-bootstrap CI of total wait only,27 draws. Native uses full prefill,not Last8. All three published paired starts remain included.')


def balance():
    old=read('historical_winner_source_guard_audit_v1.json');cur=read('historical_current_ready_D_comparison_v1.json')
    fig,ax=plt.subplots(figsize=(8,6));maximum=0.
    for axis,r in old.get('P_D_host_balance_by_axis',{}).items():
        x=r['P_submit_to_ready_ms'];y=r['D_whole_request_guard_mean_ms'];maximum=max(maximum,x,y)
        ax.scatter(x,y,color=COLORS[axis],marker='s');ax.annotate('Historical BI1 '+axis,(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    for row in cur.get('rows',[]):
        costs=row['canonical']['rows'];x=1000*np.mean([r['P_service'] for r in costs]);y=1000*np.mean([r['D_first_service']+r['D_decode'] for r in costs]);maximum=max(maximum,x,y)
        ax.scatter(x,y,color=COLORS[row['axis']],marker='o',alpha=.6)
        ax.annotate(f'Current BI{int(row["BI"])} {row["axis"]}',(x,y),xytext=(4,-10),textcoords='offset points',fontsize=7)
    ax.plot([0,maximum*1.1],[0,maximum*1.1],'k--',lw=1);ax.set(xlabel='P host service demand mean (ms/request)',ylabel='D whole-request host demand mean (ms/request)',xlim=(0,maximum*1.1),ylim=(0,maximum*1.1))
    ax.grid(alpha=.2)
    return save(fig,'historical_current_P_D_host_balance',
        'TRACE_ANALYSIS | Host service envelopes include launch,transfer and scheduler work. This is not a GPU-utilization or sustainable-capacity map.\n'
        'Points from separate executions are not a shared physical timeline. Equality is an explanatory reference,not a fitted winner threshold.')


def main():
    if not (ROOT/'analysis/historical_winner_source_guard_audit_v1.json').exists():return
    manifest=[ratio(),historical_guard(),balance(),exact_chains()]
    transport=transport_factorial()
    if transport:manifest.append(transport)
    write(ROOT/'analysis/historical_bridge_figures.json',dict(figures=manifest))
    print('HISTORICAL_FIGURES',len(manifest))


if __name__=='__main__':main()
