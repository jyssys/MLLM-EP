"""Plot measured online latency, goodput, stage overlaps, and control comparisons."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from statistics import median

PALETTE = ('#315579','#1c8275','#a56d50','#758693','#9461a2')
METHODS = {
    'img_local_q1': 'Local QPS 1', 'img_encoder_q1': 'E helper QPS 1',
    'img_prefill_q1': 'P helper QPS 1', 'img_visual_q1': 'Visual-first E QPS 1',
    'img_adaptive_q1': 'Admission-gated E QPS 1', 'img_ve_2d_q1': 'E+P / 2D QPS 1',
    'img_local_q25': 'Local QPS 2.5', 'img_encoder_q25': 'E helper QPS 2.5',
    'eager_local_q1': 'Eager Local', 'eager_encoder_q1': 'Eager E helper',
}


def read(path):
    return json.loads(path.read_text())


def save(fig,out,name,note):
    fig.text(.08,.025,note,fontsize=8,color='#425465')
    fig.savefig(out/(name+'.png'),facecolor='white',dpi=160,bbox_inches='tight')
    plt.close(fig)


def performance(trials,out):
    arms=[r for r in trials if r['run'] in METHODS and r['attention_engine']=='graph']
    if not arms:
        return
    names=[METHODS[r['run']] for r in arms]
    fig,axes=plt.subplots(2,2,figsize=(12,7.4))
    fig.subplots_adjust(left=.2,hspace=.34,wspace=.72,bottom=.13)
    for ax,metric,quantile,title in zip(axes.flat,['ttft_ms','tpot_ms','itl_ms','E2E'],
            ['p95','p95','p99','p95'],['TTFT p95 (ms)','TPOT p95 (ms)','ITL p99 (ms)','E2E p95 (ms)'],strict=True):
        key='e2e_ms' if metric=='E2E' else metric
        observed=[r['latencies'][key][quantile] for r in arms]
        ax.barh(names,observed,color=[PALETTE[i%len(PALETTE)] for i in range(len(arms))])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Matched Qwen2.5-VL online latency')
    save(fig,out,'online_latency','Actual finite arrival traces and natural output. Local and EADA share all three H200s and the same engine.')

    fig,ax=plt.subplots(figsize=(9,4.6))
    fig.subplots_adjust(left=.27,bottom=.19)
    goodput=[r['goodput_requests_per_second'] for r in arms]
    bars=ax.barh(names,goodput,color=[PALETTE[i%len(PALETTE)] for i in range(len(arms))])
    ax.invert_yaxis()
    ax.set_xlabel('Joint-SLO goodput (req/s)')
    ax.set_title('Observed goodput at matched offered load')
    ax.bar_label(bars,fmt='%.3f',fontsize=8)
    ax.margins(x=.15)
    save(fig,out,'online_goodput','Historical SLO TTFT ≤5 s, every ITL ≤34 ms, and TPOT ≤33 ms. Failures remain in the offered population.')


def capacity(trials,out):
    vals=[r for r in trials if r['run'].startswith(('pressure_','pressure2_'))]
    if not vals:
        return
    names=[f"{'2' if r['run'].startswith('pressure2_') else '4'} GiB\n"
           f"{'Local' if r['helper']=='none' else 'E helper'}" for r in vals]
    fig,axes=plt.subplots(2,2,figsize=(9,6.5))
    fig.subplots_adjust(bottom=.18,wspace=.5,hspace=.52)
    values=([r['max_prepared_batch'] for r in vals],[r['complete'] for r in vals],
            [r['stage_kv_pools'][0]['peak_active_pages'] for r in vals],
            [r['page_wait_request_count'] for r in vals])
    for ax,title,data in zip(axes.flat,['Max prepared batch','Completion count','Peak D pages used','Requests waiting for KV pages'],
            values,strict=True):
        bars=ax.bar(names,data,color=PALETTE[:len(names)])
        ax.set_title(title)
        ax.bar_label(bars,fmt='%d')
        ax.margins(y=.18)
    fig.suptitle('Controlled online KV pressure')
    save(fig,out,'online_capacity','Each matched pair has the same D arena budget. Live KV, observed concurrency, and request completion have separate scales.')


def timelines(path,paths,seconds):
    if not paths:
        return
    rows=[]
    for file in paths:
        d=read(file)
        t=d['start_monotonic']; stop=t+seconds
        def add(rank,label,a,b):
            if b>t and a<stop and b>=a:
                rows.append({'arm':file.parent.name,'stage':rank,'activity':label,
                             'begin':max(0.,a-t),'end':min(seconds,b-t)})
        for r in d['requests']:
            if 'e_gpu_begin' in r and 'e_end' in r:
                add('E','Vision encode',r['e_gpu_begin'],r['e_end'])
            if 'p_start' in r and 'p_end' in r:
                add('P','Prefill',r['p_start'],r['p_end'])
        for step in d['decode_steps']:
            add('D','Decode',step['begin'],step['end'])
            if step['assisted'] and d['helper'] in ('encoder','idle-encoder'):
                add('E','Helper-dependent D step',step['begin'],step['end'])
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=('arm','stage','activity','begin','end'),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def show_timeline(path,out):
    if not path.exists():
        return
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    if not rows:
        return
    names=list(dict.fromkeys(row['arm'] for row in rows))
    fig,axes=plt.subplots(len(names),1,figsize=(11,2.5*len(names)),squeeze=False,sharex=True)
    fig.subplots_adjust(hspace=.5,bottom=.18)
    for name,ax in zip(names,axes[:,0],strict=True):
        for row in rows:
            if row['arm']!=name:
                continue
            low=float(row['begin']); high=float(row['end'])
            pos={'E':2,'P':1,'D':0}[row['stage']]
            help_claim=row['activity']=='Helper-dependent D step'
            ax.broken_barh([(low,high-low)],(pos-.23,.46),
                           facecolor=PALETTE[2] if help_claim else PALETTE[pos],
                           alpha=.4 if help_claim else .8)
        ax.set_yticks([0,1,2],['Decode','Prefill','Encoder'])
        ax.set_title(METHODS.get(name,name),loc='left')
    axes[-1,0].set_xlabel('Host monotonic elapsed seconds')
    fig.suptitle('Actual E/P/D stage enclosures')
    save(fig,out,'online_stage_timeline','Orange marks steps that depend on E, using the full host Decode step boundary. Separate native CUPTI trace measures kernels.')


def text_controls(pairs,out):
    labels={'txt_matched_prefill':'Text LLM · matched token shape','txt_real_prefill':'Text LLM · LongBench',
            'mm_image_replay_encoder':'MLLM · images / fixed output',
            'mm_text_encoder':'MLLM · text-only control'}
    selected=[row for row in pairs if row['arm'].removesuffix('_retry') in labels]
    if not selected:
        return
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    fig.subplots_adjust(left=.26,bottom=.21,wspace=.65)
    for key,ax,title in zip(('tpot_ms','itl_ms'),axes,('TPOT p95 change (%)','ITL p99 change (%)'),strict=True):
        values=[row['latency_change_fraction'][key] for row in selected]
        ax.barh([labels[row['arm'].removesuffix('_retry')] for row in selected],
                [x*100 for x in values],color=PALETTE[:len(values)])
        ax.axvline(0,color='#424852',linewidth=1)
        ax.tick_params(axis='y',labelsize=8)
        ax.set_title(title)
    fig.suptitle('P-assisted text PD and E-assisted MLLM EPD')
    save(fig,out,'llm_mllm_control','Within-workload relative changes. Shape-matched fixed-output and independent natural texts retain separate baselines.')


def decode_profile(witness_path,out):
    if witness_path is None:
        return
    profiles=read(witness_path)['diagnostic_full28_stage_event_profile']
    if not profiles:
        return
    x=[f"B{row['batch']}\n{sum(row['prompt_rows']):,} rows" for row in profiles]
    groups={'input_norm_qkv_rope_and_kv_append':'Input norm, QKV, RoPE, KV append',
            'attention':'Attention','output_projection_residual':'Output projection and residual',
            'mlp_norm_residual':'MLP, norm and residual','last_norm_head_argmax_ms':'Final norm, LM head and sampling'}
    fig,ax=plt.subplots(figsize=(8.6,5.3))
    totals=[0.] * len(profiles)
    for (key,label),color in zip(groups.items(),PALETTE,strict=True):
        values=[median(step['stage_totals_ms'][key] if key in step['stage_totals_ms'] else step[key]
                       for step in row['decode_steps']) for row in profiles]
        ax.bar(x,values,label=label,bottom=totals,color=color)
        totals=[a+b for a,b in zip(totals,values,strict=True)]
    for i,row in enumerate(profiles):
        share=median(step['attention_gpu_enclosure_ms']/step['decode_gpu_enclosure_ms'] for step in row['decode_steps'])
        ax.text(i,totals[i],f"Attention {100*share:.1f}%",ha='center',va='bottom',fontsize=8)
    ax.margins(y=.17)
    ax.set_ylabel('Median GPU event enclosure (ms)')
    ax.set_title('Qwen2.5-VL Full28 Decode stage enclosures')
    ax.legend(loc='upper left',fontsize=8,ncol=2,bbox_to_anchor=(0,-.16))
    fig.subplots_adjust(bottom=.29)
    save(fig,out,'decode_event_breakdown','Diagnostic only. Uses real image prompts with fixed generation and local Eager FA3. GPU events may enclose submission bubbles.')


def restart_ranges(summary,out):
    if summary is None or len(summary['restart_blocks'])<2:
        return
    measures={'ttft_ms':'TTFT p95','tpot_ms':'TPOT p95','itl_ms':'ITL p99','goodput':'Joint-SLO goodput'}
    fig,ax=plt.subplots(figsize=(8.8,4.8))
    fig.subplots_adjust(left=.26,bottom=.21)
    for i,(key,name) in enumerate(measures.items()):
        samples=summary['relative_changes'].get(key,{}).get('paired',[])
        if samples:
            estimates=[100*sample for sample in samples]
            ax.hlines(i,min(estimates),max(estimates),color=PALETTE[i],linewidth=2)
            ax.scatter(estimates,[i]*len(estimates),c=PALETTE[i],s=44,zorder=3)
        ax.text(-.03,i,name,transform=ax.get_yaxis_transform(),ha='right',va='center')
    ax.axvline(0,color='#5a6571',linewidth=1)
    ax.set_yticks([])
    ax.set_ylim(3.5,-.5)
    ax.set_xlabel('EADA change relative to Local (%)')
    ax.set_title('Three independent QPS 2.5 paired restarts')
    save(fig,out,'online_restart_ranges','Markers are independent 200-arrival paired runs. Lines show the observed range across restarts.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--timeline',required=True,type=Path)
    p.add_argument('--capture-timeline',type=Path,nargs='*')
    p.add_argument('--stage-witness',type=Path)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    result=read(args.summary)
    timelines(args.timeline,args.capture_timeline,12.)
    performance(result['trials'],args.out)
    restart_ranges(result.get('medium_load_independent_restart_summary'),args.out)
    capacity(result['trials'],args.out)
    text_controls(result['paired'],args.out)
    show_timeline(args.timeline,args.out)
    decode_profile(args.stage_witness,args.out)


if __name__=='__main__':
    main()
