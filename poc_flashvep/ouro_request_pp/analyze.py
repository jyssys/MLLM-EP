"""Keep clean serving metrics separate from GPU brackets and heavy traces."""
import collections
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contract import ROOT, write


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def union(intervals):
    if not intervals:return 0.
    total=0.;left,right=sorted(intervals)[0]
    for a,b in sorted(intervals)[1:]:
        if a>right:
            total+=right-left;left,right=a,b
        else:right=max(right,b)
    return total+right-left


def summarize(d):
    outs=d['outputs']
    ttft=[r['token_times_s'][0] for r in outs]
    end=[r['token_times_s'][-1] for r in outs]
    tpot=[(r['token_times_s'][-1]-r['token_times_s'][0])/(len(r['ids'])-1) for r in outs if len(r['ids'])>1]
    events=d['events']
    comm_count=collections.Counter();comm_bytes=collections.Counter()
    if events:
        for e in events:
            comm_count[e['transfer_kind'] or 'local']+=1
            comm_bytes[e['transfer_kind'] or 'local']+=e['transfer_bytes']
    else:
        for c,b in zip(d['transfer_counts'],d['transfer_bytes']):
            comm_count.update(c);comm_bytes.update(b)
    return dict(placement=d['placement'],depth=d['depth'],C=d['C'],P=d.get('case',{}).get('P',d['input_lengths'][0]),O=d['output_len'],
        wall_s=d['wall_s'],requests_per_s=len(outs)/d['wall_s'],
        tokens_per_s=sum(len(r['ids']) for r in outs)/d['wall_s'],
        TTFT_P50_ms=float(np.median(ttft)*1000),TTFT_P90_ms=float(np.quantile(ttft,.9)*1000),
        TTFT_mean_ms=float(np.mean(ttft)*1000),
        E2E_P50_ms=float(np.median(end)*1000),E2E_P90_ms=float(np.quantile(end,.9)*1000),
        E2E_mean_ms=float(np.mean(end)*1000),
        TPOT_P50_ms=float(np.median(tpot)*1000) if tpot else None,
        TPOT_mean_ms=float(np.mean(tpot)*1000) if tpot else None,
        TPOT_P90_ms=float(np.quantile(tpot,.9)*1000) if tpot else None,
        hidden_handoffs=comm_count['hidden'],hidden_bytes=comm_bytes['hidden'],
        token_return_count=comm_count['token_return'],token_return_bytes=comm_bytes['token_return'],
        terminal_scalar_D2H_count=sum(len(r['ids']) for r in outs),
        input_H2D_bytes=comm_bytes['input_H2D'],
        local_KV_staging_logical_bytes=sum(d.get('local_KV_staging_logical_bytes',[0])),
        graph_captures=d.get('graph_captures_during_run'),
        max_peak_HBM_GiB=max(m['peak_bytes'] for m in d['memory'])/2**30,
        rank_weights_GiB=[m['weights_bytes']['total']/2**30 for m in d['memory']],
        rank_KV_GiB=[m['KV_bytes']/2**30 for m in d['memory']],
        rank_queue_wait_ms=d.get('rank_queue_wait_ms'),
        rank_worker_host_ms=d.get('rank_host_worker_ms'),
        graph_warmup_wall_s=d.get('exact_length_warmup_wall_s'),
        evidence='LIVE_MEASURED',trace=d['trace'],graphs=d['CUDA_graph_decode'])


def overlap(d):
    events=d['events'];best=None;total=0.
    ordered=sorted(events,key=lambda e:e['GPU_start_s'])
    for i,a in enumerate(ordered):
        for b in ordered[i+1:]:
            if b['GPU_start_s']>=a['GPU_end_s']:break
            if a['rank']==b['rank'] or a['id']==b['id']:continue
            lo=max(a['GPU_start_s'],b['GPU_start_s']);hi=min(a['GPU_end_s'],b['GPU_end_s'])
            margin=(a['anchor_uncertainty_ms']+b['anchor_uncertainty_ms'])/1000
            legal=hi-lo-margin
            if legal>0 and (best is None or legal>best['conservative_overlap_s']):
                best=dict(left_request=a['id'],right_request=b['id'],ranks=[a['rank'],b['rank']],
                    steps=[a['step'],b['step']],loops=[a['loop'],b['loop']],
                    conservative_overlap_s=legal,interval_s=[lo,hi],margin_s=margin)
    by_rank=[]
    for rank in range(4):
        es=[e for e in events if e['rank']==rank]
        span=union([(e['GPU_start_s'],e['GPU_end_s']) for e in es])
        stage_span=union([(e.get('GPU_stage_start_s',e['GPU_start_s']),e.get('GPU_stage_end_s',e['GPU_end_s'])) for e in es])
        by_rank.append(dict(rank=rank,physical_GPU=rank+4,stage_calls=len(es),
            compute_stream_bracket_ms=sum(e['compute_ms'] for e in es),
            head_stream_bracket_ms=sum(e['head_ms'] for e in es),
            copy_stream_bracket_ms=sum(e['transfer_ms'] for e in es),
            queue_wait_ms=sum(e['queue_ms'] for e in es),
            compute_bracket_union_s=span,stage_bracket_union_s=stage_span,
            no_stage_bracket_fraction=1-stage_span/d['wall_s'],
            actual_GPU_kernel_idle='Nsight lane only. Stage brackets can contain launch gaps.'))
    return dict(best_overlap=best,ranks=by_rank,
        caveat='Cross-GPU CUDA-event anchors have host calibration uncertainty. Confirm physical kernels with Nsight.')


def paired_ratio(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<3:
        return dict(layer_over_loop_speedup=float(np.median(a)/np.median(b)),CI95=None,
                    unit='Fewer than 3 paired workload repeats. Interval not reported.')
    idx=np.random.default_rng(42).integers(0,len(a),(2000,len(a)))
    return dict(layer_over_loop_speedup=float(np.median(a)/np.median(b)),
        CI95=np.quantile(np.median(a[idx],axis=1)/np.median(b[idx],axis=1),[.025,.975]).tolist(),
        unit='Paired complete workload repeat, not token/rank/layer')


def main():
    results=[];details=[]
    for f in sorted((ROOT/'raw').glob('*/*json*')):
        d=read(f);s=summarize(d);s['file']=str(f.relative_to(ROOT));results.append(s)
        if d['events']:
            o=overlap(d);o['file']=s['file'];details.append(o)
            fig,ax=plt.subplots(figsize=(12,4))
            colors=plt.get_cmap('tab10')
            for e in d['events']:
                left=e['GPU_start_s'];width=e['GPU_end_s']-left
                ax.broken_barh([(left,width)],(e['rank']-.35,.7),facecolors=colors(int(e['id'].split('-')[-1])%10))
            ax.set(yticks=range(4),yticklabels=['GPU4','GPU5','GPU6','GPU7'],xlabel='Seconds from request admission',
                title=f"{d['placement']} R{d['depth']} C{d['C']} — CUDA stream brackets, request colors")
            fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True)
            fig.savefig(ROOT/'figures'/f"{f.parent.name}_timeline.png",dpi=160);plt.close(fig)
    write(ROOT/'results_raw.json',results);write(ROOT/'overlap_event_evidence.json',details)
    with (ROOT/'results.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in results for k in r));w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader()
        for r in results:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    comparisons=[]
    grouped=collections.defaultdict(dict)
    for s in results:
        if not s['file'].startswith('raw/primary_'):continue
        key=(s['depth'],s['C'],s['P'],s['O'])
        grouped[key].setdefault(s['placement'],[]).append(s)
    for key,arms in sorted(grouped.items()):
        if set(arms)!={'layer_pp4','loop_pp4'}:continue
        a,b=arms['layer_pp4'],arms['loop_pp4']
        if len(a)!=len(b):continue
        comparisons.append(dict(depth=key[0],C=key[1],P=key[2],O=key[3],N_repeats=len(a),
            layer_median_s=float(np.median([s['wall_s'] for s in a])),
            loop_median_s=float(np.median([s['wall_s'] for s in b])),
            **paired_ratio([s['wall_s'] for s in a],[s['wall_s'] for s in b])))
    write(ROOT/'paired_comparison.json',comparisons)
    for P,O in [(128,32),(128,128),(512,32),(512,128)]:
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for arm in ['layer_pp4','loop_pp4']:
            rows=[s for s in results if s['file'].startswith('raw/primary_') and s['placement']==arm and s['depth']==4 and s['P']==P and s['O']==O]
            cs=sorted({s['C'] for s in rows})
            if not cs:continue
            axs[0].plot(cs,[np.median([s['tokens_per_s'] for s in rows if s['C']==C]) for C in cs],marker='o',label=arm)
            axs[1].plot(cs,[np.median([s['TPOT_P50_ms'] for s in rows if s['C']==C]) for C in cs],marker='o',label=arm)
        axs[0].set(xlabel='Concurrent requests',ylabel='Output tokens/s')
        axs[1].set(xlabel='Concurrent requests',ylabel='Request TPOT P50 (ms)')
        for ax in axs:ax.legend()
        fig.suptitle(f'LIVE_MEASURED fixed R4, input {P}, output {O}');fig.tight_layout()
        (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures'/f'scaling_P{P}_O{O}.png',dpi=160);plt.close(fig)
    print('ANALYZED',len(results),'physical runs',len(comparisons),'paired comparisons')


if __name__=='__main__':main()
