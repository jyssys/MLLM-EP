"""Diagnostic cost characterization. No sum of overlaps becomes E2E share."""
import csv
import gzip
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write,digest
from .predictor import phase

def csv_out(path,rows):
    if not rows:
        return
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summarize(values):
    return dict(N=len(values),mean=float(np.mean(values)),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),p99=float(np.quantile(values,.99)))

def main():
    plan=json.loads((ROOT/'p2_surface_plan.json').read_text())
    cohorts=[];quanta=[];exposure=[];indices=[];assignments=[]
    itl_bins=defaultdict(list)
    for job in plan['jobs']:
        folder=ROOT/'raw'/job['id']
        assert (folder/'completed.json').exists()
        identity=json.loads((folder/'identity.json').read_text())
        assignments.append(dict(run_id=job['id'],preferred_GPU=job['gpu'],actual_GPU=identity['physical_GPU'],UUID=identity['UUID'],split=job['split']))
        for file in sorted(folder.glob('*_T*.json')):
            data=json.loads(file.read_text());qs=data['quantum_observations']
            itls=[1000*x for r in data['requests'] for x in r['ITLs_s']]
            ttfts=[1000*r['TTFT_s'] for r in data['requests'] if r['TTFT_s'] is not None]
            long=data['requests'][-1]
            tokens=sum(len(r['output_ids']) for r in data['requests'])
            elapsed=data['end']-data['start']
            row=dict(run_id=job['id'],GPU=identity['physical_GPU'],P=data['P'],N=data['N'],C=data['configured_C'],policy=data['policy'],
                split=data['split'],trial=data['trial'],requests=len(data['requests']),output_tokens=tokens,
                elapsed_s=elapsed,drained_output_TPS=tokens/elapsed,TTFT_mean_ms=float(np.mean(ttfts)),
                long_TTFT_ms=1000*long['TTFT_s'],ITL_p50_ms=float(np.quantile(itls,.50)),
                ITL_p95_ms=float(np.quantile(itls,.95)),ITL_p99_ms=float(np.quantile(itls,.99)),
                gaps=len(itls),gaps_gt100ms=sum(x>100 for x in itls),gaps_gt200ms=sum(x>200 for x in itls),
                capped=sum(r['finish_reason']=='length' for r in data['requests']),quanta=len(qs),
                evidence='DIAGNOSTIC_EVENT_LANE_NOT_CLEAN_CAPACITY')
            cohorts.append(row)
            itl_bins[(row['P'],row['N'],row['C'],row['policy'])].extend(itls)
            indices.append(dict(run=job['id'],path=str(file.relative_to(ROOT)),sha256=digest(file),GPU=identity['physical_GPU'],
                schema=qs[0].get('observer_schema_version',1),kind='Aligned CUDA event timeline, not kernel service'))
            for q in qs:
                prefill_work=sum(s['tokens']*s['group'] for s in q['spans'] if s['phase'] in ('prefill','prefilling_recurrent'))
                quanta.append(dict(run_id=job['id'],GPU=identity['physical_GPU'],P=row['P'],N=row['N'],C=row['C'],
                    policy=row['policy'],split=row['split'],trial=row['trial'],index=q['quantum'],phase=phase(q),
                    G=q['actual_G'],rows=q['actual_total_rows'],decode_rows=q['actual_decode_rows'],context_max=q['context_max'],
                    depth_start=q['recurrent_depth_start'],prefill_token_loop_work=prefill_work,
                    scheduling_impact_ms=q['scheduling_impact_ms'],device_stream_interval_ms=q['device_stream_interval_ms'],
                    enqueue_to_ready_ms=q['enqueue_to_ready_ms'],host_submit_ms=1000*(q['host_submit_end_s']-q['host_begin_s']),
                    staged_bytes=q['staged_bytes'],graph_entries=q['graph_entries'],
                    graph_replay_delta=q['graph_counts_end'].get('replays',0)-q['graph_counts'].get('replays',0),
                    graph_capture_delta=q['graph_counts_end'].get('captures',0)-q['graph_counts'].get('captures',0),
                    schema=q.get('observer_schema_version',1)))
            pref=[q for q in qs if any(s['phase'] in ('prefill','prefilling_recurrent') for s in q['spans'])]
            for r in data['requests']:
                for gap,(a,b) in enumerate(zip(r['token_times'],r['token_times'][1:])):
                    interruptions=[q for q in pref if a<=q['host_begin_s']<b]
                    exposure.append(dict(run_id=job['id'],policy=row['policy'],C=row['C'],P=row['P'],N=row['N'],trial=row['trial'],
                        request_id=r['id'],gap=gap,ITL_ms=1000*(b-a),prefill_submissions=len(interruptions),
                        prefill_token_loop_work=sum(s['tokens']*s['group'] for q in interruptions for s in q['spans'] if s['phase'] in ('prefill','prefilling_recurrent')),
                        evidence='ACTUAL_TOKEN_GAP_EXPOSURE_NOT_CAUSAL_ATTRIBUTION'))
    csv_out(ROOT/'results_raw.csv',cohorts)
    csv_out(ROOT/'quantum_observations.csv.gz',quanta)
    csv_out(ROOT/'full_token_exposure.csv.gz',exposure)
    csv_out(ROOT/'gpu_assignment_history.csv',assignments)
    write(ROOT/'gpu_trace_index.json',indices)
    summary=[]
    for key,values in itl_bins.items():
        p,n,c,policy=key
        cell=[r for r in cohorts if (r['P'],r['N'],r['C'],r['policy'])==key]
        qq=[r for r in quanta if (r['P'],r['N'],r['C'],r['policy'])==key and r['prefill_token_loop_work']]
        summary.append(dict(P=p,N=n,C=c,policy=policy,cohorts=len(cell),independent_starts=len({r['run_id'] for r in cell}),
            pooled_ITL_p50_ms=float(np.quantile(values,.5)),pooled_ITL_p95_ms=float(np.quantile(values,.95)),pooled_ITL_p99_ms=float(np.quantile(values,.99)),
            mean_long_TTFT_ms=float(np.mean([r['long_TTFT_ms'] for r in cell])),mean_drained_TPS=float(np.mean([r['drained_output_TPS'] for r in cell])),
            median_prefill_quantum_ms=float(np.median([r['scheduling_impact_ms'] for r in qq])),
            median_token_loop_work_per_ms=float(np.median([r['prefill_token_loop_work']/max(.001,r['scheduling_impact_ms']) for r in qq])),
            mean_quanta=float(np.mean([r['quanta'] for r in cell])),max_staged_bytes=max(r['staged_bytes'] for r in quanta if (r['P'],r['N'],r['C'],r['policy'])==key)))
    csv_out(ROOT/'cost_surface_summary.csv',summary)
    pairs=[];rng=np.random.default_rng(20261023)
    for p,n,c in sorted({(r['P'],r['N'],r['C']) for r in cohorts}):
        for reference,candidate in [('atomic','staged4'),('staged4','grouped'),('atomic','grouped'),('grouped','mixed')]:
            cells={mode:{(r['run_id'],r['trial']):r for r in cohorts if r['P']==p and r['N']==n and r['C']==c and r['policy']==mode} for mode in (reference,candidate)}
            for metric in ('long_TTFT_ms','ITL_p95_ms','ITL_p99_ms','drained_output_TPS'):
                by_run=defaultdict(list)
                for key,a in cells[reference].items():
                    b=cells[candidate][key]
                    by_run[key[0]].append(100*(b[metric]-a[metric])/a[metric])
                means=np.array([np.mean(v) for v in by_run.values()])
                draws=means[rng.integers(0,len(means),size=(5000,len(means)))].mean(axis=1)
                pairs.append(dict(P=p,N=n,C=c,reference=reference,candidate=candidate,metric=metric,
                    independent_start_pairs=len(means),cohort_pairs=sum(map(len,by_run.values())),
                    candidate_percent_change=float(means.mean()),CI95_low=float(np.quantile(draws,.025)),CI95_high=float(np.quantile(draws,.975)),
                    scope='Paired mean cohort metric changes. Cohort p99 is not pooled token p99. Diagnostic lane only.'))
    csv_out(ROOT/'p2_comparison_pairs.csv',pairs)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    policies=('atomic','split','grouped','grouped3','staged4','mixed')
    chunks=(64,128,256,512,768,1024,1536,2048)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,metric,label in zip(axes,('median_prefill_quantum_ms','mean_long_TTFT_ms','pooled_ITL_p99_ms'),('Median prefill quantum (ms)','Mean long-request TTFT (ms)','Pooled observed ITL p99 (ms)')):
        array=np.array([[next(r[metric] for r in summary if r['P']==8177 and r['N']==8 and r['C']==c and r['policy']==mode) for c in chunks] for mode in policies])
        im=ax.imshow(array,aspect='auto');fig.colorbar(im,ax=ax,label='ms')
        ax.set(xticks=range(8),xticklabels=chunks,yticks=range(6),yticklabels=policies,xlabel='Configured C (tokens)',title=label)
    fig.suptitle('Measured diagnostic event lane: P8177 / N8, 12 cohorts per cell, 32-output cap')
    fig.savefig(out/'cost_heatmaps.png',dpi=170);fig.savefig(out/'cost_heatmaps.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    for mode in policies:
        selected=sorted([r for r in summary if r['P']==8177 and r['N']==8 and r['policy']==mode],key=lambda r:r['C'])
        ax.plot([r['mean_long_TTFT_ms'] for r in selected],[r['pooled_ITL_p99_ms'] for r in selected],marker='o',label=mode)
        for r in selected:
            ax.annotate(str(r['C']),(r['mean_long_TTFT_ms'],r['pooled_ITL_p99_ms']),fontsize=6)
    ax.set(xlabel='Mean long-request TTFT (ms)',ylabel='Pooled observed ITL p99 (ms)',title='Diagnostic C/G operating points, P8177 / N8')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'diagnostic_pareto.png',dpi=170);plt.close(fig)
    write(ROOT/'surface_analysis_summary.json',dict(cohorts=len(cohorts),quantum_observations=len(quanta),
        token_gap_observations=len(exposure),full_token_gaps_with_multiple_prefills=sum(r['prefill_submissions']>=2 for r in exposure),
        schema_counts={str(s):sum(r['schema']==s for r in quanta) for s in set(r['schema'] for r in quanta)},
        scope='Measured event lane. No clean serving-capacity or online adaptive claim.'))
    print(json.dumps(json.loads((ROOT/'surface_analysis_summary.json').read_text()),indent=2))

if __name__=='__main__':
    main()
