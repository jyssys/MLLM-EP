"""Keep observer controls, profiler service time, and client time separate."""
import csv
import gzip
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write,digest
from .surface_analysis import csv_out

def main():
    pair_rows=[];kernel_rows=[];trace_index=[]
    for gpu in range(8):
        version=2 if gpu in (0,1) else 1
        folder=ROOT/'raw'/f'p2_observer_control_gpu{gpu}_v{version}'
        assert (folder/'completed.json').exists(),folder
        files={p.name:json.loads(p.read_text()) for p in folder.glob('*_clean_r*.json')}
        for name,clean in files.items():
            observed=json.loads((folder/name.replace('_clean_','_events_')).read_text())
            ci=[x*1000 for r in clean['requests'] for x in r['ITLs_s']]
            oi=[x*1000 for r in observed['requests'] for x in r['ITLs_s']]
            same=all(a['output_ids']==b['output_ids'] for a,b in zip(clean['requests'],observed['requests']))
            a=clean['end']-clean['start'];b=observed['end']-observed['start']
            pair_rows.append(dict(run_id=clean['run_id'],GPU=gpu,policy=clean['policy'],C=clean['configured_C'],P=clean['P'],N=clean['N'],trial=clean['trial'],
                clean_elapsed_s=a,event_elapsed_s=b,event_overhead_percent=100*(b/a-1),
                clean_pooled_ITL_p95_ms=float(np.quantile(ci,.95)),event_pooled_ITL_p95_ms=float(np.quantile(oi,.95)),
                clean_pooled_ITL_p99_ms=float(np.quantile(ci,.99)),event_pooled_ITL_p99_ms=float(np.quantile(oi,.99)),
                token_exact=same,requests=len(clean['requests']),
                exact_request_sequences=sum(a['output_ids']==b['output_ids'] for a,b in zip(clean['requests'],observed['requests'])),
                equal_request_lengths=sum(len(a['output_ids'])==len(b['output_ids']) for a,b in zip(clean['requests'],observed['requests'])),
                clean_output_tokens=sum(len(r['output_ids']) for r in clean['requests']),event_output_tokens=sum(len(r['output_ids']) for r in observed['requests']),
                evidence='LIVE_MEASURED_MATCHED_OBSERVER_CONTROL'))
        for trace in folder.glob('*_heavy_trace.json'):
            data=json.loads(trace.read_text())
            kernels=[e for e in data['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='kernel']
            by_name=defaultdict(list)
            for event in kernels:
                by_name[event['name']].append(event['dur'])
            for name,values in by_name.items():
                kernel_rows.append(dict(trace=str(trace.relative_to(ROOT)),GPU=gpu,kernel=name,calls=len(values),
                    summed_service_ms=sum(values)/1000,scope='Summed kernel service, not non-overlapped E2E share'))
            trace_index.append(dict(path=str(trace.relative_to(ROOT)),sha256=digest(trace),kernels=len(kernels),
                scope='Heavy profiler only. CUDA graph replay kernels included where profiler reports them.'))
            if 'grouped_C512' in trace.name and kernels:
                first=min(e['ts'] for e in kernels)
                streams=sorted({e['tid'] for e in kernels},key=str)
                fig,ax=plt.subplots(figsize=(13,3.5),constrained_layout=True)
                for i,stream in enumerate(streams):
                    spans=[((e['ts']-first)/1000,e['dur']/1000) for e in kernels if e['tid']==stream]
                    ax.broken_barh(spans,(i-.35,.7))
                ax.set(yticks=range(len(streams)),yticklabels=[str(s) for s in streams],xlabel='Time from first observed kernel (ms)',
                    ylabel='CUDA stream ID',title=f'Actual heavy-profile kernel timeline: GPU{gpu}, Grouped C512, eight-output cap')
                fig.savefig(ROOT/'figures'/f'kernel_timeline_gpu{gpu}.png',dpi=160);plt.close(fig)
    csv_out(ROOT/'observer_control_pairs.csv',pair_rows)
    csv_out(ROOT/'kernel_service_summary.csv',kernel_rows)
    write(ROOT/'heavy_profile_index.json',trace_index)
    summaries=[]
    rng=np.random.default_rng(20261023)
    for p,n in ((2033,4),(8177,8)):
        for policy in ('atomic','staged4','grouped','grouped3','split','mixed'):
            rows=[r for r in pair_rows if r['P']==p and r['policy']==policy]
            groups=defaultdict(list)
            for row in rows:
                groups[row['run_id']].append(row['event_overhead_percent'])
            means=np.array([np.mean(v) for v in groups.values()])
            draws=means[rng.integers(0,len(means),size=(5000,len(means)))].mean(axis=1)
            summaries.append(dict(P=p,N=n,policy=policy,independent_starts=len(groups),paired_cohorts=len(rows),
                mean_event_overhead_percent=float(means.mean()),CI95_low=float(np.quantile(draws,.025)),CI95_high=float(np.quantile(draws,.975)),
                token_exact_cohorts=sum(r['token_exact'] for r in rows),total_cohorts=len(rows)))
    csv_out(ROOT/'observer_control_summary.csv',summaries)
    write(ROOT/'observer_controls_summary.json',dict(paired_cohorts=len(pair_rows),token_exact_cohorts=sum(r['token_exact'] for r in pair_rows),
        paired_requests=sum(r['requests'] for r in pair_rows),exact_request_sequences=sum(r['exact_request_sequences'] for r in pair_rows),
        equal_request_lengths=sum(r['equal_request_lengths'] for r in pair_rows),
        heavy_traces=len(trace_index),evidence='LIVE_MEASURED_CONTROLS',
        limitation='The retry uses two active measurement GPUs with owned burners on six others. Other controls use the original eight-worker queue. Match within-start pairs, not cross-start raw latency.'))
    print(json.dumps(json.loads((ROOT/'observer_controls_summary.json').read_text()),indent=2))

if __name__=='__main__':
    main()
