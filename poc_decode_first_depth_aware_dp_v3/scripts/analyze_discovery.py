"""Request-level statistics from actual online exits. No scheduling speedup inference."""
import collections
import csv
import statistics
import numpy as np
from poc_decode_first_depth_aware_dp_v3.contract import *

def interval(values):
    a=np.asarray(values,float)
    rng=np.random.default_rng(20261018)
    indices=rng.integers(0,len(a),size=(20000,len(a)))
    return np.quantile(a[indices].mean(1),[.025,.975]).tolist()

def main():
    source=ROOT/'outputs/discovery128_graph_b1_v1'
    assert read(source/'status.json')['success']
    frozen=read(ROOT/'artifacts/discovery_frozen.json')['rows']
    byid={r['id']:r for r in frozen}
    rows=[]
    traces=[]
    for path in source.glob('*.json'):
        raw=read(path)
        if not isinstance(raw,dict) or 'mean_depth' not in raw:continue
        row=dict(id=raw['id'],dataset=raw['dataset'],split=byid[raw['id']]['split'],
            prompt_tokens=raw['prompt_tokens'],output_tokens=raw['output_length'],
            mean_depth=raw['mean_depth'],within_request_depth_std=statistics.pstdev(e['depth'] for e in raw['depth_events']),
            total_loop_work=sum(e['depth'] for e in raw['depth_events']),
            TTFT_ms=raw['TTFT_ms'],E2E_ms=raw['E2E_ms'],decode_ms=raw['decode_ms'],
            decode_ms_per_token=raw['decode_ms']/max(1,raw['output_length']-1),
            terminated_EOS=raw['finish_reason']=='eos',cap=raw['cap'],
            label='MEASURED_ONLINE_TRACE_LANE')
        rows.append(row)
        traces.extend(dict(id=raw['id'],dataset=raw['dataset'],**e) for e in raw['depth_events'])
    assert set(byid)=={r['id'] for r in rows}
    rows=sorted(rows,key=lambda r:r['id'])
    out=ROOT/'artifacts'
    write(out/'request_depth_summary.json',rows)
    with (out/'request_depth_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import gzip
    with gzip.open(out/'decode_depth_traces.jsonl.gz','wt') as f:
        for r in traces:f.write(json.dumps(r)+'\n')
    summaries=[]
    for domain in sorted({r['dataset'] for r in rows}):
        values=[r for r in rows if r['dataset']==domain]
        depths=[r['mean_depth'] for r in values]
        summaries.append(dict(dataset=domain,n=len(values),mean_depth=statistics.mean(depths),
            CI95_mean_depth=interval(depths),between_request_std=statistics.pstdev(depths),
            depth_range=[min(depths),max(depths)],EOS=sum(r['terminated_EOS'] for r in values),
            cap_truncated=sum(not r['terminated_EOS'] for r in values),
            mean_output_tokens=statistics.mean(r['output_tokens'] for r in values)))
    # This regression is retrospective cost diagnosis, not an admission predictor.
    domains=sorted({r['dataset'] for r in rows})
    base=np.asarray([[1,r['prompt_tokens']/1000,(r['output_tokens']-1)/256]+
        [int(r['dataset']==d) for d in domains[1:]] for r in rows])
    depth=np.asarray([r['total_loop_work']/768 for r in rows])[:,None]
    augmented=np.concatenate([base,depth],axis=1)
    y=np.asarray([r['decode_ms'] for r in rows])
    train=np.asarray([r['split']=='calibration' for r in rows])
    test=~train
    models=[]
    for name,x in [('length_prompt_domain',base),('plus_actual_depth_work',augmented)]:
        coef=np.linalg.lstsq(x[train],y[train],rcond=None)[0]
        pred=x[test]@coef
        errors=y[test]-pred
        models.append(dict(name=name,heldout_n=int(test.sum()),MAE_ms=float(np.mean(abs(errors))),
            R2=float(1-np.sum(errors**2)/np.sum((y[test]-np.mean(y[test]))**2))))
    histogram=collections.Counter(e['depth'] for e in traces)
    write(out/'depth_heterogeneity.json',dict(domains=summaries,histogram=dict(histogram),
        requests=len(rows),decode_tokens=len(traces),retrospective_cost_regressions=models,
        request_mean_depth_range=[min(r['mean_depth'] for r in rows),max(r['mean_depth'] for r in rows)],
        repeated_loop_work=sum(r['total_loop_work'] for r in rows),
        notes=['All timing is trace-lane B1. It is not live DP4 scheduling evidence.',
               'The common cap is 256 with natural EOS. Truncated responses do not establish task accuracy.',
               'Actual depth is a retrospective diagnostic feature, not a causal admission feature.']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for domain in domains:
        values=[r for r in rows if r['dataset']==domain]
        ax[0].scatter([r['output_tokens'] for r in values],[r['mean_depth'] for r in values],label=domain,s=15)
        ax[1].scatter([r['mean_depth'] for r in values],[r['decode_ms_per_token'] for r in values],s=15)
    ax[0].set(xlabel='Actual output tokens (natural EOS, cap 256)',ylabel='Actual mean decode depth')
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Actual mean decode depth',ylabel='Trace-lane decode ms/token')
    fig.tight_layout();fig.savefig(figures/'depth_heterogeneity.png',dpi=150);plt.close(fig)
    print('DISCOVERY',len(rows),dict(histogram),models)

if __name__=='__main__':main()
