"""Summarize the frozen continuous baselines with start-level intervals."""
import collections
import json
import numpy as np
from .common import ROOT,write
from .figures import interval
from .confirmation_analysis import paired_ratio


def main():
    groups=collections.defaultdict(dict)
    for p in sorted((ROOT/'analysis').glob('Main_*_ShortAnswer200_*_BI0_v3_metrics.json')):
        row=json.loads(p.read_text());cfg=json.loads((ROOT/'raw'/row['name']/'configuration.json').read_text())
        quality=json.loads((ROOT/'analysis'/f'{row["name"]}_quality_gate.json').read_text())
        groups[(row['method'],row['offered_QPS'])][cfg['arrival_seed']]=dict(metrics=row,cfg=cfg,quality=quality)
    fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','mean_TBT_ms','TPOT_p95_ms',
        'generated_output_tokens_per_second','SLO_qualified_output_tokens_per_second',
        'Joint_SLO_attainment','completed_requests_per_second','output_tokens','actual_arrival_QPS')
    cells=[];comparisons=[]
    for (method,rate),starts in sorted(groups.items()):
        cells.append(dict(method=method,QPS=rate,n=len(starts),seeds=sorted(starts),
            means={f:float(np.mean([r['metrics'][f] for r in starts.values()])) for f in fields},
            CI95={f:interval([r['metrics'][f] for r in starts.values()]) for f in fields},
            full200_quality_all_starts=all(r['quality']['CI_passed'] for r in starts.values()),
            names=[r['metrics']['name'] for r in starts.values()]))
    for rate in sorted({key[1] for key in groups}):
        a=groups.get(('Agg',rate),{});b=groups.get(('Native',rate),{});seeds=sorted(set(a)&set(b))
        if not seeds:continue
        assert all(a[s]['cfg']['arrival_offsets']==b[s]['cfg']['arrival_offsets'] and
            a[s]['cfg']['dataset_SHA256']==b[s]['cfg']['dataset_SHA256'] for s in seeds)
        comparisons.append(dict(QPS=rate,n=len(seeds),seeds=seeds,
            Agg_over_Native={f:paired_ratio([a[s]['metrics'][f] for s in seeds],[b[s]['metrics'][f] for s in seeds]) for f in fields[:8]}))
    write(ROOT/'analysis/BI0_continuous_baseline_statistics_v3.json',dict(cells=cells,comparisons=comparisons,
        evidence='MEASURED_ONLINE',
        scope='Common B8192/S1024/KV16, C3/async default, official UCX initialization, Natural EOS. '
        'Finite arrival-to-drain cohorts, not sustainable capacity. CI resamples paired independent starts, not requests. '
        'This is a LongBench benchmark using DuetServe-style methodology, not numerical reproduction of its fixed8000/200 workload.'))


if __name__=='__main__':main()
