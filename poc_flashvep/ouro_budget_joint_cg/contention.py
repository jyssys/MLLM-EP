"""Matched same-GPU interference analysis. Not scheduler speedup."""
import csv
import json
import numpy as np
from .common import ROOT, write

def main():
    rows=[]
    for folder in sorted((ROOT/'raw').glob('cb_p0_*')):
        if not (folder/'completed.json').exists():
            continue
        identity=json.loads((folder/'identity.json').read_text())
        job=identity['job']
        for file in folder.glob('cal_P*_r*.json'):
            d=json.loads(file.read_text())
            rows.append(dict(run_id=job['id'],GPU=identity['physical_GPU'],startup=job['startup'],
                parallel_workers=job['parallel_workers'],P=d['P_bin'],N=d['N'],repetition=d['repetition'],
                median_ITL_ms=d['median_ITL_ms'],p95_ITL_ms=d['p95_ITL_ms'],p99_ITL_ms=d['p99_ITL_ms'],
                drained_output_tokens_s=d['drained_output_tokens_s'],evidence='LIVE_MEASURED_CALIBRATION'))
    comparisons=[]
    rng=np.random.default_rng(20261023)
    for gpu in range(8):
        for length,width in [(2048,4),(8192,8)]:
            for k in (2,4,8):
                a=[r for r in rows if (r['GPU'],r['P'],r['N'],r['parallel_workers'])==(gpu,length,width,1)]
                b=[r for r in rows if (r['GPU'],r['P'],r['N'],r['parallel_workers'])==(gpu,length,width,k)]
                if not a or not b:
                    continue
                by_a={(r['startup'],r['repetition']):r for r in a}
                pairs=[(by_a[(r['startup'],r['repetition'])],r) for r in b if (r['startup'],r['repetition']) in by_a]
                for metric in ['median_ITL_ms','p99_ITL_ms','drained_output_tokens_s']:
                    x=np.array([p[0][metric] for p in pairs]);y=np.array([p[1][metric] for p in pairs])
                    starts=sorted(set(p[0]['startup'] for p in pairs))
                    deltas=[]
                    for draw in range(5000):
                        indices=[]
                        for s in rng.choice(starts,len(starts),replace=True):
                            group=[i for i,p in enumerate(pairs) if p[0]['startup']==s]
                            indices.extend(rng.choice(group,len(group),replace=True))
                        deltas.append(100*(float(y[indices].mean()/x[indices].mean())-1))
                    comparisons.append(dict(GPU=gpu,P=length,N=width,parallel_workers=k,metric=metric,
                        paired_cohorts=len(pairs),independent_starts=len(starts),solo_mean=float(x.mean()),parallel_mean=float(y.mean()),
                        percent_change=100*(float(y.mean()/x.mean())-1),CI95_low=float(np.percentile(deltas,2.5)),CI95_high=float(np.percentile(deltas,97.5)),
                        trigger=bool(abs(y.mean()/x.mean()-1)>.03),evidence='LIVE_MEASURED_PAIRED_CONTROL_NOT_POLICY_GAIN'))
    for name,values in [('contention_calibration.csv',rows),('contention_paired_comparisons.csv',comparisons)]:
        if values:
            with (ROOT/name).open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]))
                writer.writeheader();writer.writerows(values)
    expected=64
    complete=sum((f/'completed.json').exists() for f in (ROOT/'raw').glob('cb_p0_*'))
    value=dict(completed_jobs=complete,registered_jobs=expected,observations=len(rows),
        triggered_pairs=sum(r['trigger'] for r in comparisons),
        status='COMPLETE_REQUIRES_CONCURRENCY_REVIEW' if complete==expected else 'PARTIAL',
        limitation='Only two independent starts per GPU/concurrency. Interference trigger is not a neutrality theorem. Second-context phases are not separately barrier-aligned.')
    write(ROOT/'contention_summary.json',value)
    if complete==expected:
        lines=['# GPU contention calibration','',
            'All 64 registered device/start jobs completed.',
            'Each GPU has two independent starts for each of the 1/2/4/8-worker conditions.',
            'Each start has three warmed cohorts at P≈2K/N4 and P≈8K/N8.',
            'These are bounded natural calibration runs on legacy inputs, not new held-out serving results.','',
            '| Prompt bin | N | Workers | Median-ITL drift range across GPUs, % | Drained throughput drift range, % |',
            '| --- | ---: | ---: | ---: | ---: |']
        for p,n in [(2048,4),(8192,8)]:
            for k in (2,4,8):
                med=[r['percent_change'] for r in comparisons if r['P']==p and r['parallel_workers']==k and r['metric']=='median_ITL_ms']
                rate=[r['percent_change'] for r in comparisons if r['P']==p and r['parallel_workers']==k and r['metric']=='drained_output_tokens_s']
                lines.append(f'| ≈{p} | {n} | {k} | [{min(med):.3f}, {max(med):.3f}] | [{min(rate):.3f}, {max(rate):.3f}] |')
        lines+=['','Positive ITL drift means slower token gaps. Negative throughput drift means lower drained output rate.',
            'The paired CSV preserves same-GPU/start comparisons and 95% bootstrap intervals.',
            'Only two process starts support each device/concurrency cell.',
            'The 3% threshold is a review trigger, not a proof of neutral interference.','',
            'The first context has a warmed group-start barrier.',
            'The second context can have start skew across workers.',
            'Telemetry retains power, clocks, load, memory, and disk counters.',
            'The present analysis does not establish the physical cause of drift.',
            'CPU control, driver contention, graph preparation, and power effects need separate attribution.','',
            'Numerical diagnostics can continue under their explicit non-performance scope.',
            'Clean serving comparisons require matched worker load and GPU rotation.',
            'No adaptive policy gain or sustainable capacity follows from this calibration.']
        (ROOT/'gpu_contention.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(value,indent=2))

if __name__=='__main__':
    main()
