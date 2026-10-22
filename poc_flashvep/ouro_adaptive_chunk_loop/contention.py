"""Matched same-GPU interference analysis. Not scheduler speedup."""
import csv
import json
import numpy as np
from .common import ROOT, write

def main():
    rows=[]
    for folder in sorted((ROOT/'raw').glob('p0_*')):
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
    complete=sum((f/'completed.json').exists() for f in (ROOT/'raw').glob('p0_*'))
    value=dict(completed_jobs=complete,registered_jobs=expected,observations=len(rows),
        triggered_pairs=sum(r['trigger'] for r in comparisons),
        status='COMPLETE_REQUIRES_CONCURRENCY_REVIEW' if complete==expected else 'PARTIAL',
        limitation='Only two independent starts per GPU/concurrency. Interference trigger is not a neutrality theorem. Second-context phases are not separately barrier-aligned.')
    write(ROOT/'contention_summary.json',value)
    print(json.dumps(value,indent=2))

if __name__=='__main__':
    main()
