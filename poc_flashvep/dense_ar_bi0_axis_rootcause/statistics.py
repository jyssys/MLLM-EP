"""Use independent server starts, not request count, for performance intervals."""
import itertools
import json
import numpy as np
from .common import ROOT,write,digest


def interval(a,b):
    assert len(a)==len(b) and len(a)>=3
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    draws=[]
    for indices in itertools.product(range(len(a)),repeat=len(a)):
        den=b[list(indices)].mean()
        draws.append(float(a[list(indices)].mean()/den) if den>0 else None)
    valid=[v for v in draws if v is not None]
    return dict(ratio=float(a.mean()/b.mean()) if b.mean()>0 else None,
        CI95=[float(np.percentile(valid,p)) for p in (2.5,97.5)] if len(valid)==len(draws) else None,
        starts=len(a),resamples=len(draws),
        method='Exact enumeration of paired start-level empirical bootstrap. Ratio of start-metric means. Only3 independent starts give limited tail precision.')


def identity(a,b):
    ca=json.loads((ROOT/'raw'/a/'configuration.json').read_text())
    cb=json.loads((ROOT/'raw'/b/'configuration.json').read_text())
    keys=['dataset_SHA256','arrival_offsets','arrival_seed','token_budget','max_num_seqs','batch_invariant',
        'source_credit_limit','execution_family','KV_block_tokens','async_scheduling','output_cap','controlled_pipeline']
    checks={key:ca.get(key)==cb.get(key) for key in keys}
    assert all(checks.values()),(a,b,checks)
    ha=json.loads((ROOT/'raw'/a/'source_hashes.json').read_text());hb=json.loads((ROOT/'raw'/b/'source_hashes.json').read_text())
    core='poc_flashvep/dense_ar_bi0_axis_rootcause/serving.py'
    assert ha[core]==hb[core],(a,b,'coordinator mismatch')
    return dict(checks=checks,coordinator_SHA256=ha[core],scope='Same C0 coordinator core. Full source archives can differ in unrelated analysis or controller files.')


def absolute_means(arms,seeds):
    fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms',
        'SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second',
        'completed_requests_per_second','Joint_SLO_attainment','output_tokens','actual_arrival_QPS')
    output={}
    for axis,cells in arms.items():
        output[axis]={field:float(np.mean([cells[s]['metrics'][field] for s in seeds])) for field in fields}
        output[axis]['per_start']=[dict(seed=s,name=cells[s]['name'],
            metrics={field:cells[s]['metrics'][field] for field in fields}) for s in seeds]
    return output


def main():
    rows=json.loads((ROOT/'analysis/last12_confirmation_v1.json').read_text())['results']
    repair=ROOT/'analysis/last12_native_reference_repair.json'
    if repair.exists():
        record=json.loads(repair.read_text());rows=[r for r in rows if r['name']!=record['original']]+[record['replacement']]
    extra=ROOT/'analysis/full200_allcuts_confirmation_v1.json'
    if extra.exists():rows.extend(json.loads(extra.read_text())['results'])
    rows=[r for r in rows if r['status']=='MEASURED_ONLINE'];comparisons=[]
    seeds=(2026112071,2026112072,2026112073)
    native={r['seed']:r for r in rows if r['axis']=='Native'}
    for cut in (2,4,8,12):
        arms={axis:{r['seed']:r for r in rows if r['axis']==axis and r['cut']==cut} for axis in ('Layer','Token')}
        if not all(set(seeds)<=set(v) for v in arms.values()):continue
        provenance=[identity(arms['Layer'][s]['name'],arms['Token'][s]['name']) for s in seeds]
        metrics={field:interval([arms['Layer'][s]['metrics'][field] for s in seeds],
                               [arms['Token'][s]['metrics'][field] for s in seeds])
                 for field in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second')}
        protection={}
        for axis in ('Layer','Token'):
            bounds={field:interval([arms[axis][s]['metrics'][field] for s in seeds],
                                  [native[s]['metrics'][field] for s in seeds])
                    for field in ('ITL_p99_ms','TPOT_p95_ms')}
            protection[axis]=dict(metrics=bounds,passed=all(v['CI95'][1]<=1.05 for v in bounds.values()))
        qualified=all(arms[axis][s]['quality']['CI_passed'] for axis in arms for s in seeds)
        winner='INCONCLUSIVE'
        if metrics['TTFT_p95_ms']['CI95'][1]<1:winner='Layer'
        elif metrics['TTFT_p95_ms']['CI95'][0]>1:winner='Token'
        comparisons.append(dict(cut=cut,seeds=list(seeds),N=200,QPS=2.35,metrics=metrics,
            decode_protection=protection,quality_CI_passed=qualified,TTFT_winner=winner,
            absolute_start_means=absolute_means(dict(arms,Native=native),seeds),
            provenance=provenance,names={axis:[arms[axis][s]['name'] for s in seeds] for axis in arms}))
    result=dict(comparisons=comparisons,evidence='MEASURED_ONLINE',
        SLO=dict(TTFT_ms=5000,every_ITL_ms=34,TPOT_ms=33),
        scope='Natural EOS cap2048, C0 actual1/1/1, c4,h4,KV128. Finite cohorts, not sustainable capacity. '
        'Different BI0 query geometry can change natural outputs. Task-quality CI is separate from strict same-geometry correctness. '
        'A TTFT winner alone does not pass the Layer sweet-spot success gate.')
    write(ROOT/'analysis/full200_confirmation_statistics.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
