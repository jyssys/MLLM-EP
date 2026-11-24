"""Recompute per-start token metrics and joint SLO without changing thresholds."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

from .common import ROOT,write,digest

SLO=dict(TTFT_ms=5000,ITL_ms=34,TPOT_ms=33)


def percentile(x,q):return float(np.percentile(x,q)) if len(x) else None


def gates(r):
    valid=r.get('status')=='MEASURED_ONLINE'
    if not valid:return dict(TTFT_PASS=False,ITL_PASS=False,TPOT_PASS=False,JOINT_PASS=False,category='FAILED_EXECUTION')
    t=r['TTFT_ms']<=SLO['TTFT_ms']
    i=all(value<=SLO['ITL_ms'] for value in r['ITL_ms'])
    p=r['TPOT_ms']<=SLO['TPOT_ms']
    failed=[name for name,passed in [('TTFT',t),('ITL',i),('TPOT',p)] if not passed]
    category='JOINT_PASS' if not failed else '+'.join(failed)+'_FAIL'
    return dict(TTFT_PASS=t,ITL_PASS=i,TPOT_PASS=p,JOINT_PASS=t and i and p,category=category)


def summarize(run):
    file=ROOT/'raw'/(run+'_requests.jsonl');completion=ROOT/'state'/(run+'_complete.json')
    rows=[json.loads(line) for line in file.read_text().splitlines()]
    done=json.loads(completion.read_text());duration=done['duration_seconds']
    valid=[r for r in rows if r['status']=='MEASURED_ONLINE'];planned=done.get('planned_count',done['actual_count'])
    categories=Counter();category_tokens=Counter();annotated=[]
    for r in rows:
        g=gates(r);categories[g['category']]+=1
        category_tokens[g['category']]+=r.get('output_tokens',len(r.get('token_ids',[])))
        annotated.append(dict(**r,**g))
    itls=[x for r in valid for x in r['ITL_ms']];t=[r['TTFT_ms'] for r in valid]
    p=[r['TPOT_ms'] for r in valid];tokens=sum(r['output_tokens'] for r in valid)
    value=dict(run=run,raw_SHA256=digest(file),planned_count=planned,actual_count=len(rows),
        completed_count=len(valid),failed_count=len(rows)-len(valid),not_run_count=planned-len(rows),
        primary_execution_valid=len(valid)==planned and done['passed'],serial=done['serial'],
        duration_seconds=duration,TTFT_mean_ms=float(np.mean(t)) if t else None,
        TTFT_p50_ms=percentile(t,50),TTFT_p95_ms=percentile(t,95),TTFT_p99_ms=percentile(t,99),
        ITL_mean_ms=float(np.mean(itls)) if itls else None,ITL_p95_ms=percentile(itls,95),
        ITL_p99_ms=percentile(itls,99),ITL_max_ms=max(itls) if itls else None,
        TPOT_p95_ms=percentile(p,95),total_output_tokens=tokens,
        raw_output_tokens_per_second=tokens/duration,completed_requests_per_second=len(valid)/duration,
        total_input_output_tokens_per_second=(sum(r['usage']['prompt_tokens'] for r in valid)+tokens)/duration,
        SLO_goodput=sum(r['output_tokens'] for r in annotated if r['JOINT_PASS'])/duration,
        categories=categories,category_output_tokens=category_tokens,SLO=SLO,
        gate_attainment={field:sum(r[field] for r in annotated)/planned for field in
            ('TTFT_PASS','ITL_PASS','TPOT_PASS','JOINT_PASS')},
        ITL_observation_count=len(itls),latency_request_denominator=len(valid),attainment_denominator=planned,
        metric_contract='Completion-token IDs received through HTTP SSE,including EOS. Co-delivered tokens have zero ITL. '
            'TPOT=(last-first)/(N-1),single-output TPOT=0 and empty ITL passes. Throughput uses planned first arrival to drain. '
            'No stage token duplication. Failed or missing requests cannot become Primary performance. '
            'Finite-cohort throughput is not sustainable capacity. This file alone does not establish quality or decode safety.')
    write(ROOT/'analysis'/(run+'_metrics.json'),value)
    return value


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();print(json.dumps(summarize(args.run),indent=2))


if __name__=='__main__':main()
