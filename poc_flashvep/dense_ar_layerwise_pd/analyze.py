"""Use request delivery times and keep unmeasured execution times explicit."""
import json,math,statistics
from .common import *


def records(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def percentile(values,q):
    x=sorted(values)
    if not x:return None
    at=(len(x)-1)*q;i=int(at);j=math.ceil(at);return x[i]+(x[j]-x[i])*(at-i)


def metrics(name):
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):return dict(name=name,complete=False)
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());rows=records(folder/'requests.jsonl')
    assert len(rows)==cfg['count'] and len({r['sample_id'] for r in rows})==len(rows)
    itl=[1000*(b['time']-a['time'])/(b['count']-a['count']) for r in rows for a,b in zip(r['token_events'],r['token_events'][1:]) if b['count']>a['count']]
    span=max(r['finish'] for r in rows)-min(r['arrival'] for r in rows)
    slo=json.loads((ROOT/'configs/frozen_SLO_anchor.json').read_text())['qwen8_L8000']
    out=dict(name=name,complete=True,count=len(rows),distinct_prompt_ids=len({r['prompt_id'] for r in rows}),
        offered_QPS=cfg.get('arrival_rate'),completed_QPS=len(rows)/span,input_tokens_s=sum(r['input_tokens'] for r in rows)/span,
        output_tokens_s=sum(r['output_tokens'] for r in rows)/span,
        SLO_goodput=sum(r['ttft_ms']<=slo['TTFT_ms'] and r['tpot_ms']<=slo['TPOT_ms'] for r in rows)/span,
        TTFT_attainment=sum(r['ttft_ms']<=slo['TTFT_ms'] for r in rows)/len(rows),
        TPOT_attainment=sum(r['tpot_ms']<=slo['TPOT_ms'] for r in rows)/len(rows),
        maximum_ITL_ms=max(itl,default=None),ITL_threshold_exceedances=sum(t>slo['ITL_ms'] for t in itl),
        ITL_intervals=len(itl),coalesced_nonfirst_events=sum(b['count']-a['count']>1 for r in rows for a,b in zip(r['token_events'],r['token_events'][1:])),
        Service_TTFT='NOT_MEASURED',exact_execution_queue_latency='NOT_MEASURED',frozen_SLO=slo,
        scope='Per-request coordinator delivery metrics. Coalesced intervals use event-average latency. Finite-cohort drain rates are not sustainable capacity. Serial diagnostic timings are not clean online performance.')
    arrivals=[r['arrival'] for r in rows]
    submission_path=folder/'actual_submissions.jsonl'
    sends=[r['actual_send_start'] for r in records(submission_path if submission_path.exists() else folder/'offered_requests.jsonl') if r['actual_send_start'] is not None]
    out['realized_arrival_QPS']=(len(rows)-1)/(max(arrivals)-min(arrivals)) if max(arrivals)>min(arrivals) else None
    out['actual_submit_QPS']=(len(sends)-1)/(max(sends)-min(sends)) if max(sends)>min(sends) else None
    for label,key in (('TTFT','ttft_ms'),('TPOT','tpot_ms'),('E2E','e2e_ms')):
        for q in (.5,.95,.99):out[f'{label}_p{int(q*100)}_ms']=percentile([r[key] for r in rows],q)
    for q in (.5,.95,.99):out[f'ITL_p{int(q*100)}_ms']=percentile(itl,q)
    out['Host_arrival_to_worker_accept_p95_ms']=percentile([1000*(r['admission']-r['arrival']) for r in rows],.95)
    out['Decode_absolute_SLO_pass']=out['TPOT_p95_ms']<=slo['TPOT_ms'] and out['ITL_p99_ms']<=slo['ITL_ms']
    write(folder/'metrics.json',out);return out


def compare(reference,candidate,allow_reference_prefix=False):
    a={r['sample_id']:r for r in records(ROOT/'raw'/reference/'requests.jsonl')}
    b={r['sample_id']:r for r in records(ROOT/'raw'/candidate/'requests.jsonl')};failures=[]
    for i,y in b.items():
        x=a.get(i)
        if x is None or any(x[k]!=y[k] for k in ('input_ids','seed')):
            failures.append(dict(sample_id=i,reason='INCOMPARABLE_INPUT_OR_SEED'));continue
        expected=x['output_ids'][:len(y['output_ids'])] if allow_reference_prefix else x['output_ids']
        if expected!=y['output_ids']:failures.append(dict(sample_id=i,reason='OUTPUT_MISMATCH',reference=expected,candidate=y['output_ids']))
    return dict(reference=reference,candidate=candidate,compared=len(b),passed=bool(b) and not failures and (allow_reference_prefix or len(a)==len(b)),
        failures=failures,scope='Exact generated tokens at matched input and Seed. Not a full KV/Hidden state check.')


def main():
    initialize();plan=json.loads((ROOT/'state/native_path_recovery_plan.json').read_text())
    values=[metrics(n) for n in plan['jobs']]
    if not all(r['complete'] for r in values):print(json.dumps(dict(status='INCOMPLETE',metrics=values)));return
    a,b,c=plan['jobs'];checks=[compare(a,b),compare(a,c)]
    write(ROOT/'analysis/native_initial_qualification.json',dict(time=time.time(),metrics=values,checks=checks,
        passed=all(r['passed'] for r in checks),state_scope='Output-only N8 qualification. Full valid KV, Hidden, and layer-count qualification is still required.'))
    print(json.dumps(dict(passed=all(r['passed'] for r in checks),checks=checks,
        metrics=[{k:r[k] for k in ('name','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')} for r in values])))


if __name__=='__main__':main()
