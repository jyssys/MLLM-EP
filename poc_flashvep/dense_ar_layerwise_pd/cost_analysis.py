"""Summarize measured enclosures without adding percentile components."""
import json,re,statistics
from .common import *
from .analyze import metrics,percentile,records
from .serving_analysis import work
from .state_qualification import events


def distribution(values):
    return dict(count=len(values),p50=percentile(values,.5),p95=percentile(values,.95),p99=percentile(values,.99),maximum=max(values,default=None))


def analyze(plan,output):
    rows=[]
    for n in json.loads((ROOT/'state'/plan).read_text())['jobs']:
        folder=ROOT/'raw'/n
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
        spans={}
        for role in ('prefill','decode'):
            p=folder/f'CUDA_event_enclosures_{role}.json'
            if not p.exists():continue
            data=json.loads(p.read_text());selected=[x for x in data['rows'] if all(re.fullmatch(r'q\d+',r) for r in x['requests'])]
            spans[role]=dict(anchor_uncertainty_ms=data['anchor_uncertainty_ms'],
                prefill_event_enclosure_ms=distribution([x['elapsed_ms'] for x in selected if x['prefilling']]),
                Decode_event_enclosure_ms=distribution([x['elapsed_ms'] for x in selected if not x['prefilling']]),
                prefill_batch_sizes=sorted({len(x['requests']) for x in selected if x['prefilling']}),
                all_model_enclosures_ms=sum(x['elapsed_ms'] for x in selected))
        cfg=json.loads((ROOT/'configs'/(n+'.json')).read_text());rq=records(folder/'requests.jsonl')
        handoffs={x['request_id']:x['time'] for x in records(folder/'native_handoffs.jsonl')}
        life=[]
        for role in ('prefill','decode'):
            p=folder/f'KV_lifetime_{role}.json'
            life.append(json.loads(p.read_text()) if p.exists() else dict(passed='NOT_MEASURED'))
        rows.append(dict(name=n,layer_mode=cfg['layer_mode'],cut=cfg['layer_cut'],metrics=metrics(n),
            CUDA_stream_event_enclosures=spans,work=work(n),normal_KV_lifetime=life,
            handoff_ready_to_first_output_ms=distribution([1000*(r['first_token']-handoffs['q'+str(r['sample_id'])]) for r in rq]),
            scope='Handoff-ready is a source worker host timestamp. It is not the first transfer kernel. Serial and batch controls are independent calibration, not clean serving improvements.'))
    write(ROOT/'analysis'/output,dict(rows=rows,
        scope='Measured stream enclosures can include launch gaps. Native telemetry can include submission overhead. No exclusive component sum or p95 sum is reported as online TTFT.'))
    print(json.dumps([dict(name=r['name'],spans=r['CUDA_stream_event_enclosures'],TTFT_median=r['metrics']['TTFT_p50_ms'],work=r['work']) for r in rows],indent=2))


if __name__=='__main__':
    import sys
    analyze(*sys.argv[1:])
