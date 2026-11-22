"""Use additive observed boundaries without renaming host time as GPU time."""
import collections
import json
import numpy as np
from .common import ROOT,write


def records(path):return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def analyze(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((folder/'configuration.json').read_text())
    assert json.loads((folder/'dispatcher_result.json').read_text())['complete']
    assert cfg['method']!='Agg' and not cfg.get('reference_single')
    if cfg.get('whole_prefill_deflection'):
        from .deflection_service_accounting import analyze as routed
        return routed(name)
    requests={f"q{r['sample_id']}":r for r in records(folder/'requests.jsonl')}
    submissions={(r['request_id'],r['role']):r['time'] for r in records(folder/'request_submissions.jsonl')}
    admissions={(r['request_id'],r['role']):r['worker_accepted'] for r in records(folder/'admissions.jsonl')}
    ready={r['request_id']:r['source_ready'] for r in records(folder/'handoffs.jsonl')}
    queries={}
    for role in ('prefill','decode'):
        values=json.loads((folder/f'actual_queries_{role}.json').read_text())['rows']
        for row in values:queries.setdefault((row['request_id'],role),row['time'])
    rows=[]
    for rid,r in sorted(requests.items(),key=lambda item:int(item[0][1:])):
        p=submissions[rid,'prefill'];accepted=admissions[rid,'prefill'];pq=queries[rid,'prefill']
        ds=submissions[rid,'decode'];da=admissions[rid,'decode'];dq=queries[rid,'decode']
        first_worker=r['token_events'][0]['worker_time'];first_client=r['first_token']
        boundaries=[r['arrival'],p,accepted,pq,ready[rid],ds,da,dq,first_worker,first_client]
        labels=['Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms',
            'P_prepare_to_ready_ms','Ready_to_D_submit_ms','D_submit_to_accept_ms',
            'D_accept_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms']
        row=dict(request_id=rid,prompt_id=r['prompt_id'],input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],
            TTFT_ms=r['TTFT_ms'],D_decode_residency_ms=1000*(r['finish']-first_client))
        row.update({key:1000*(b-a) for key,a,b in zip(labels,boundaries,boundaries[1:])})
        assert abs(sum(row[key] for key in labels)-row['TTFT_ms'])<1e-3
        row['P_submit_to_ready_ms']=1000*(ready[rid]-p)
        row['D_submit_to_finish_ms']=1000*(r['finish']-ds)
        rows.append(row)
    fields=[k for k in rows[0] if k.endswith('_ms')]
    result=dict(name=name,rows=rows,means={k:float(np.mean([r[k] for r in rows])) for k in fields},
        execution_scope='C0_ONE_ACTIVE_PER_STAGE' if cfg.get('controlled_pipeline') else 'C3_CONTINUOUS',
        evidence='TRACE_ANALYSIS',
        scope='Exact additive mean client-TTFT boundaries. Prepare is a host input-preparation boundary. '
            'It is not GPU kernel start. Service enclosures include host work,transfer and synchronization. '
            'The sum does not decompose TTFT p95. Decode residency is not pure decode GPU time. '
            'Under C3,P submit-to-ready includes worker queueing and other requests between chunks. '
            'It is not an exclusive per-request service cost.')
    write(ROOT/'analysis'/(name+'_service_accounting.json'),result)
    return result


def compare(left,right):
    a=analyze(left);b=analyze(right)
    for field in ('dataset_SHA256','arrival_offsets','output_cap','ignore_eos','batch_invariant','gpus'):
        ca=json.loads((ROOT/'raw'/left/'configuration.json').read_text())
        cb=json.loads((ROOT/'raw'/right/'configuration.json').read_text())
        assert ca[field]==cb[field],('Unmatched trace contract',field)
    aa={r['request_id']:r for r in a['rows']};bb={r['request_id']:r for r in b['rows']}
    assert set(aa)==set(bb) and all(aa[r]['prompt_id']==bb[r]['prompt_id'] for r in aa)
    return dict(left=left,right=right,mean_delta_ms={k:a['means'][k]-b['means'][k] for k in a['means']},
        output_work_equal=all(aa[r]['output_tokens']==bb[r]['output_tokens'] for r in aa),
        scope='Observed path differences locate a delay. They do not identify an exclusive GPU operator cause.')


if __name__=='__main__':
    import sys
    print(json.dumps(analyze(sys.argv[1])['means'],indent=2))
