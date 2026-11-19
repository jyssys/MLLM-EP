"""Use worker acceptance/output notices without naming them GPU kernels."""
import collections
import json
import statistics
from .common import ROOT,write
from .historical_audit import records


def analyze(name):
    folder=ROOT/'raw'/name
    host=json.loads((ROOT/'analysis'/(name+'_host_paths.json')).read_text())
    rows={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    source=[]
    for path in host['request_paths']:
        rid=path['request_id'];row=rows[rid]
        start,ready=path['boundaries_monotonic'][1:3]
        accepted=row['admission']
        assert start<=accepted<=ready
        source.append(dict(request_id=rid,submit_to_P_worker_accept_ms=1000*(accepted-start),
            P_worker_accept_to_ready_ms=1000*(ready-accepted)))
    destination=[]
    for c in host['cohorts']:
        group=[rows[rid] for rid in c['requests']]
        all_first=max(r['token_events'][0]['worker_time'] for r in group)
        last_worker=max(r['token_events'][-1]['worker_time'] for r in group)
        last_client=max(r['finish'] for r in group)
        spans=dict(D_submit_to_all_first_worker_ms=1000*(all_first-c['D_host_submit']),
            all_first_to_last_worker_ms=1000*(last_worker-all_first),
            last_worker_to_last_client_ms=1000*(last_client-last_worker))
        total=1000*(last_client-c['D_host_submit'])
        assert abs(sum(spans.values())-total)<1e-5
        destination.append(dict(cohort=c['cohort'],size=c['size'],requests=c['requests'],
            spans_ms=spans,host_slot_ms=total,
            per_request_ms={k:v/c['size'] for k,v in spans.items()}))
    value=dict(name=name,source=source,destination=destination,
        mean_source_ms={k:statistics.mean(r[k] for r in source) for k in source[0] if k!='request_id'},
        mean_D_per_request_ms={k:statistics.mean(r['per_request_ms'][k] for r in destination)
            for k in destination[0]['per_request_ms']},
        evidence='TRACE_ANALYSIS',
        scope='Exact notice-clock identities. P acceptance is a worker software boundary. All-first D notice follows each cohort member first output. Preparation contains transfer, import, scheduling and Forward. Worker Decode residence includes launch/scheduling gaps. None is exclusive GPU compute or DMA duration.')
    write(ROOT/'analysis'/(name+'_slot_decomposition.json'),value)
    return value


def main():
    values=[]
    for path in sorted((ROOT/'analysis').glob('*_service_queue_audit.json')):
        name=path.name.removesuffix('_service_queue_audit.json')
        values.append(analyze(name))
    write(ROOT/'analysis/slot_decomposition_index.json',dict(names=[r['name'] for r in values],
        evidence='TRACE_ANALYSIS',scope='All available new host/worker boundary decompositions. Profiles remain separate from clean Primary starts.'))
    print('SLOT_DECOMPOSITIONS',len(values),flush=True)


if __name__=='__main__':
    main()
