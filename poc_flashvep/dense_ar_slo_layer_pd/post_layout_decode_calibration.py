"""Reuse retained async Decode events instead of borrowing a stale lookup."""
import fcntl
from .common import *
from .analyze import records,percentile


def _build():
    target=ROOT/'configs/canonical_cost_model_v5.json'
    if target.exists():return json.loads(target.read_text())
    model=json.loads((ROOT/'configs/canonical_cost_model_v4.json').read_text());table={};audit=[]
    for name in model['primitive_calibration_names']:
        folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
        batch=cfg['P_group_size'];requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
        events=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())['rows']
        steps=[];seen=set()
        for event in events:
            ids=event.get('requests',[])
            if not event.get('graph_replay') or event['query_tokens']!=len(ids):continue
            if len(ids)!=batch or not all(r in requests for r in ids):continue
            key=tuple(sorted(ids));assert key not in seen,'Unexpected second Decode step in cap2 calibration'
            seen.add(key);host=[]
            for rid in ids:
                a,b=requests[rid]['token_events'];assert b['count']-a['count']==1
                host.append(1000*(b['worker_time']-a['worker_time']))
            steps.append(dict(requests=ids,stream_event_ms=event['elapsed_ms'],
                maximum_worker_emission_interval_ms=max(host)))
        assert len(steps)==15,(name,len(steps))
        table[str(batch)]=percentile([r['maximum_worker_emission_interval_ms'] for r in steps],.95)
        audit.append(dict(name=name,batch=batch,steps=steps,p95_worker_group_max_ms=table[str(batch)],
            stream_event_p95_ms=percentile([r['stream_event_ms'] for r in steps],.95)))
    model.update(model_version='CANONICAL_ASYNC_WITH_RECALIBRATED_DECODE_V5',
        decode_step_by_batch_p95_ms=table,decode_step_p95_ms=max(table.values()),
        decode_recalibration_names=[r['name'] for r in audit],
        borrowed_components=['negotiation_idle_ms','negotiation_busy_ms','uncertainty'],
        scope='Same exact async runtime and canonical layout. Prefix, suffix, handoff and commit tables remain v4. Recalibrate Decode from fifteen cap2 groups per batch. Use the p95 of each group maximum worker-emission interval. The existing online2ms Decode margin remains separate. CUDA stream events are retained for comparison, not called exclusive kernels. Scope is Input8000 and short Output16. These dependent small samples do not establish p99 or WCET. Validate completion residuals and deadline coverage on later starts.')
    write(target,model)
    write(ROOT/'analysis/post_layout_Decode_cost_calibration.json',dict(runs=audit,
        model_SHA256=digest(target),evidence='MEASURED',
        scope='Isolated instrumented cap2 calibration, not clean serving. Worker intervals include Host work. Stream event enclosures include launch gaps. No physical idle or GPU resource counters are inferred.'))
    return model


def build():
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        return _build()


if __name__=='__main__':build()
