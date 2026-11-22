"""Audit model batches and graph calls. Do not infer exclusive GPU time."""
import collections
import json
import time
from .common import ROOT,write


def main():
    rows=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.name.startswith(('ThrottleAB_Discovery200_','RemoteSetupAB_Discovery200_')):
            continue
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:
            continue
        cfg=json.loads((folder/'configuration.json').read_text())
        data=json.loads((folder/'actual_queries_decode.json').read_text())['rows']
        batches=collections.defaultdict(list)
        for r in data:
            batches[r['batch_id']].append(r)
        types=collections.Counter()
        for group in batches.values():
            p=sum(r['start']<r['original_input_tokens'] for r in group)
            d=len(group)-p
            types['MIXED' if p and d else 'DECODE_ONLY' if d else 'PREFILL_ONLY']+=1
        steps=json.loads((folder/'engine_steps_decode.json').read_text())
        runtime=json.loads((folder/'actual_runtime_decode.json').read_text())
        rows.append(dict(name=folder.name,QPS=cfg['qps'],axis=cfg.get('axis_kind','Native'),
            D_compute_deferral=bool(cfg.get('d_compute_throttle')),
            D_READ_deferral=bool(cfg.get('defer_D_remote_setup')),
            compile_mode=runtime['compile_mode'],cudagraph_mode=runtime['cudagraph_mode'],
            actual_model_batches=len(batches),logical_batch_counts=dict(types),
            graph_replay_calls=steps['graph_replays'],
            first_query_scope='A Native last-prompt-query recomputation is logical PREFILL but can use a one-query graph.',
            evidence='TRACE_ANALYSIS'))
    write(ROOT/'analysis/decode_graph_and_mixed_path_audit_v1.json',dict(time=time.time(),rows=rows,
        evidence='TRACE_ANALYSIS',
        scope='Actual host prepare records and graph-manager replay counters after the measured boundary. '
            'Counts are not graph GPU durations or an exclusive per-operator cost decomposition. '
            'Prefill deferral changes mixed-batch frequency and graph eligibility together. '
            'These counters do not isolate their separate causal contributions. '
            'READ deferral also changes associated transfer/stream timing. A host-only DMA-independent cause is not established.'))
    for row in rows:
        print(row['name'],row['logical_batch_counts'],'graph_calls',row['graph_replay_calls'],flush=True)


if __name__=='__main__':
    main()
