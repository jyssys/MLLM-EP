"""Retain the frozen client-visible metric definitions and expose geometry."""
import collections
import json
from .runtime_binding import activate
activate()
from .common import ROOT,write,digest
from ..dense_ar_bi0_shortanswer_pd.metrics import summarize as original,delivery_intervals


def summarize(name):
    result=original(name);folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    result['configuration_QPS']=cfg.get('qps')
    result['offered_QPS']=None if cfg.get('serial') else cfg.get('qps')
    result['output_contract']='CONTROLLED_FIXED_OUTPUT' if cfg.get('ignore_eos') else 'NATURAL_EOS'
    result['evidence']='DIAGNOSTIC' if cfg.get('correctness_only') or cfg.get('kernel_activity') or cfg.get('serial') else 'MEASURED_ONLINE'
    result['execution_scope']='SERIAL_HOST_SERVICE_CONTROL' if cfg.get('serial') else 'OPEN_LOOP_ONLINE'
    result['actual_geometry']={}
    for path in folder.glob('actual_queries_*.json'):
        role=path.stem.removeprefix('actual_queries_');rows=json.loads(path.read_text())['rows'];batches={}
        for row in rows:batches.setdefault(row['batch_id'],[]).append(row)
        hist=collections.Counter()
        for values in batches.values():
            prefill=sum(v['API_prefilling'] for v in values);decode=len(values)-prefill
            hist[prefill,decode]+=1
        result['actual_geometry'][role]=dict(batch_count=len(batches),
            prefill_decode_request_histogram=[dict(prefill=p,decode=d,steps=n) for (p,d),n in sorted(hist.items())],
            maximum_requests=max((r['requests'] for r in rows),default=0),
            maximum_queries=max((r['actual_batch_queries'] for r in rows),default=0),
            maximum_padded_queries=max((r['padded_queries'] for r in rows),default=0))
    result.update(B=cfg['token_budget'],S=cfg['max_num_seqs'],GPU_pair=cfg['gpus'],
        dataset_SHA256=cfg['dataset_SHA256'],KV_block_tokens=cfg['KV_block_tokens'],
        Source_credit=cfg.get('source_credit_limit') if cfg.get('controlled_pipeline') else cfg.get('continuous_source_credit_limit',0),
        runtime_source_archive_SHA256=json.loads((folder/'dispatcher_result.json').read_text())['source_SHA256'])
    write(ROOT/'analysis'/(name+'_metrics.json'),result);return result


if __name__=='__main__':
    import sys
    r=summarize(sys.argv[1]);print(json.dumps({k:v for k,v in r.items() if k not in ('by_task','per_request','actual_geometry')},indent=2))
