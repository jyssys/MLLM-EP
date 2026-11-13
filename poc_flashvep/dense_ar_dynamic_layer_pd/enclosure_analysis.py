"""Do not label a model enclosure or its residual as exclusive GPU or CPU time."""
import collections,statistics
from .common import *
from .analyze import records,percentile


def execution_class(row):
    if row['role'].startswith('replica'):
        return ('COLOCATED_PREFILL',36) if row['prefilling'] and row['query_tokens']==8000*len(row['requests']) else ('COLOCATED_FULL36_DECODE',36)
    if row['role']=='prefill':
        return 'P_PREFIX',row['cut'] if row.get('layer_mode')=='MIGRATE' else 36
    suffix=(row.get('layer_mode')=='MIGRATE' and row['prefilling'] and row.get('cut',36)<36
        and row['query_tokens']==8000*len(row['requests']))
    return ('D_SUFFIX',row['cut']) if suffix else ('D_FULL36_DECODE',36)


def distribution(values):
    return dict(count=len(values),mean=statistics.mean(values) if values else None,
        p50=percentile(values,.5),p95=percentile(values,.95),p99=percentile(values,.99),
        maximum=max(values,default=None))


def analyze(name):
    folder=ROOT/'raw'/name;window=json.loads((folder/'window.json').read_text())
    requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    events={};details=[]
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    roles=('replica0','replica1') if cfg['arm']=='DP2' else ('prefill','decode')
    for role in roles:
        path=folder/f'CUDA_event_enclosures_{role}.json'
        if not path.exists():continue
        events[role]=json.loads(path.read_text())
        for row in events[role]['rows']:
            if not row['requests'] or not all(rid in requests for rid in row['requests']):continue
            details.append(dict(row,evidence='MEASURED_STREAM_ENCLOSURE_NOT_KERNEL_UNION'))
    grouped=collections.defaultdict(list)
    for row in details:
        kind,cut=execution_class(row)
        row['execution_class']=kind;row['effective_cut']=cut
        key=(row['role'],kind,cut,len(row['requests']))
        grouped[key].append(row)
    groups=[]
    for key,values in grouped.items():
        role,kind,cut,batch=key
        native=next((json.loads(p.read_text()) for p in folder.glob(f'native_{role}_*.json')
            if 'host_step_spans' in json.loads(p.read_text())),None)
        residuals=[]
        if native:
            spans=native['host_step_spans']
            for row in values:
                step=next((s for s in spans if s['start']<=row['host_submit']<=s['end']),None)
                if step:residuals.append(1000*(step['end']-step['start'])-row['elapsed_ms'])
        groups.append(dict(role=role,execution_class=kind,cut=cut,batch=batch,
            stream_enclosure_ms=distribution([r['elapsed_ms'] for r in values]),
            engine_Host_step_minus_model_event_ms=distribution(residuals),
            scope='The residual includes metadata, sampling, synchronization, and other GPU work. It is not exclusive CPU overhead.'))
    migration=[]
    handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    for rid,req in requests.items():
        suffix=next((r for r in details if r['execution_class']=='D_SUFFIX' and rid in r['requests']),None)
        if suffix and rid in handoffs:
            migration.append(dict(request_id=rid,layers=36-suffix['cut'],
                source_ready_to_suffix_start_ms=1000*(suffix['start_monotonic']-handoffs[rid]),
                suffix_enclosure_ms=suffix['elapsed_ms'],
                suffix_end_to_first_token_ms=1000*(req['first_token']-suffix['end_monotonic'])))
    return dict(name=name,groups=groups,migration=migration,enclosures=details,
        anchors={role:{k:v for k,v in event.items() if k!='rows'} for role,event in events.items()},
        physical_kernel_busy='NOT_MEASURED',exclusive_wire_time='NOT_IDENTIFIED',
        exclusive_CPU_time='NOT_IDENTIFIED',resource_contention_contribution='NOT_IDENTIFIED',
        scope='Measured stream enclosures and Host timestamps. No exclusive-time sum or causal contribution percentage is inferred.')


def main():
    rows=[]
    for path in sorted((ROOT/'raw').glob('enclosure_*_events_v1')):
        receipt=path/'dispatcher_result.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('complete'):rows.append(analyze(path.name))
    write(ROOT/'analysis/enclosure_decomposition.json',dict(results=rows))
    print('ENCLOSURE_DECOMPOSITION',len(rows))


if __name__=='__main__':main()
