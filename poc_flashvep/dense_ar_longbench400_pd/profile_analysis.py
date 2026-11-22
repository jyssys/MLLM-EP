"""Join kernel records to actual engine frames through CUPTI API correlation."""
import collections
import csv
import json
import numpy as np
from .common import ROOT,write


def union(intervals):
    total=0;left=right=None
    for a,b in sorted(intervals):
        if right is None:left,right=a,b
        elif a<=right:right=max(right,b)
        else:total+=right-left;left,right=a,b
    return total+(right-left if right is not None else 0)


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),
        p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)),max=float(np.max(values))) if values else None


def analyze(name,role='decode'):
    folder=ROOT/'raw'/name;path=folder/f'activity_{role}'
    counters=json.loads((path/'activity_counters.json').read_text());assert counters['passed']
    scopes=json.loads((path/'correlation_scopes.json').read_text());steps={s['external_id']:s for s in scopes['steps']}
    operators={s['external_id']:s for s in scopes['operators']}
    cfg=json.loads((folder/'configuration.json').read_text())
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    lengths={'q'+str(i):r['input_tokens'] for i,r in enumerate(data)}
    partition_path=folder/f'axis_events_{role}.json'
    partitions={r['request_id']:r for r in json.loads(partition_path.read_text())['rows'] if r['event']=='PARTITION'} if partition_path.exists() else {}
    def category(op):
        if op is None:return 'GRAPH_OR_UNTAGGED_RUNTIME'
        value=op['category'];prefix=op.get('module_prefix') or ''
        if value=='GEMM_O_OR_DOWN':
            if prefix.endswith('.o_proj'):return 'GEMM_O_PROJ'
            if prefix.endswith('.down_proj'):return 'GEMM_DOWN'
        return value
    ids={}
    for row in csv.DictReader((path/'kernels.csv.external.csv').open()):
        if int(row['external_kind'])==3:ids[int(row['correlation'])]=int(row['external_id'])
    frames=collections.defaultdict(list);categories=collections.defaultdict(list);all_kernels=[];copies=[];unmapped=[]
    for row in csv.DictReader((path/'kernels.csv').open()):
        start,end=int(row['start_ns']),int(row['end_ns'])
        if row['kind']!='kernel':
            if row['kind'] in ('memcpy','memcpy2'):copies.append((start,end))
            continue
        all_kernels.append((start,end));external=ids.get(int(row['correlation']))
        op=operators.get(external)
        step=op['step_id'] if op else external
        if step not in steps:unmapped.append((start,end));continue
        record=dict(start_ns=start,end_ns=end,graph=int(row['graph_id'])>0,
            category=category(op),name_id=int(row['name_id']),operator_external_id=external if op else None)
        frames[step].append(record);categories[record['category']].append((start,end))
    rows=[]
    for step_id,kernels in frames.items():
        step=steps[step_id];graphs=[r for r in kernels if r['graph']]
        queries=step['queries'];single=queries[0] if len(queries)==1 else None
        phase='NO_QUERY_FRAME'
        if single is not None and single['request_id'] in lengths:
            length=lengths[single['request_id']];part=partitions.get(single['request_id'],{})
            if role=='prefill':phase='P_PREFIX_PREFILL'
            elif single['start']>=length:phase='DECODE'
            elif not part.get('upper') and not part.get('suffix') and single['start']==length-1 and single['count']==1:phase='FIRST_PROMPT_QUERY_RECOMPUTE'
            elif part.get('upper') and single['start']<part['prefix']:phase='PREFIX_UPPER_CONTINUATION'
            else:phase='SUFFIX_FULL36_CONTINUATION'
        elif len(queries)>1:phase='MULTI_REQUEST_FRAME'
        row_categories={key:union([(r['start_ns'],r['end_ns']) for r in kernels if r['category']==key])/1e6 for key in {r['category'] for r in kernels}}
        rows.append(dict(step_id=step_id,queries=queries,kernel_count=len(kernels),phase=phase,
            GPU_kernel_union_ms=union([(r['start_ns'],r['end_ns']) for r in kernels])/1e6,
            GPU_kernel_enclosure_ms=(max(r['end_ns'] for r in kernels)-min(r['start_ns'] for r in kernels))/1e6,
            full_decode_graph_kernel_union_ms=union([(r['start_ns'],r['end_ns']) for r in graphs])/1e6 if graphs else None,
            full_decode_graph_enclosure_ms=(max(r['end_ns'] for r in graphs)-min(r['start_ns'] for r in graphs))/1e6 if graphs else None,
            engine_step_host_wall_ms=1000*(step['host_end']-step['host_start']),
            operator_kernel_union_ms=row_categories,outputs=step.get('outputs'),
            graph_kernel_count=len(graphs),request_id=single['request_id'] if single else None))
    first=[];decode=[]
    for row in rows:
        if row['request_id'] not in lengths or not row['graph_kernel_count']:continue
        query=row['queries'][0];length=lengths[row['request_id']]
        if query['start']==length-1:first.append(row)
        elif query['start']>=length:decode.append(row)
    groups={}
    fields=('GPU_kernel_union_ms','GPU_kernel_enclosure_ms','full_decode_graph_kernel_union_ms',
        'full_decode_graph_enclosure_ms','engine_step_host_wall_ms')
    for label,part in (('first_prompt_query_recompute',first),('subsequent_decode',decode)):
        groups[label]={field:stats([r[field] for r in part if r[field] is not None]) for field in fields}
    phase_summary={}
    for phase in sorted({r['phase'] for r in rows}):
        part=[r for r in rows if r['phase']==phase]
        keys=sorted({key for r in part for key in r['operator_kernel_union_ms']})
        phase_summary[phase]=dict(frames=len(part),
            total_GPU_kernel_union_ms=sum(r['GPU_kernel_union_ms'] for r in part),
            per_frame_GPU_kernel_union_ms=stats([r['GPU_kernel_union_ms'] for r in part]),
            tagged_operator_kernel_union_ms={key:sum(r['operator_kernel_union_ms'].get(key,0) for r in part) for key in keys})
    operation_rows=[]
    for identifier,op in operators.items():
        intervals=[(r['start_ns'],r['end_ns']) for r in frames.get(op['step_id'],[]) if r['operator_external_id']==identifier]
        shape=op.get('input_shape');weight=op.get('weight_shape')
        operation_rows.append(dict(**op,attributed_GPU_kernel_union_ms=union(intervals)/1e6,
            actual_GEMM_MNK=dict(M=int(np.prod(shape[:-1])),N=weight[0],K=weight[1])
                if category(op).startswith('GEMM') and shape and weight and len(weight)==2 else None))
    result=dict(name=name,role=role,frames=rows,summary=groups,kernel_records=len(all_kernels),
        phase_summary=phase_summary,actual_operators=operation_rows,
        unmapped_kernels=len(unmapped),mapped_kernel_fraction=1-len(unmapped)/max(1,len(all_kernels)),
        total_kernel_union_ms=union(all_kernels)/1e6,copy_engine_union_ms=union(copies)/1e6 if cfg.get('activity_copies') else None,
        operator_kernel_union_ms={key:union(values)/1e6 for key,values in categories.items()},
        evidence='DIAGNOSTIC_PENDING_OBSERVER_GATE',
        scope='GPU-exclusive union uses kernel timestamps. GPU enclosure includes gaps between kernels. '
            'Engine wall time can overlap asynchronous GPU work. These values must not be added. '
            'Graph kernels map through runtime API correlation,not host interval containment. '
            'First-query recomputation is separate from subsequent decode. Operator percentages require measured attribution coverage.')
    write(ROOT/'analysis'/(name+'_'+role+'_kernel_decomposition.json'),result)
    return result


if __name__=='__main__':
    import sys
    value=analyze(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'decode')
    print(json.dumps({k:v for k,v in value.items() if k!='frames'},indent=2))
