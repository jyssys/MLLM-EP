"""Export actual Nsight occurrences and keep service separate from causality."""
import argparse
import ast
import collections
import csv
import gzip
import sqlite3
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_dllm_multigpu_e2e_atlas.intervals import kernel_partition,union_ms


def classify(name):
    low=name.lower()
    if 'nccl' in low:return 'TP_ALL_REDUCE' if 'allreduce' in low else 'OTHER_NCCL'
    if 'flash_fwd' in low or 'fmha' in low:return 'ATTENTION_COMPUTE'
    if 'fused_moe_kernel' in low:return 'ROUTED_EXPERT_COMPUTE'
    if any(v in low for v in ['moe','align_block','topk','top_k']):return 'ROUTER_PERMUTATION'
    if 'memcpy' in low:return 'IN_DEVICE_COPY_KERNEL_UNRESOLVED'
    if any(v in low for v in ['gemm','gemv','nvjet','xmma']):return 'DENSE_GEMM_UNRESOLVED'
    if any(v in low for v in ['rms','layer_norm','rotary','rope']):return 'NORM_ROPE'
    if 'elementwise' in low:return 'ELEMENTWISE_UNRESOLVED'
    return 'OTHER_KERNEL_UNRESOLVED'


def replay_count(selected):
    return len({k['correlation'] for k in selected
        if k.get('node') is not None or k['name']=='NATIVE_CUDA_GRAPH_OPAQUE'})


def table(path,records):
    if not records:return
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in records))),lineterminator='\n')
        writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sqlite',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    conn=sqlite3.connect(args.sqlite);conn.row_factory=sqlite3.Row
    strings=dict(conn.execute('select id,value from StringIds'))
    nvtx=[]
    for r in conn.execute('select * from NVTX_EVENTS where end is not null'):
        row=dict(r);text=row['text'] or strings.get(row['textId'],'');row['text']=text
        if text.startswith(('ATLAS','DIAG','DIAGNOSTIC_')):nvtx.append(row)
    requests={r['globalTid']:r for r in nvtx if r['text'].startswith('ATLAS_REQUEST|')}
    spans_by_tid=collections.defaultdict(list)
    for row in nvtx:spans_by_tid[row['globalTid']].append(row)
    for values in spans_by_tid.values():values.sort(key=lambda r:r['end']-r['start'])
    runtime={}
    apis=[]
    for r in conn.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        r=dict(r);tid=r['globalTid']
        if tid not in requests:continue
        root=requests[tid]
        if not(root['start']<=r['start']<=root['end']):continue
        containing=[s for s in spans_by_tid[tid] if s['start']<=r['start'] and s['end']>=r['end']]
        tag=containing[0]['text'] if containing else 'UNRESOLVED'
        diag=any(s['text'].startswith('DIAGNOSTIC_') for s in containing)
        iteration=next((s['text'].split('|',3)[3] for s in containing if s['text'].startswith('ATLAS|') and s['text'].split('|',3)[3]!='None'),None)
        iteration=None if iteration is None else ast.literal_eval(iteration)
        record=dict(start=r['start']/1e6,end=r['end']/1e6,correlation=r['correlationId'],globalTid=tid,
            globalPid=tid//(2**24)*(2**24),name=strings[r['nameId']],scope=tag,diagnostic=diag,
            iteration=iteration,request_id=root['text'].split('|')[1])
        runtime[(record['globalPid'],r['correlationId'])]=record;apis.append(record)
    kernels=[];copies=[]
    for r in conn.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        r=dict(r);api=runtime.get((r['globalPid'],r['correlationId']))
        if api is None:continue
        name=strings[r['shortName']]
        kernels.append(dict(start=r['start']/1e6,end=r['end']/1e6,rank=r['deviceId'],physical_gpu=GPUS[r['deviceId']],
            stream=r['streamId'],node=r['graphNodeId'],correlation=r['correlationId'],name=name,
            category=classify(name),scope=api['scope'],diagnostic=api['diagnostic'],iteration=api['iteration'],
            request_id=api['request_id'],launch_start=api['start'],launch_end=api['end']))
    tables={r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables:
        for r in conn.execute('select * from CUPTI_ACTIVITY_KIND_GRAPH_TRACE'):
            r=dict(r);api=runtime.get((r['globalPid'],r['correlationId']))
            if api is None:continue
            kernels.append(dict(start=r['start']/1e6,end=r['end']/1e6,rank=r['deviceId'],physical_gpu=GPUS[r['deviceId']],
                stream=r['streamId'],node=None,correlation=r['correlationId'],name='NATIVE_CUDA_GRAPH_OPAQUE',
                category='NATIVE_GRAPH_ENVELOPE_NOT_KERNEL_SERVICE',scope=api['scope'],diagnostic=api['diagnostic'],
                iteration=api['iteration'],request_id=api['request_id'],launch_start=api['start'],launch_end=api['end']))
    copy_enum=dict(conn.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
    for r in conn.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
        r=dict(r);api=runtime.get((r['globalPid'],r['correlationId']))
        if api is None:continue
        copies.append(dict(start=r['start']/1e6,end=r['end']/1e6,rank=r['deviceId'],physical_gpu=GPUS[r['deviceId']],
            stream=r['streamId'],correlation=r['correlationId'],bytes=r['bytes'],name=copy_enum[r['copyKind']],
            category='CUDA_MEMCPY_'+copy_enum[r['copyKind']],scope=api['scope'],diagnostic=api['diagnostic'],
            iteration=api['iteration'],request_id=api['request_id']))
    summary=[];budgets=[];services=[]
    for rank in range(4):
        selected=[k for k in kernels+copies if k['rank']==rank and not k['diagnostic']]
        if not selected:continue
        begin=min(k['start'] for k in selected);end=max(k['end'] for k in selected)
        partition=kernel_partition(selected,begin,end)
        for part in partition:budgets.append(dict(rank=rank,physical_gpu=GPUS[rank],**part,
            validity='EXCLUSIVE_RANK_ACTIVITY_NOT_PROVEN_FULL_E2E_CRITICAL_PATH'))
        grouped=collections.defaultdict(list)
        for k in selected:grouped[k['category']].append(k)
        for category,group in grouped.items():
            services.append(dict(rank=rank,category=category,count=len(group),
                service_ms=sum(k['end']-k['start'] for k in group),
                union_ms=union_ms([(k['start'],k['end']) for k in group])))
        summary.append(dict(rank=rank,physical_gpu=GPUS[rank],activity_envelope_ms=end-begin,
            active_union_ms=union_ms([(k['start'],k['end']) for k in selected]),
            idle_inside_envelope_ms=sum(p['end']-p['start'] for p in partition if p['active_count']==0),
            additive_service_ms=sum(k['end']-k['start'] for k in selected),
            graph_replay_correlations=replay_count(selected),
            all_to_all_kernels=sum('alltoall' in k['name'].lower() for k in selected)))
    for name,data in [('kernels',kernels),('copies',copies),('runtime',apis),('nvtx',nvtx)]:
        with gzip.open(out/f'{name}.json.gz','wt') as stream:json.dump(data,stream)
    table(out/'kernel_service.csv',services);table(out/'exclusive_rank_activity.csv',budgets)
    write(out/'summary.json',dict(ranks=summary,source=str(Path(args.sqlite)),sqlite_sha256=sha(args.sqlite),
        profile_scope='DIAGNOSTIC_NODE_TRACE; never headline clean latency',
        source_clock='Nsight common capture clock in ms; CUDA deviceId maps logical rank, checked against worker UUIDs',
        dependency_status='PARTIAL: stream/node IDs and launch correlation observed. Complete graph-edge/rank-wait attribution unresolved.',
        exclusive_activity_is_removable_cost=False,actual_replay_occurrences=True,
        last_replay_multiplication=False,external_diag_excluded=any(k['diagnostic'] for k in kernels)))
    print('Parsed actual kernels',len(kernels),'physical copies',len(copies),'rank summaries',summary)


if __name__=='__main__':main()
