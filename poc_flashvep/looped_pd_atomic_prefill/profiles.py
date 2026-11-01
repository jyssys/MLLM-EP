"""Read recorded CUDA intervals. Never substitute NVML for kernel busy."""
import sqlite3
import subprocess
import collections
import bisect
from .common import *
from .evidence import status,compare
from .analyze import save_csv


def union(intervals):
    total=0;end=None
    for lo,hi in sorted(intervals):
        if end is None or lo>end:total+=hi-lo;end=hi
        elif hi>end:total+=hi-end;end=hi
    return total


def pid_roles(folder,cfg):
    result={}
    for p in folder.glob('atomic_spans_*.json'):
        role=json.loads(p.read_text())['role'];pid=int(p.stem.rsplit('_',1)[1])
        if role in ('prefill','decode'):result[pid]=(role,cfg['gpus'][0 if role=='prefill' else 1])
    for role,gpu in zip(('prefill','decode'),cfg['gpus']):
        p=folder/(role+'_ready.json')
        if p.exists():result[json.loads(p.read_text())['pid']]=(role,gpu)
    return result


def parse(folder,cfg):
    report=folder/'nsys_report.nsys-rep';database=folder/'nsys_report.sqlite'
    if not report.exists():return dict(name=cfg['name'],status='NOT_AVAILABLE',reason='No completed Nsight report')
    if not database.exists():
        result=subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=false','--output='+str(database),str(report)],text=True,capture_output=True)
        write(folder/'nsys_export_result.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:return dict(name=cfg['name'],status='BLOCKED_PROFILE_EXPORT',reason=result.stderr[-1000:])
    conn=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    schema={name:[r[1] for r in conn.execute('PRAGMA table_info("'+name+'")')] for name in tables}
    write(folder/'nsys_schema.json',schema)
    if not {'NVTX_EVENTS','CUPTI_ACTIVITY_KIND_KERNEL'}<=tables:
        return dict(name=cfg['name'],status='BLOCKED_PROFILE_ACTIVITY',reason='No NVTX or CUDA kernel activity table')
    strings={r['id']:r['value'] for r in conn.execute('SELECT * FROM StringIds')} if 'StringIds' in tables else {}
    def label(r):return r['text'] if 'text' in r.keys() and r['text'] else strings.get(r['textId'],'') if 'textId' in r.keys() else ''
    windows=[r for r in conn.execute('SELECT * FROM NVTX_EVENTS') if label(r)=='APD_MEASURED_WINDOW' and r['end'] is not None]
    if len(windows)!=1:return dict(name=cfg['name'],status='BLOCKED_PROFILE_WINDOW',windows=len(windows))
    marker=windows[0];window=json.loads((folder/'window.json').read_text());origin=marker['start']
    if cfg.get('nsys_P_only'):
        anchor=json.loads((folder/'P_profile_anchor.json').read_text())
        origin+=int((window['start']-anchor['time'])*1e9)
    duration=window['last_arrival']-window['start'];left=origin+int(.2*duration*1e9);right=origin+int(.9*duration*1e9)
    compatibility=duration<=0
    if compatibility:left=marker['start'];right=marker['end']
    roles=pid_roles(folder,cfg);kernels=collections.defaultdict(list);names=collections.Counter();events=[];unmapped=0
    forward=collections.defaultdict(list)
    for r in conn.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        text=label(r)
        if not text.startswith('APD_FORWARD|'):continue
        raw=r['globalTid'] if 'globalTid' in r.keys() else None
        pid=(raw>>24)&0xffffff if raw is not None else None
        if pid in roles and marker['start']<=r['start']<marker['end']:
            forward[pid].append(dict(start=r['start'],end=r['end'],label=text,kernels=[]))
    for pid in forward:forward[pid].sort(key=lambda r:r['start'])
    starts={pid:[r['start'] for r in values] for pid,values in forward.items()};launches={}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables or not {'globalTid','correlationId','start'}<=set(schema[table]):continue
        for r in conn.execute('SELECT * FROM "'+table+'" WHERE start>=? AND start<=?',(marker['start'],marker['end'])):
            pid=(r['globalTid']>>24)&0xffffff
            if pid in forward:launches[(pid,r['correlationId'])]=r['start']
    attributed=collections.Counter();unattributed=collections.Counter()
    for r in conn.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<=?',(marker['start'],marker['end'])):
        raw=r['globalPid'];pid=raw if raw in roles else (raw>>24)&0xffffff
        if pid not in roles:continue
        timestamp=launches.get((pid,r['correlationId']));index=bisect.bisect_right(starts.get(pid,[]),timestamp)-1 if timestamp is not None else -1
        if index>=0 and timestamp<forward[pid][index]['end']:
            forward[pid][index]['kernels'].append((r['start'],r['end']));attributed[roles[pid][0]]+=1
        else:unattributed[roles[pid][0]]+=1
    batch_rows=[]
    for pid,values in forward.items():
        for value in values:
            tokens=value['label'].split('|');activity=value['kernels'];complete=bool(activity) and all(b<=marker['end'] for a,b in activity)
            batch_rows.append(dict(role=roles[pid][0],physical_GPU=roles[pid][1],operation=tokens[1],
                logical_rows=int(tokens[2].split('=')[1]),loop_count=int(tokens[3].split('=')[1]),
                host_submit_ms=(value['end']-value['start'])/1e6,recorded_kernels=len(activity),
                recorded_kernel_union_ms=union(activity)/1e6 if activity else None,
                recorded_GPU_span_ms=(max(b for a,b in activity)-min(a for a,b in activity))/1e6 if activity else None,
                complete_inside_measured_window=complete,
                scope='Actual CUDA-launch correlation to one ModelRunner submission. Logical rows are not fixed-tile Down GEMM rows. Missing graph internals cannot become zero compute.'))
    save_csv('profiles/'+cfg['name']+'_actual_batch_gpu_time.csv',batch_rows)
    for r in conn.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<?',(left,right)):
        raw=r['globalPid'] if 'globalPid' in r.keys() else None
        pid=raw if raw in roles else (raw>>24)&0xffffff if raw is not None else None
        if pid not in roles:unmapped+=1;continue
        role,gpu=roles[pid];a=max(left,r['start']);b=min(right,r['end']);kernels[role].append((a,b))
        sid=r['demangledName'] if 'demangledName' in r.keys() else r['shortName'] if 'shortName' in r.keys() else None
        name=strings.get(sid,'UNKNOWN_KERNEL');names[(role,gpu,name)]+=1
        events.append(dict(role=role,physical_GPU=gpu,start_s=(a-origin)/1e9,end_s=(b-origin)/1e9,kernel=name))
    graph_mode=cfg.get('nsys_graph_trace','node')=='graph';graph_intervals=collections.defaultdict(list)
    if graph_mode:
        for table,columns in schema.items():
            if not table.startswith('CUPTI_ACTIVITY_KIND_') or 'GRAPH' not in table or not {'start','end','globalPid'}<=set(columns):continue
            for r in conn.execute('SELECT * FROM "'+table+'" WHERE end>? AND start<?',(left,right)):
                raw=r['globalPid'];pid=raw if raw in roles else (raw>>24)&0xffffff
                if pid in roles:graph_intervals[roles[pid][0]].append((max(left,r['start']),min(right,r['end'])))
    summary=[]
    for role,index in [('prefill',0),('decode',1)]:
        intervals=kernels[role];busy=union(intervals)
        partial=(graph_mode or cfg.get('nsys_P_only')) and role=='decode'
        summary.append(dict(role=role,physical_GPU=cfg['gpus'][index],kernel_busy_fraction=busy/(right-left) if right>left and not partial else None,
            recorded_non_graph_kernel_fraction=busy/(right-left) if right>left else None,
            GPU_graph_active_fraction=union(graph_intervals[role])/(right-left) if right>left and graph_intervals[role] else None,
            kernel_busy_union_ms=busy/1e6,window_ms=(right-left)/1e6,kernel_count=len(intervals),
            idle_or_unrecorded_fraction=1-busy/(right-left) if right>left and not partial else None,
            scope=('Full measured-cohort window for tool compatibility only. ' if compatibility else 'Central 20-90% arrival window. ')+
                ('Decode is excluded from this P-only profile. No D activity or idle inference. ' if cfg.get('nsys_P_only') and role=='decode' else
                 'Decode graph internals are not collected. Graph duration is not exclusive kernel busy. ' if partial else '')+
                'Union of recorded CUDA kernels. Not SM occupancy. Unrecorded DMA is not GPU idleness.'))
    save_csv('profiles/'+cfg['name']+'_kernel_names.csv',[dict(role=k[0],physical_GPU=k[1],kernel=k[2],invocations=v) for k,v in names.items()])
    save_csv('profiles/'+cfg['name']+'_kernel_intervals.csv',events)
    APIs=collections.defaultdict(list);API_intervals=collections.defaultdict(list)
    table='CUPTI_ACTIVITY_KIND_RUNTIME'
    if table in tables and {'start','end','globalTid','nameId'}<=set(schema[table]):
        for r in conn.execute('SELECT * FROM "'+table+'" WHERE end>? AND start<?',(left,right)):
            pid=(r['globalTid']>>24)&0xffffff
            if pid not in roles:continue
            role,gpu=roles[pid];interval=(max(left,r['start']),min(right,r['end']))
            APIs[(role,gpu,strings.get(r['nameId'],'UNKNOWN_API'))].append(interval)
            API_intervals[role].append(interval)
    save_csv('profiles/'+cfg['name']+'_host_cuda_apis.csv',[dict(role=k[0],physical_GPU=k[1],API=k[2],calls=len(v),
        cumulative_host_ms=sum(b-a for a,b in v)/1e6,
        scope='Profiled host CUDA API durations. Helper threads and device work can overlap. Not additive E2E components.') for k,v in APIs.items()])
    copies=[];table='CUPTI_ACTIVITY_KIND_MEMCPY'
    if table in tables and {'start','end','globalPid'}<=set(schema[table]):
        for r in conn.execute('SELECT * FROM "'+table+'" WHERE end>? AND start<?',(left,right)):
            raw=r['globalPid'];pid=raw if raw in roles else (raw>>24)&0xffffff
            if pid not in roles:continue
            role,gpu=roles[pid]
            copies.append(dict(owner_role=role,owner_physical_GPU=gpu,start_s=(max(left,r['start'])-origin)/1e9,
                end_s=(min(right,r['end'])-origin)/1e9,bytes=r['bytes'] if 'bytes' in r.keys() else None,
                raw_copy_kind=r['copyKind'] if 'copyKind' in r.keys() else None,
                scope='Recorded CUDA copy activity only. This table does not prove that every NIXL/UCX DMA operation is visible.'))
    save_csv('profiles/'+cfg['name']+'_recorded_cuda_copies.csv',copies)
    output=dict(name=cfg['name'],status='MEASURED_TOOL_COMPATIBILITY' if compatibility else 'MEASURED_PROFILED_CUDA_ACTIVITY',central_arrival_window_s=.7*duration,
        phase_marker_ns=[marker['start'],marker['end']],effective_parent_window_origin_ns=origin,profiled_process_scope='P_ONLY' if cfg.get('nsys_P_only') else 'PROCESS_TREE',devices=summary,unmapped_kernel_events=unmapped,
        host_CUDA_API_interval_union_fraction={role:union(v)/(right-left) for role,v in API_intervals.items()} if right>left else {},
        recorded_CUDA_copy_events=len(copies),recorded_CUDA_copy_bytes=sum(r['bytes'] or 0 for r in copies),
        attributed_forward_kernel_events=dict(attributed),unattributed_forward_kernel_events=dict(unattributed),
        process_ID_scope='Nsight globalPid contains a host tag. Linux PID is bits 24-47 and is checked against measured worker PIDs.',
        source='Actual Nsight/CUPTI kernel intervals, never inferred from host duration.')
    clean_name=cfg.get('profile_clean_reference')
    if clean_name and status(clean_name)=='COMPLETE':
        clean=json.loads((ROOT/'raw'/clean_name/'window.json').read_text())
        a=window['end']-window['start'];b=clean['end']-clean['start']
        parity=compare(clean_name,cfg['name'])
        output.update(clean_reference=clean_name,profile_wall_overhead_fraction=a/b-1,
            exact_profile_clean_output_parity=parity['passed'],
            compared_profile_clean_requests=parity['compared'],
            overhead_scope='Single matched arrival-to-drain comparison. It is not an independent-start confidence interval.',
            eligible_profile_transfer_to_clean=abs(a/b-1)<=.05 and unmapped==0 and summary[0]['kernel_count']>0 and parity['passed'])
    else:output['eligible_profile_transfer_to_clean']=False
    write(folder/'profile_activity_analysis.json',output);return output


def main():
    output=[]
    for p in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(p.read_text())
        if not isinstance(cfg,dict) or not (cfg.get('nsys') or cfg.get('nsys_P_only')) or status(cfg['name'])!='COMPLETE':continue
        try:output.append(parse(ROOT/'raw'/cfg['name'],cfg))
        except Exception as error:output.append(dict(name=cfg['name'],status='BLOCKED_PROFILE_ANALYSIS',error=repr(error)))
    write(ROOT/'profile_activity.json',output);print('PROFILE_ANALYSES',len(output));return output


if __name__=='__main__':main()
