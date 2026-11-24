"""Measure logical READ holds and throttle exposure without calling them idle."""
import collections
import json
import time
import numpy as np
from .common import ROOT,REPO,write,digest,preserve
from .registry import historical_runs,current_runs
from .request_trace import trace,external
from .failure_analysis import save_csv
from ..dense_ar_p_critical_path.critical_analysis import Timeline


def state_spans(rows,left,right,field):
    rows=sorted(rows,key=lambda r:r['time'])
    if not rows:return [],[]
    active=[];known=[]
    for index,row in enumerate(rows):
        a=max(left,row['time']);b=min(right,rows[index+1]['time'] if index+1<len(rows) else right)
        if a>=b or field not in row:continue
        known.append([a,b])
        if row[field]>0:active.append([a,b])
    return active,known


def read_holds(rows):
    opened={};spans=collections.defaultdict(list)
    for row in sorted(rows,key=lambda row:row['time']):
        if row['event']=='DEFER_READ_SETUP':
            rid=external(row.get('request_id'))
            if rid is not None:opened.setdefault(rid,row['time'])
        elif row['event']=='LOAD_METADATA':
            for request in row['pending']:
                rid=external(request)
                if rid is not None:opened.setdefault(rid,row['time'])
            for request in row['submitted']:
                rid=external(request)
                if rid in opened:
                    spans[rid].append([opened.pop(rid),row['time']])
    return dict(spans),opened


def analyze():
    preserve();starts=[];request_rows=[];hashes={}
    selected=[item for item in historical_runs() if item['family']=='Historical_Main400_Natural'
        and item['label'] not in ('Agg','Deflection')]+current_runs()
    for item in selected:
        folder=item['root']/'raw'/item['name'];cfg,metric,requests=trace(folder)
        window=json.loads((folder/'window.json').read_text());left,right=window['start'],window['end']
        meta={key:value for key,value in item.items() if key not in ('root','label_scope')}
        throttle_path=folder/'D_compute_throttle.json'
        throttle=json.loads(throttle_path.read_text()) if throttle_path.exists() else None
        active,known=state_spans(throttle['rows'],left,right,'running_nonchunk') if throttle else ([],[])
        nonchunk=Timeline(active);coverage=Timeline(known)
        remote_path=folder/'D_remote_setup_deferral.json'
        remote=json.loads(remote_path.read_text()) if remote_path.exists() else None
        holds,open_requests=read_holds(remote['rows']) if remote else ({},{})
        if remote:assert remote['passed'] and not remote['pending']
        local=[]
        for row in requests:
            rid=row['request_id'];b=row['boundaries'];lo,hi=b['D_accept'],b['D_prepare']
            completed_hold=1000*Timeline(holds.get(rid,[])).area(lo,hi) if remote else None
            hold_ms=completed_hold if rid not in open_requests else None
            transfer=row['remote_read_breakdown'];ordered=transfer.get('ordered',False)
            point=transfer['receive_ACK'] if ordered else None
            duration=hi-point if point is not None else None
            value=dict(meta,request_id=rid,prompt_id=row['prompt_id'],input_tokens=row['input_tokens'],
                output_tokens=row['output_tokens'],D_accept_to_prepare_ms=1000*(hi-lo),
                READ_policy_enabled=bool(cfg.get('defer_D_remote_setup')),
                observed_READ_hold_inside_accept_prepare_ms=hold_ms,
                completed_READ_hold_observation_ms=completed_hold,
                READ_parser_request_closed=rid not in open_requests if remote else None,
                ACK_to_prepare_ms=1000*duration if duration is not None else None,
                ACK_to_prepare_nonchunk_condition_overlap_ms=1000*nonchunk.area(point,hi)
                    if throttle and point is not None else None,
                ACK_to_prepare_throttle_state_coverage_ms=1000*coverage.area(point,hi)
                    if throttle and point is not None else None,
                scope='READ hold is a directly recorded policy queue. Nonchunk overlap is exposure to the recorded throttle condition. '
                    'Neither is an exclusive GPU cost,physical idle or independent causal contribution. Do not add nested holds to the outer path.')
            request_rows.append(value);local.append(value)
        numeric=('D_accept_to_prepare_ms','observed_READ_hold_inside_accept_prepare_ms','ACK_to_prepare_ms',
            'ACK_to_prepare_nonchunk_condition_overlap_ms','ACK_to_prepare_throttle_state_coverage_ms')
        means={field:float(np.mean([row[field] for row in local if row[field] is not None]))
            if any(row[field] is not None for row in local) else None for field in numeric}
        starts.append(dict(meta,n=len(local),means=means,
            known_request_counts={field:sum(row[field] is not None for row in local) for field in numeric},
            READ_policy_status='OBSERVED' if remote else 'DISABLED_OR_NOT_INSTRUMENTED',
            throttle_status='OBSERVED' if throttle else 'DISABLED_OR_NOT_INSTRUMENTED',
            READ_parser_open_requests=open_requests,
            scheduler_state_coverage_fraction=coverage.area(left,right)/(right-left) if throttle else None))
        for path in (throttle_path,remote_path):
            if path.exists():hashes[str(path.relative_to(REPO))]=digest(path)
    value=dict(time=time.time(),passed=True,starts=starts,source_files_SHA256=hashes,evidence='TRACE_ANALYSIS_LOGICAL_POLICY_HOLDS',
        scope='Direct READ-policy waiting and source-backed throttle-condition exposure have distinct meanings. '
            'A registration-open intervention can move waiting into READ or compute admission without removing total waiting. '
            'Recorded nonchunk state is not the same as output-started request count. Missing states remain uninstrumented. '
            'No unsafe compute-throttle removal or active-decode insertion is performed.')
    write(ROOT/'analysis/logical_D_gate_overlap_analysis.json',value)
    save_csv(ROOT/'figures/logical_D_gate_per_start.csv',starts)
    save_csv(ROOT/'figures/logical_D_gate_requests.csv',request_rows)
    preserve();return value


if __name__=='__main__':
    value=analyze();print('LOGICAL_D_GATE_ANALYSIS',len(value['starts']),'starts',flush=True)
