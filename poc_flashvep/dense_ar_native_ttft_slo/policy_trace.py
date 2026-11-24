"""Audit measured registration holds,source credits and realized batch geometry."""
import bisect
import collections
import json
import time
import numpy as np
from .common import ROOT,REPO,initialize,preserve,write,digest
from .registry import analysis_runs
from .reproduction import records
from .request_trace import external,read_json_rows
from .failure_analysis import save_csv


def weighted_states(rows,left,right,fields):
    if not rows:return dict(status='NOT_INSTRUMENTED')
    rows=sorted(rows,key=lambda r:r['time']);times=[r['time'] for r in rows]
    index=bisect.bisect_right(times,left)-1
    result={key:0. for key in fields};maximum={key:None for key in fields}
    minimum={key:None for key in fields};covered={key:0. for key in fields};zero={key:0. for key in fields}
    cursor=left;current=rows[index] if index>=0 else {}
    for row in rows[index+1:]+[dict(time=right)]:
        end=min(right,row['time'])
        if end>cursor:
            for key in fields:
                if key not in current:continue
                value=current[key];duration=end-cursor
                result[key]+=duration*value;covered[key]+=duration;zero[key]+=duration*(value==0)
                maximum[key]=value if maximum[key] is None else max(maximum[key],value)
                minimum[key]=value if minimum[key] is None else min(minimum[key],value)
        cursor=end
        if cursor>=right:break
        current=row
    complete={key:abs(covered[key]-(right-left))<=1e-6 for key in fields}
    return dict(status='MEASURED_HOST_SCHEDULER_STATE',
        time_weighted_means={key:value/(right-left) if complete[key] else None for key,value in result.items()},
        maxima=maximum,minima=minimum,
        zero_fractions={key:zero[key]/(right-left) if complete[key] else None for key in fields},
        observed_duration_time_weighted_means={key:result[key]/covered[key] if covered[key] else None for key in fields},
        observed_duration_zero_fractions={key:zero[key]/covered[key] if covered[key] else None for key in fields},
        field_coverage_fraction={key:covered[key]/(right-left) for key in fields},
        field_status={key:'OBSERVED_FULL_WINDOW' if complete[key] else 'PARTIAL_OR_NOT_INSTRUMENTED' for key in fields},
        state_missing_at_window_start=index<0,
        scope='Piecewise-constant recorded scheduler states over the measured cohort window. '
            'Missing fields remain None,not zero. Minima refer to observed host snapshots,not continuous GPU state. '
            'This is not GPU activity or decode-safe capacity.')


def analyze():
    initialize();preserve();starts=[];requests=[];source_files={};first_sources=[]
    for item in analysis_runs():
        if item['label']=='Deflection':continue
        folder=item['root']/'raw'/item['name'];cfg=json.loads((folder/'configuration.json').read_text())
        output=records(folder/'requests.jsonl');window=json.loads((folder/'window.json').read_text())
        meta={key:value for key,value in item.items() if key!='root'}
        meta.update(seed=cfg['seed'],GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'])
        observed_sources=collections.Counter(r['token_events'][0].get('source','MISSING') for r in output)
        first_sources.append(dict(meta,protocol=cfg.get('protocol'),request_count=len(output),first_sources=dict(observed_sources)))
        if item['label']=='Agg':continue
        geometry={}
        for role in ('prefill','decode'):
            batches=collections.defaultdict(list)
            for row in read_json_rows(folder/f'actual_queries_{role}.json'):batches[row['batch_id']].append(row)
            histogram=collections.Counter((sum(r['API_prefilling'] for r in values),sum(not r['API_prefilling'] for r in values)) for values in batches.values())
            pure=[d for (p,d),count in histogram.items() for _ in range(count) if p==0 and d>0]
            prefill=[p for (p,d),count in histogram.items() for _ in range(count) if p>0]
            geometry[role]=dict(steps=len(batches),histogram=[dict(prefill=p,decode=d,steps=n) for (p,d),n in sorted(histogram.items())],
                pure_decode_step_weighted_mean=float(np.mean(pure)) if pure else None,pure_decode_maximum=max(pure,default=0),
                prefill_request_step_weighted_mean=float(np.mean(prefill)) if prefill else None)
        guard_path=folder/'D_request_registration_deferral.json';events=[];releases={}
        if guard_path.exists():
            guard=json.loads(guard_path.read_text());assert guard['passed'] and not guard['pending'] and not guard['ready']
            for row in guard['rows']:
                rid=external(row.get('request_id'))
                if rid is None:continue
                if row['event']=='RELEASE_COMMAND':releases[rid]=row['time']
                elif row['event']=='RECEIVE_COMMAND' and row['kind']=='add':events.append(row)
        accepted={(r['request_id'],r['role']):r['worker_accepted'] for r in records(folder/'admissions.jsonl')}
        submitted={(r['request_id'],r['role']):r['time'] for r in records(folder/'request_submissions.jsonl')}
        observed=[]
        for event in events:
            rid=external(event['request_id']);release=releases[rid] if event['deferred'] else event['time']
            parts=dict(D_submit_to_registration_receive_ms=1000*(event['time']-submitted[rid,'decode']),
                registration_guard_hold_ms=1000*(release-event['time']),
                registration_release_to_D_accept_ms=1000*(accepted[rid,'decode']-release))
            assert all(value>=-1e-3 for value in parts.values()),(item['name'],rid,parts)
            assert abs(sum(parts.values())-1000*(accepted[rid,'decode']-submitted[rid,'decode']))<1e-3
            row=dict(meta,request_id=rid,deferred=event['deferred'],decode_active_at_receive=event['decode_active'],**parts)
            requests.append(row);observed.append(row)
        from ..dense_ar_p_critical_path.critical_analysis import credits
        credit=credits(folder)
        queues={role:weighted_states(read_json_rows(folder/f'scheduler_queues_{role}.json'),window['start'],window['end'],
            ('waiting_total','running','remote_KV_waiting','active_decode','decode_ready_waiting','free_KV_blocks')) for role in ('prefill','decode')}
        starts.append(dict(meta,source_credit=cfg.get('continuous_source_credit_limit',0),D_cap=cfg.get('D_running_admission_cap'),
            defer_D_request_registration=cfg.get('defer_D_request_registration',False),defer_D_remote_setup=cfg.get('defer_D_remote_setup',False),
            geometry=geometry,registration=dict(status='INSTRUMENTED' if guard_path.exists() else 'POLICY_NOT_USED_OR_NOT_INSTRUMENTED',
                requests=len(observed),deferred_requests=sum(row['deferred'] for row in observed),
                means_ms={field:float(np.mean([row[field] for row in observed])) for field in (
                    'D_submit_to_registration_receive_ms','registration_guard_hold_ms','registration_release_to_D_accept_ms')} if observed else {}),
            source_credit_state={key:value for key,value in credit.items() if key not in (
                'credit_spans','source_window_full_intervals','per_request_arrival_to_dispatch_full_window_overlap_ms',
                'per_request_blocked_ms','per_request_observed_credit_eligible_pending_ms')},queues=queues))
        for path in (guard_path,folder/'credit_events.jsonl',folder/'scheduler_queues_prefill.json',folder/'scheduler_queues_decode.json'):
            if path.exists():source_files[str(path.relative_to(REPO))]=digest(path)
    value=dict(time=time.time(),passed=True,starts=starts,first_token_source_audit=first_sources,source_files_SHA256=source_files,
        evidence='TRACE_ANALYSIS',scope='Registration hold is directly observed while an output-started request remains running. '
            'Its presence does not prove physical GPU idle. Source-window overlap is not exclusive causal credit delay. '
            'Actual pure-decode batch is separate from D Cap and Scheduler S. '
            'The first-token source audit distinguishes official proxy from P-first-token reuse.')
    write(ROOT/'analysis/policy_credit_batch_audit.json',value)
    save_csv(ROOT/'figures/policy_credit_batch_per_start.csv',starts)
    save_csv(ROOT/'figures/D_registration_request_breakdown.csv',requests)
    save_csv(ROOT/'figures/first_token_source_audit.csv',first_sources)
    print('POLICY_CREDIT_BATCH_AUDIT_PASS',len(starts),'starts',len(requests),'registration paths',flush=True)
    preserve();return value


if __name__=='__main__':analyze()
