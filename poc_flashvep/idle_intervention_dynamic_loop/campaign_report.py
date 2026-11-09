"""Derive reproducible tables without treating diagnostic timings as serving results."""
import csv
import re
from collections import Counter,defaultdict
from .common import *
from .analyze import metrics,percentile
from .qualification import qualify


def read(path):
    return json.loads(path.read_text())


def event_records(folder,begin):
    return [r for path in folder.glob('intervention_*.json')
        for r in read(path).get('records',[]) if r['time']>=begin]


def collect():
    configure();rows=[];decisions=[];quanta=[];transfers=[];memory=[];migrations=[];graphs=[]
    queued=set(read(ROOT/'state/queue.json'))
    environment=read(ROOT/'state/environment.json')
    totals=[int(value)*2**20 for value in re.findall(r',\s*(\d+) MiB,',environment['GPU_query'])]
    assert totals and len(set(totals))==1,'Memory percentages require an explicit homogeneous NVML capacity'
    physical_total=totals[0]
    for path in sorted((ROOT/'configs').glob('*.json')):
        if path.name=='frozen_SLO.json':continue
        cfg=read(path);name=path.stem;folder=ROOT/'raw'/name
        receipt=folder/'dispatcher_result.json'
        dispatch=read(receipt) if receipt.exists() else {}
        status=(dispatch.get('status') or ('COMPLETE' if dispatch.get('complete') else 'INCOMPLETE_RECEIPT')) if receipt.exists() else (
            'EVIDENCE_DIRECTORY_WITHOUT_RECEIPT' if folder.exists() else
            'DECLARED_QUEUE_WITHOUT_DISPATCH' if name in queued else 'NOT_ENQUEUED')
        if receipt.exists() and not dispatch.get('complete') and dispatch.get('returncode') not in (None,0):
            if status=='RUNNING':status='TERMINATED_NONFINAL_STATUS'
            elif status=='INCOMPLETE_RECEIPT':status='FAILED_NO_FINAL_STATUS'
        row=dict(name=name,model=cfg['model'],method=cfg.get('intervention_method'),count=cfg['count'],
            load=cfg.get('load'),arrival_seed=cfg.get('arrival_seed'),phase=cfg.get('phase'),
            D_kv_blocks=cfg.get('D_kv_blocks'),graph_budget=cfg.get('deep_graph_budget'),
            dispatch_status=status,recorded_status=dispatch.get('status'),returncode=dispatch.get('returncode'),in_declared_queue=name in queued,
            complete=bool(dispatch.get('complete')),elapsed_seconds=dispatch.get('elapsed_seconds'),
            evidence_class='STATE_DIAGNOSTIC' if cfg.get('state_probe') else
                'DEVICE_EVENT_DIAGNOSTIC' if cfg.get('observe_device_execution') or cfg.get('calibration_serial') else
                'INSTRUMENTED' if cfg.get('observe_service_timing') or cfg.get('sparse_device_trace') else 'MARKER_FREE',
            independent_prompt_count=cfg.get('data_scope',{}).get('distinct_prompt_ids'),
            kernel_union='NOT_MEASURED',physical_idle='NOT_MEASURED',measured_P_compute_saving='NOT_MEASURED')
        if not row['complete']:
            rows.append(row);continue
        try:
            reference=None
            if cfg.get('state_probe') and not cfg.get('calibration_serial'):
                reference=f'n32_{cfg["model"]}_native_full_state32' if cfg['count']==32 else f'a_{cfg["model"]}_native_state8'
            if cfg.get('comparison_reference'):reference=cfg['comparison_reference']
            q=qualify(name,reference);m=metrics(name)
            row.update(m)
            row.update(qualification=q['passed'],full_work=q['work']['passed'],
                lifetime=q['lifetime']['passed'],adaptive_diversity=q['offload']['adaptive_diversity'],
                output_exact=q.get('output',{}).get('passed','NOT_MEASURED'),
                state_exact=q.get('state',{}).get('exact','NOT_MEASURED'),
                handoff_exact=q.get('async_handoff',q.get('handoff',{})).get('passed','NOT_MEASURED'))
            before=read(folder/'before_measured.json');after=read(folder/'after_measured.json')
            pkey=next(key for key in after if '-prefill-' in key)
            dkey=next(key for key in after if '-decode-' in key)
            row['actual_P_prefill_rows']=after[pkey]['issued_loop_rows']-before[pkey]['issued_loop_rows']
            row['actual_P_ModelRunner_submissions']=after[pkey]['issued_submissions']-before[pkey]['issued_submissions']
            row['actual_D_ModelRunner_submissions']=after[dkey]['issued_submissions']-before[dkey]['issued_submissions']
            row['P_active_rows_saved']=cfg['count']*cfg['length']*cfg['depth']-row['actual_P_prefill_rows']
            row['active_row_saving_scope']='Measured active recurrent rows relative to no-reuse full Prefill. Not padded GEMM rows, FLOPs, or critical-path time.'
            begin=read(folder/'measurement_begin.json')['time'];events=event_records(folder,begin)
            committed={(r['tid'],r['epoch']) for r in events if r['event']=='SOURCE_COMMIT'}
            for item in events:
                if item['event']=='ADMISSION':decisions.append(dict(name=name,model=cfg['model'],method=cfg.get('intervention_method'),evidence_class=row['evidence_class'],
                    source_committed=(item['tid'],item['epoch']) in committed,**item))
                if item['event'] in ('D_QUANTUM_SUBMITTED','D_QUANTUM_HOST_COMPLETION'):
                    quanta.append(dict(name=name,model=cfg['model'],evidence_class=row['evidence_class'],**item))
            row['rejection_reasons']=dict(Counter(r.get('rejection_reason','NOT_RECORDED_IN_EARLY_ADAPTER')
                for r in events if r['event']=='ADMISSION' and not r['accepted']))
            row['source_armed']=sum(r['event']=='SOURCE_ARM_DECISION' and r['armed'] for r in events)
            row['source_arm_decisions']=sum(r['event']=='SOURCE_ARM_DECISION' for r in events)
            row['safety_latches']=sum(r['event']=='EXTRA_WORK_SUSPENDED' for r in events)
            row['source_lease_timeouts']=sum(r['event']=='SOURCE_LEASE_TIMEOUT' for r in events)
            row['execution_quanta']=sum(r['event']=='D_QUANTUM_SUBMITTED' for r in events)
            payload=[read(p) for p in folder.glob('wire_payload_*.json')]
            row['measured_KV_payload_bytes']=sum(p.get('counts',{}).get('measured',{}).get('KV_bytes',0) for p in payload)
            row['measured_hidden_payload_bytes']=sum(p.get('counts',{}).get('measured',{}).get('hidden_bytes',0) for p in payload)
            life=folder/'lifetime_events.jsonl';by_tid=defaultdict(dict)
            if life.exists():
                for line in life.read_text().splitlines():
                    r=json.loads(line)
                    if r['time']>=begin:by_tid[r['tid']][r['event']]=r['time']
            for tid,items in by_tid.items():
                if 'COORDINATOR_commit' in items and 'COORDINATOR_received' in items:
                    transfers.append(dict(name=name,model=cfg['model'],tid=tid,
                        commit_to_received_ms=1000*(items['COORDINATOR_received']-items['COORDINATOR_commit']),
                        commit_to_source_release_ms=1000*(items['COORDINATOR_released']-items['COORDINATOR_commit'])
                            if 'COORDINATOR_released' in items else None,
                        scope='Host message receipts. Not exclusive DMA time.'))
            decode=[r for r in events if r['event']=='SUPPLY_SAMPLE' and r.get('role')=='decode']
            row['D_min_free_blocks']=min((r['KV_free_blocks'] for r in decode),default=None)
            row['D_sampled_budget_p50_ms']=percentile([r['predicted_budget_ms'] for r in decode],.5)
            row['D_sampled_budget_p95_ms']=percentile([r['predicted_budget_ms'] for r in decode],.95)
            for file in folder.glob('deep_graph_cache_*.json'):
                graph=read(file);counts=graph['counts'].get('measured',{});selects=counts.get('selects',0)
                graphs.append(dict(name=name,model=cfg['model'],role=graph['role'],budget=graph['budget'],
                    selects=selects,existing_hits=counts.get('existing_hits',0),fallbacks=counts.get('fallbacks',0),
                    hit_ratio=counts.get('existing_hits',0)/selects if selects else 'NOT_APPLICABLE_NO_PREFILL_CALL',
                    measured_captures=counts.get('captures',0),measured_prepare_host_seconds=counts.get('prepare_host_s',0.),
                    scope=graph['scope']))
            expected_seeds=set(range(cfg['seed'],cfg['seed']+cfg['count']))
            for file in folder.glob('migration_*.jsonl'):
                for line in file.read_text().splitlines():
                    value=json.loads(line)
                    if value.get('seed') in expected_seeds and value.get('time',0)>=begin:
                        migrations.append(dict(name=name,model=cfg['model'],**value))
            for file in folder.glob('memory_inventory_close_*.json'):
                value=read(file);role=value['role']
                progress=folder/('memory_progress_'+file.stem.rsplit('_',1)[1]+'.jsonl')
                total=read_first_total(progress)
                for key,amount in value.items():
                    if key.endswith('_bytes') and isinstance(amount,(int,float)):
                        memory.append(dict(name=name,model=cfg['model'],role=role,component=key,bytes=amount,
                            GiB=amount/2**30,HBM_percent=100*amount/physical_total,
                            HBM_total_bytes=physical_total,CUDA_accessible_total_bytes=total,
                            CUDA_accessible_percent=100*amount/total if total else None,scope=value['scope']))
        except Exception as error:row['analysis_error']=repr(error)
        rows.append(row)
    write(ROOT/'serving_runs.json',rows);write_csv(ROOT/'serving_runs.csv',rows)
    write(ROOT/'actual_decisions.json',decisions);write_csv(ROOT/'actual_decisions.csv',decisions)
    write(ROOT/'actual_quanta.json',quanta);write_csv(ROOT/'handoff_host_receipts.csv',transfers)
    write_csv(ROOT/'gpu_memory_inventory.csv',memory);write_csv(ROOT/'actual_migration_payloads.csv',migrations)
    write_csv(ROOT/'graph_cache_observations.csv',graphs)
    write_csv(ROOT/'failure_inventory.csv',[r for r in rows if not r['complete'] or r.get('qualification') is False])
    return rows


def read_first_total(path):
    if not path.exists():return None
    with path.open() as handle:
        for line in handle:
            value=json.loads(line)
            if value.get('device_total_bytes'):return value['device_total_bytes']
    return None


def write_csv(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value
                for key,value in row.items()})


if __name__=='__main__':
    rows=collect()
    print(json.dumps(dict(configurations=len(rows),complete=sum(r['complete'] for r in rows),
        actual_diverse=[r['name'] for r in rows if r.get('adaptive_diversity')],
        analysis_errors=[r for r in rows if r.get('analysis_error')]),indent=2))
