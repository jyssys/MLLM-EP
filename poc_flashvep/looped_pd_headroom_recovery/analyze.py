"""Derive tables from actual events. Never replace missing kernels with zero."""
import collections
import csv
import statistics
from .common import *


def records(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def table(name, rows):
    rows=list(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with (ROOT/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:('NOT_MEASURED' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in row.items()})


def main():
    configure()
    from ..looped_pd_atomic_prefill import analyze as inherited, evidence
    inherited.ROOT=ROOT
    evidence.ROOT=ROOT
    # Reuse frozen old-cell SLOs for original-length comparisons only.
    old_slo=PRIOR/'configs/frozen_SLO.json'
    if old_slo.exists() and not (ROOT/'configs/frozen_SLO.json').exists():
        write(ROOT/'configs/frozen_SLO.json',json.loads(old_slo.read_text()))
    matrix=[];runs=[];memory=[];admissions=[];occupancy=[];shares=[];gates=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or 'name' not in cfg:continue
        name=cfg['name'];folder=ROOT/'raw'/name
        result=json.loads((folder/'dispatcher_result.json').read_text()) if (folder/'dispatcher_result.json').exists() else {}
        matrix.append(dict(name=name,model=cfg['model'],length=cfg['length'],cap=cfg['cap'],chunk=cfg['chunk'],
            worker=cfg['worker'],phase=cfg['phase'],requested=cfg['count'],gpus=cfg['gpus'],
            D_token_budget=cfg.get('D_token_budget',cfg['token_budget']),D_KV_blocks=cfg.get('D_kv_blocks',cfg['kv_blocks']),
            transport=cfg.get('native_transport_limits',{}),
            status='COMPLETE' if result.get('complete') else 'FAILED' if result else 'RUNNING' if folder.exists() else 'NOT_RUN',
            config_SHA256=digest(path)))
        if (folder/'requests.jsonl').exists() and not cfg.get('capacity_mode'):
            row,requests,components=inherited.serving_one(cfg)
            if row:
                row['eligible_clean']=bool(result.get('complete') and len(requests)==cfg['count'] and
                    cfg['phase'] in ('qps_pilot','aggregated_QPS_pilot','final_repeat') and
                    not any(cfg.get(k) for k in ('state_probe','verify_transfer','nsys','cuda_only_profile','P_only_profiler','observe_transport')))
                row['timing_isolation_clean']=row['eligible_clean']
                if cfg['phase'] not in ('qps_pilot','aggregated_QPS_pilot','final_repeat','bounded_stage_profiling','profile_control'):
                    row['target_QPS']=None
                runs.append(row)
            shares.extend(components)
        for p in folder.glob('memory_progress_*.jsonl'):
            for snapshot in records(p):
                total=snapshot['device_total_bytes'];pool=snapshot['KV_pool_bytes']
                row={k:v for k,v in snapshot.items() if k not in ('requests','scope')}
                row.update(name=name,model=cfg['model'],length=cfg['length'],cap=cfg['cap'],
                    active_request_count=len(snapshot['requests']))
                for k,v in list(row.items()):
                    if k.endswith('_bytes') and isinstance(v,(int,float)):
                        row[k+'_GiB']=v/1024**3
                        row[k+'_percent_HBM']=100*v/total
                # Allocator totals include weights and KV. This residual is not
                # an identified exclusive graph/workspace/staging measurement.
                if snapshot.get('weight_bytes') is not None:
                    row['unattributed_allocator_bytes']=snapshot['allocator_allocated_bytes']-pool-snapshot['weight_bytes']
                    row['unattributed_allocator_GiB']=row['unattributed_allocator_bytes']/1024**3
                row['graph_workspace_staging_separation']='NOT_MEASURED'
                memory.append(row)
                used=snapshot['KV_used_blocks']
                occupancy.append(dict(name=name,model=cfg['model'],time=snapshot['time'],role=snapshot['role'],
                    total_blocks=snapshot['KV_total_blocks'],allocated_blocks=used,free_blocks=snapshot['KV_free_blocks'],
                    allocated_bytes=used*snapshot['KV_block_bytes'],allocated_GiB=used*snapshot['KV_block_bytes']/1024**3,
                    allocated_percent_pool=100*used/snapshot['KV_total_blocks'],
                    allocated_percent_HBM=100*used*snapshot['KV_block_bytes']/total,
                    request_phases=collections.Counter(r['phase'] for r in snapshot['requests']),
                    scope='Owner allocation snapshot. Coordinator reservations are separate.'))
        for observation in records(folder/'admission_reasons.jsonl'):
            for waiting in observation['waiting']:
                reasons=waiting['reasons']
                D_blocked=any(s.startswith('DECODE_') for s in reasons)
                P_blocked=any(s.startswith('PREFILL_') for s in reasons)
                admissions.append(dict(name=name,model=cfg['model'],time=observation['time'],**waiting,
                    D_credit_blocked=D_blocked,P_credit_blocked=P_blocked,
                    classification='D_CREDIT_BLOCKED' if D_blocked else 'UNADMITTED_P_RESOURCE_WAIT' if P_blocked else 'UNADMITTED_FIT_ELIGIBLE',
                    scope='Sampled refusal. Not exclusive waiting duration or an admitted P-ready compute queue.'))
        if (folder/'recovery_gate.json').exists():
            value=json.loads((folder/'recovery_gate.json').read_text())
            gates.append(dict(name=name,model=cfg['model'],passed=value['passed'],checks=value))
    table('config_matrix.csv',matrix)
    table('serving_runs.csv',runs)
    write(ROOT/'serving_runs.json',runs)
    table('gpu_memory_accounting.csv',memory)
    table('kv_admission_events.csv',admissions)
    table('kv_occupancy_timeline.csv',occupancy)
    table('e2e_component_shares.csv',shares)
    table('correctness_gates.csv',gates)
    aggregate=[]
    for key in sorted({(r['name'],r['component']) for r in shares}):
        values=[r for r in shares if (r['name'],r['component'])==key]
        aggregate.append(dict(name=key[0],component=key[1],requests=len(values),
            share_percent=100*sum(v['component_ms'] for v in values)/sum(v['E2E_ms'] for v in values),
            component_ms_mean=statistics.mean(v['component_ms'] for v in values),
            evidence='MEASURED_HOST_EXCLUSIVE_INTERVALS_NOT_GPU_OR_DMA_SHARES'))
    table('e2e_component_shares_summary.csv',aggregate)
    write(ROOT/'state/analysis.json',dict(time=time.time(),configs=len(matrix),serving_rows=len(runs),memory_snapshots=len(memory)))
    print(json.dumps(dict(configs=len(matrix),serving_rows=len(runs),memory_snapshots=len(memory),gates=gates)))


if __name__=='__main__':
    main()
