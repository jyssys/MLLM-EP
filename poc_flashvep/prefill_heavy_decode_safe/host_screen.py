"""Screen ownership and service. Never infer physical idle from NVML."""
import json,statistics
from .common import *
from .analyze import native,summary
from .capacity_analysis import analyze as capacity


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def matched_reference(calibration,compute_slots):
    if not calibration:return None
    choices=[g for g in calibration['groups'] if g['batch']<=compute_slots
        and g['completed_repeats']>=3 and g['parity_and_work']]
    return max(choices,key=lambda g:g['P_requests_per_s']['p50']) if choices else None


def screen(name,window=None,reference=None,output_path=None):
    initialize();folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    metrics=native(name,reference)
    if not metrics['complete']:return dict(name=name,status='NOT_COMPLETE')
    extent=json.loads((folder/'window.json').read_text())
    left,right=window or (extent['start']+.2*(extent['last_arrival']-extent['start']),extent['start']+.9*(extent['last_arrival']-extent['start']))
    reference_name=cfg.get('capacity_reference');previous_reference=reference_name
    corrections=ROOT/'state/matched_graph_capacity_reference_map.json'
    if corrections.exists():reference_name=json.loads(corrections.read_text()).get('replacements',{}).get(reference_name,reference_name)
    calibration=capacity(reference_name) if reference_name else None
    unrestricted=calibration['strongest_finite_P_reference'] if calibration else None
    best=matched_reference(calibration,cfg['P_slots'])
    ref=best['P_requests_per_s']['p50'] if best else None
    from ..looped_pd_headroom_recovery import queues
    from ..looped_pd_autonomous_headroom import ready_queue_audit
    queues.ROOT=ROOT;ready_queue_audit.ROOT=ROOT;ready_queue_audit.activate=configure
    old,series=queues.analyze(name,extent,observation=(left,right))
    stats=json.loads((folder/'closed_stats.json').read_text())['metrics']
    measured=[m for m in stats if m['request_id'].startswith('r0_')]
    completed=[m['created_monotonic']+m['timings']['prefill_compute_completed'] for m in measured
        if 'prefill_compute_completed' in m['timings'] and left<=m['created_monotonic']+m['timings']['prefill_compute_completed']<=right]
    rate=len(completed)/(right-left)
    audit=ready_queue_audit.audit(cfg,dict(observation_start=left,observation_end=right,
        P_fraction_reference=rate/ref if ref else None,profile_perturbation_qualified=False,
        P_actual_kernel_busy_central=None))
    admission=[r for r in records(folder/'admission_reasons.jsonl') if left<=r['time']<=right]
    refusal=sum(any(any(s.startswith('DECODE_') for s in w['reasons']) for w in r['waiting']) for r in admission)
    credit=[]
    for sample in records(folder/'queue_observations.jsonl'):
        if not left<=sample['time']<=right:continue
        for peer in sample['worker_credits'].values():
            if peer['role']=='decode':credit.append(dict(time=sample['time'],free_blocks=peer['total_blocks']-peer['reserved_blocks'],
                reserved_blocks=peer['reserved_blocks'],reserved_slots=peer['slots'],total_blocks=peer['total_blocks']))
    needed=((cfg['length']+cfg['cap']-2)//16+1)*cfg['depth']
    owner=[]
    for p in folder.glob('memory_progress_*.jsonl'):owner.extend(r for r in records(p) if left<=r['time']<=right)
    nvml=[]
    for p in folder.glob('NVML-*.jsonl'):nvml.extend(r for r in records(p) if left<=r['time']<=right)
    growth=audit['half_growth'] if audit else None;maximum=audit['maximum'] if audit else None
    pressure=bool(growth is not None and growth>0 and maximum is not None and maximum>=2)
    decode=metrics['Decode_SLO_screen'];fraction=refusal/len(admission) if admission else None
    if fraction is not None and fraction>.5:classification='FREQUENT_D_KV_REFUSALS_CAUSAL_DOMINANCE_NOT_IDENTIFIED'
    elif not decode:classification='D_DECODE_TAIL_LIMITED'
    elif pressure:classification='HOST_P_PRESSURE_CANDIDATE_PHYSICAL_GATE_NOT_MEASURED'
    else:classification='NO_GROWING_AUDITED_P_BACKLOG_AT_THIS_POINT'
    c_cfg=json.loads((ROOT/'configs'/(reference_name+'.json')).read_text()) if calibration else {}
    value=dict(name=name,status='MEASURED_HOST_SCREEN_ONLY',model=cfg['model'],length=cfg['length'],cap=cfg['cap'],
        QPS=cfg['arrival_rate'],P_slots=cfg['P_slots'],D_slots=cfg['D_slots'],left=left,right=right,
        classification=classification,metrics=metrics,queue=audit,phase_only_queue=old,
        P_completed_in_window=len(completed),P_completion_rate=rate,strongest_finite_reference=ref,
        independent_capacity_reference=reference_name,original_capacity_reference=previous_reference,
        matched_reference_batch=best['batch'] if best else None,
        strongest_unrestricted_reference=unrestricted['P_requests_per_s']['p50'] if unrestricted else None,
        capacity_matching_scope='Only independently qualified batches within the same P compute-slot limit enter the 85-percent reference. The strongest unrestricted batch remains explicit. No threshold changes.',
        P_service_fraction=rate/ref if ref else None,P_reference_pool_matches=c_cfg.get('kv_blocks')==cfg['kv_blocks'],
        D_credit_refusal_snapshot_fraction=fraction,D_full_request_required_blocks=needed,
        credit_refusal_causal_dominance='NOT_IDENTIFIED_BY_SNAPSHOT_PREVALENCE',
        D_min_coordinator_free_blocks=min((r['free_blocks'] for r in credit),default=None),
        D_free_full_request_credit_fraction=statistics.mean(r['free_blocks']>=needed for r in credit) if credit else None,
        source_target_reserved_before_prefill=sum(m['timings'].get('d_reserved',float('inf'))<=m['timings'].get('prefill_started',-1.) for m in measured),
        measured_source_requests=len(measured),
        NVML={role:summary(r['sampled_utilization_percent'] for r in nvml if r['physical_GPU']==cfg['gpus'][i]) for i,role in enumerate(('P','D'))},
        owner_HBM={role:summary(r['allocator_allocated_bytes'] for r in owner if r['role']==role) for role in ('prefill','decode')},
        physical_P_kernel_busy='NOT_MEASURED',physical_D_idle='NOT_MEASURED',
        Phase_1_qualified=False,
        scope='Host ownership and finite service screen. NVML is not physical kernel busy or idle. Existing D target reservations differ from credit for another request. Profiling, source-cut feasibility, matched capacity, and independent confirmations remain separate gates.')
    write(Path(output_path) if output_path else folder/'host_discovery_screen.json',value);return value


def main():
    initialize();plan=json.loads((ROOT/'state/coarse_discovery_plan.json').read_text())
    rows=[screen(n) for n in plan['jobs']];write(ROOT/'analysis/coarse_host_screens.json',rows)
    for r in rows:
        print(r['name'],r.get('classification'),r.get('P_service_fraction'),
            (r.get('queue') or {}).get('half_growth'),r.get('D_credit_refusal_snapshot_fraction'),
            {k:r.get('metrics',{}).get(k) for k in ('TTFT_p95_ms','ITL_p99_ms','maximum_ITL_ms')})


if __name__=='__main__':main()
