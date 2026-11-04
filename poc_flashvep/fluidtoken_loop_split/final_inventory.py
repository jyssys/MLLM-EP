"""Inventory measured and missing results without treating completion as a pass."""
import csv
import time
from .common import *


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    configure()
    from .static_screen import metrics
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('final_inventory'):
        rows = []
        for path in sorted((ROOT / 'configs').glob('*.json')):
            cfg = read(path)
            if not isinstance(cfg, dict) or not all(key in cfg for key in ('name', 'model', 'count')):
                continue
            folder = ROOT / 'raw' / cfg['name']
            result = read(folder / 'dispatcher_result.json', {})
            row = dict(name=cfg['name'], model=cfg['model'], phase=cfg.get('phase'),
                model_revision=cfg.get('model_revision'), recurrent_depth=cfg.get('depth'),
                arm=cfg.get('arm'), input_tokens=cfg.get('length'), output_cap=cfg.get('cap'),
                offered=cfg['count'], nominal_QPS=cfg.get('arrival_rate'),
                GPU_mapping=json.dumps(cfg.get('gpus')), P_slots=cfg.get('P_slots'),
                D_slots=cfg.get('D_slots'), graph_rows=cfg.get('graph_rows'),
                D_KV_blocks=cfg.get('D_kv_blocks'), chunk_budget=cfg.get('chunk'),
                atomic_prompt=cfg.get('atomic_prompt'), wavefront=cfg.get('wavefront'),
                handoff=cfg.get('handoff_mode'), exact_state_instrumentation=cfg.get('state_probe', False),
                profile_seconds=cfg.get('kernel_activity_seconds', 0),
                evidence_routing_async=cfg.get('route_campaign_evidence_async', False),
                probe_admission_scope=cfg.get('D_probe_admission_scope', 'recurrent_phase'),
                probe_output_pacing=cfg.get('D_local_probe_output_pacing', False),
                config_SHA256=digest(path),
                status='NOT_RUN_TIME_BUDGET' if result.get('status') == 'NOT_RUN_TIME_BUDGET' else
                    'COMPLETE_NOT_AUTOMATICALLY_QUALIFIED' if result.get('complete') else
                    'FAILED_OR_INTERRUPTED' if result else 'NOT_RUN_OR_INCOMPLETE')
            offsets = cfg.get('arrival_offsets', [])[:cfg['count']]
            row['scheduled_finite_QPS'] = (len(offsets) - 1) / (offsets[-1] - offsets[0]) if len(offsets) > 1 and offsets[-1] > offsets[0] else None
            if result.get('complete') and (folder / 'requests.jsonl').exists():
                try:
                    value = metrics(cfg['name'])
                    row.update({key: val for key, val in value.items()
                                if key not in ('frozen_SLO', 'scope')})
                except (KeyError, ValueError, ZeroDivisionError) as error:
                    row['metric_unavailable_reason'] = repr(error)
            gates = read(folder / 'recovery_joint_screen.json', {}).get('gates', {})
            for key in ('P_actual_kernel_busy_central', 'P_fraction_reference',
                        'D_request_TPOT_p95_ms', 'D_token_ITL_p99_central_ms',
                        'D_min_coordinator_free_blocks', 'D_credit_refusal_sample_fraction'):
                row[key] = gates.get(key)
            device = read(folder / 'device_queue_audit.json', {})
            row['P_device_queue_half_growth'] = device.get('half_growth')
            row['P_device_queue_pass'] = device.get('queue_screen_pass')
            profile = read(folder / 'profile_analysis.json', {})
            row['profile_wall_perturbation_fraction'] = profile.get('matched_wall_difference_fraction')
            row['profile_eligible'] = profile.get('eligible_for_matched_clean')
            probes = read(folder / 'D_probe_completion.json', {})
            row['maximum_probe_admissions'] = cfg.get('D_probe_count', 0)
            row['actual_probe_admissions'] = probes.get('enqueued')
            row['completed_probes'] = probes.get('completed')
            row['all_admitted_probes_completed'] = probes.get('all_completed')
            admitted = read(folder / 'admitted_native_probe_correctness_v2.json', {})
            row['admitted_probe_exact_v2'] = admitted.get('exact')
            row['admitted_probe_full_work'] = admitted.get('full_work')
            row['original_probe_quota_verdict'] = admitted.get('original_quota_verdict')
            rows.append(row)
        keys = sorted(set().union(*(row.keys() for row in rows)))
        with (ROOT / 'complete_configuration_inventory.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: 'NOT_MEASURED' if row.get(key) is None else row[key] for key in keys} for row in rows)
        write(ROOT / 'complete_configuration_inventory.json', dict(
            generated_epoch=time.time(), configurations=rows,
            limitations='Completion is not correctness or SLO qualification. Heavy state checks and profiles are not clean timing. Finite rates are not sustainable capacity.'))
        from .recovery_table import main as table
        table()
    print(json.dumps(dict(configurations=len(rows), complete=sum(r['status'].startswith('COMPLETE') for r in rows))))


if __name__ == '__main__':
    main()
