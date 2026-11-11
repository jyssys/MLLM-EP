"""Build an explicit result ledger. Missing execution is not a zero result."""
import csv
from .common import *


TRACE_FLAGS = ('sparse_device_trace', 'host_cost_trace', 'local_nsys',
    'full_kernel_trace', 'observe_device_execution', 'profile_python',
    'async_prompt_state_probe', 'post_decode_state')


def collect():
    initialize()
    rows = []
    isolation_path = ROOT/'state/primaryconfirmv2_isolation_replacements.json'
    isolation = json.loads(isolation_path.read_text())['replacements'] if isolation_path.exists() else {}
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if not isinstance(cfg, dict) or 'name' not in cfg:
            continue
        folder = ROOT/'raw'/cfg['name']
        receipt = folder/'dispatcher_result.json'
        status = json.loads(receipt.read_text()) if receipt.exists() else {}
        preflight = sorted((ROOT/'state/preflight_failures').glob(cfg['name']+'_*.json'))
        qualification = folder/'qualification_with_frozen_reference.json'
        gate = json.loads(qualification.read_text()) if qualification.exists() else {}
        quantum_path = ROOT/'audit'/'quantum'/(cfg['name']+'.json')
        quantum = json.loads(quantum_path.read_text()) if quantum_path.exists() else {}
        row = dict(name=cfg['name'], phase=cfg.get('phase'),
            count=cfg.get('count'), length=cfg.get('length'), cap=cfg.get('cap'),
            arrival_seed=cfg.get('arrival_seed'), config_sha256=digest(path),
            execution_status='COMPLETE' if status.get('complete') else
                ('INCOMPLETE' if folder.exists() else
                 'NOT_RUN_REQUIRED_QUALIFICATION' if preflight else 'NOT_RUN'),
            dispatcher_returncode=status.get('returncode'),
            dispatcher_watchdog=status.get('watchdog'),
            qualification_gate=gate.get('campaign_gate_passed', gate.get('passed')),
            requested_D_prompt_loop_cap=cfg.get('D_prefill_loop_cap'),
            observed_D_prompt_quantum_gate=quantum.get('passed'),
            state_probe=bool(cfg.get('state_probe')),
            diagnostic=any(cfg.get(key) for key in TRACE_FLAGS),
            decode_group=cfg.get('decode_loop_grouping', 1),
            fixed_submit=bool(cfg.get('fixed_depth_submit')),
            independent_prompt=bool(cfg.get('independent_prompt_stream')),
            green_SM_count=cfg.get('prompt_green_sms'),
            deferred_coda=bool(cfg.get('deferred_coda_collection')),
            cohort_retirement=bool(cfg.get('preserve_decode_cohort')),
            selected_cut=cfg.get('native_partial_cut'),
            frontier=cfg.get('native_frontier'),
            wire_config_scope='REQUESTED_ONLY')
        limits = cfg.get('native_transport_limits', {})
        resolved = list(folder.glob('resolved_native_PD_*.json'))
        if not resolved:
            resolved = list(folder.glob('effective_PD_config_*.json'))
        if resolved:
            value = json.loads(resolved[0].read_text())
            # The older receipt records the dataclass directly.
            limits = value.get('resolved', value)
            row['wire_config_scope'] = 'ACTUAL_CONSTRUCTOR_RECEIPT'
        for field in ('max_inflight_bytes', 'max_draining_requests',
                'transfer_chunk_bytes', 'max_transfer_descriptors'):
            row['wire_'+field] = limits.get(field, 'NOT_RECORDED')
        if status.get('complete') and cfg.get('model') == 'huginn':
            row.update(analysis().metrics(cfg['name']))
        row['performance_eligible'] = (status.get('complete', False)
            and not row['state_probe'] and not row['diagnostic']
            and cfg.get('model') == 'huginn')
        exit_path = folder/'process_exit_audit.json'
        row['worker_exit_integrity'] = (json.loads(exit_path.read_text())['status']
            if exit_path.exists() else 'NOT_AUDITED')
        row['selection_observation_available'] = row['performance_eligible']
        row['performance_eligible'] &= row['worker_exit_integrity'] != 'FAIL'
        row['isolation_status'] = ('POSSIBLE_CPU_IO_OVERLAP' if cfg['name'] in isolation else
            'DECLARED_CLEAN_REPLACEMENT' if cfg['name'] in isolation.values() else 'NO_DECLARED_OVERLAP')
        if cfg['name'] in isolation:
            row['performance_eligible'] = False
        row['confirmation_exit_qualified'] = row['worker_exit_integrity'] == 'VERIFIED_ZERO_EXIT'
        rows.append(row)
    return rows


def main():
    rows = collect()
    destination = ROOT/'audit'
    destination.mkdir(exist_ok=True)
    write(destination/'experiment_inventory.json', rows)
    fields = sorted({key for row in rows for key in row})
    with (destination/'experiment_inventory.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list)) else value for key, value in row.items()})
    print(json.dumps(dict(configurations=len(rows),
        complete=sum(row['execution_status'] == 'COMPLETE' for row in rows),
        performance_eligible=sum(row['performance_eligible'] for row in rows))))


if __name__ == '__main__':
    main()
