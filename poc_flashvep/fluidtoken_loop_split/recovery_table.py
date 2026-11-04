"""Summarize all recovery attempts, including failed and rejected captures."""
import csv
from .common import *


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    rows = []
    for path in sorted((ROOT / 'configs').glob('recovery_huginn_*.json')):
        cfg = read(path)
        folder = ROOT / 'raw' / cfg['name']
        dispatch = read(folder / 'dispatcher_result.json', {})
        profile = read(folder / 'profile_analysis.json', {})
        screen = read(folder / 'recovery_joint_screen.json', {})
        gates = screen.get('gates', {})
        queue = screen.get('queue') or {}
        device_queue = read(folder / 'device_queue_audit.json', {})
        activity = read(folder / 'activity_capture.json', {})
        events = []
        if (folder / 'capture_events.jsonl').exists():
            events = [json.loads(line) for line in (folder / 'capture_events.jsonl').read_text().splitlines()]
        start = next((r for r in events if r['event'] == 'START'), {})
        stop = next((r for r in events if r['event'] in ('STOP', 'STOP_CLOSE')), {})
        row = dict(name=cfg['name'], input=cfg['length'], output=cfg['cap'], count=cfg['count'],
            QPS=cfg.get('arrival_rate'), GPUs=str(cfg['gpus']),
            progress_thread=cfg.get('nixl_enable_progress_thread', True),
            early_CUPTI_buffers=cfg.get('cupti_preallocated_buffers'),
            probe_max_active_decode=cfg.get('D_probe_max_active_decode'),
            complete=dispatch.get('complete', False),
            status=('COMPLETE' if dispatch.get('complete') else 'FAILED' if dispatch
                    else 'RUNNING' if folder.exists() else 'NOT_RUN'),
            watchdog=read(folder / 'watchdog.json', {}).get('reason'),
            captured_seconds=profile.get('window_seconds'),
            P_kernel_union=profile.get('kernel_busy_fraction'),
            paired_wall_change=profile.get('matched_wall_difference_fraction'),
            eligible_profile=profile.get('eligible_for_matched_clean'),
            start_call_seconds=start.get('profiler_start_call_seconds'),
            stop_call_seconds=stop['time'] - stop['stop_begin'] if stop else None,
            P_service_proximity=gates.get('P_fraction_reference'),
            P_queue_half_growth=queue.get('half_growth'),
            P_gate=queue.get('audited_P_gate'),
            P_device_qualified_queue_growth=device_queue.get('half_growth'),
            P_device_qualified_queue_pass=device_queue.get('queue_screen_pass'),
            activity_dropped=activity.get('dropped'),
            activity_invalid=activity.get('invalid'),
            D_TPOT_p95_ms=gates.get('D_request_TPOT_p95_ms'),
            D_ITL_p99_ms=gates.get('D_token_ITL_p99_central_ms'),
            D_ITL_whole_run_p99_ms=gates.get('D_token_ITL_p99_whole_run_ms'),
            D_ITL_whole_run_max_ms=gates.get('D_token_ITL_max_whole_run_ms'),
            D_free_blocks_min=gates.get('D_min_coordinator_free_blocks'),
            D_SLO_gate=gates.get('D_SLO_gate'),
            D_credit_gate=gates.get('D_credit_gate'),
            completed_probes_in_capture=screen.get('central_completed_probes'),
            evidence='MEASURED_WITH_EXPLICIT_MISSING_FIELDS')
        rows.append({key: 'NOT_MEASURED' if value is None else value for key, value in row.items()})
    target = ROOT / 'recovery_attempts.csv'
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), path=str(target))))


if __name__ == '__main__':
    main()
