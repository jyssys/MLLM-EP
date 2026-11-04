"""Derive host-observed commit, ACK, and retirement intervals separately."""
import csv
from .common import *


def observation(events, owners, tid):
    selected = [row for row in events if row.get('tid') == tid]
    source = [row for row in owners if row.get('tid') == tid and row['role'] == 'prefill']

    def first(rows, event, role=None):
        return min((row['time'] for row in rows if row['event'] == event
                    and (role is None or row['role'] == role)), default=None)

    values = dict(commit=first(selected, 'COORDINATOR_commit', 'prefill'),
        received=first(selected, 'COORDINATOR_received', 'decode'),
        P_start=first(selected, 'COORDINATOR_prefill_started', 'prefill'),
        P_complete=first(selected, 'COORDINATOR_prefill_complete', 'prefill'),
        P_release=first(selected, 'COORDINATOR_released', 'prefill'),
        ACK=first(source, 'NATIVE_ACK_RECEIVED'),
        forget=first(source, 'NATIVE_CONNECTOR_FORGET'))
    pairs = dict(commit_receipt_to_D_received_receipt_ms=('commit', 'received'),
        D_received_receipt_to_source_ACK_observation_ms=('received', 'ACK'),
        source_ACK_to_retirement_ms=('ACK', 'forget'),
        P_completion_receipt_to_source_retirement_ms=('P_complete', 'forget'),
        P_start_receipt_to_source_retirement_ms=('P_start', 'forget'),
        source_retirement_to_coordinator_credit_release_ms=('forget', 'P_release'))
    return {field: (1000 * (values[end] - values[start])
                    if values[start] is not None and values[end] is not None else None)
            for field, (start, end) in pairs.items()}


def main():
    initialize()
    from ..looped_pd_headroom_recovery.gates import percentile
    results, flat = [], []
    for path in sorted((ROOT / 'raw').glob('*/dispatcher_result.json')):
        if not json.loads(path.read_text()).get('complete'):
            continue
        folder = path.parent
        cfg = json.loads((ROOT / 'configs' / (folder.name + '.json')).read_text())
        if (cfg.get('state_probe') or cfg.get('verify_transfer') or cfg.get('calibration_serial')
                or cfg.get('worker') != 'rlt_worker'):
            continue
        events_path = folder / 'lifetime_events.jsonl'
        if not events_path.exists():
            continue
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        owners = [json.loads(line) for p in folder.glob('lifetime_owner_*.jsonl')
                  for line in p.read_text().splitlines()]
        closed = json.loads((folder / 'closed_stats.json').read_text())['metrics']
        mode = ('CUPTI_CAPTURE' if cfg.get('kernel_activity_seconds') else
                'CUDA_EVENT_MARKED' if cfg.get('observe_service_timing') else 'MARKER_FREE')
        rows = [dict(name=folder.name, model=cfg['model'], observation_mode=mode,
            request_id=row['request_id'],
            transfer_id=row['transfer_id'], **observation(events, owners, row['transfer_id']))
            for row in closed if row['request_id'].startswith('r0_')]
        assert len(rows) == cfg['count']
        summary = dict(name=folder.name, model=cfg['model'], offered=cfg['count'], requests=rows)
        fields = [field for field in rows[0] if field.endswith('_ms')]
        for field in fields:
            values = [row[field] for row in rows if row[field] is not None]
            summary[field + '_measured_count'] = len(values)
            summary[field + '_negative_observations'] = sum(value < 0 for value in values)
            for q in (.5, .95, .99):
                summary[field + '_p' + str(int(q * 100))] = percentile(values, q) if values else None
        summary['scope'] = ('Host observations after message handling and native owner callbacks. '
            'Intervals can overlap compute and transfer. They are not GPU DMA duration or exclusive E2E components. '
            'Post-handler observations across processes can reverse nearby timestamps. Negative values remain visible. '
            'The separate lifetime gate checks source ACK before retirement and complete final release.')
        write_derived(folder / 'host_transport_lifetime_timings.json', summary)
        results.append(summary)
        flat.extend(rows)
    write_derived(ROOT / 'state/host_transport_lifetime_timings.json', dict(results=results))
    if flat:
        with (ROOT / 'host_transport_lifetime_timings.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat for key in row}))
            writer.writeheader()
            writer.writerows(flat)


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('host_commit_ACK_and_retirement_intervals'):
        main()
