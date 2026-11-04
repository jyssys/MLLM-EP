"""Separate accepted work, completed requests, and measured payload accounting."""
import csv
from .common import *


def records(folder, pattern):
    return [json.loads(line) for path in folder.glob(pattern)
            for line in path.read_text().splitlines() if line]


def main():
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_atomic_prefill.evidence import full_work
    with quiet_analysis('method_work_and_payload_accounting'):
        rows = []
        for path in sorted((ROOT / 'configs').glob('*.json')):
            cfg = json.loads(path.read_text())
            if not cfg.get('split_evaluation'):
                continue
            folder = ROOT / 'raw' / cfg['name']
            receipt = folder / 'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
                continue
            marker = folder / 'measurement_begin.json'
            if not marker.exists():
                continue
            begin = json.loads(marker.read_text())['time']
            seeds = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
            select = lambda record: record['seed'] in seeds and record['time'] >= begin
            accepted = [r for r in records(folder, 'accepted_partitions_*.jsonl') if select(r)]
            migrations = [r for r in records(folder, 'migration_*.jsonl') if select(r)]
            requests = [r for r in records(folder, 'requests.jsonl') if not r.get('warmup')]
            closed = json.loads((folder / 'closed_stats.json').read_text())
            full = full_work(cfg['name'])
            native = cfg['arm'] == 'PD_G1'
            accounted = native or (len(accepted) == len(requests) == cfg['count']
                                   and len({r['seed'] for r in accepted}) == cfg['count'])
            assigned = sum(r['assigned_D_recurrent_token_rows'] for r in accepted)
            denominator = sum(r['input_tokens'] for r in requests) * cfg['depth']
            row = dict(name=cfg['name'], model=cfg['model'], arm=cfg['arm'],
                       heavy_state_instrumentation=cfg.get('state_probe', False),
                       count=cfg['count'], completed=len(requests), accounting_complete=accounted,
                       full_recurrent_work=full['passed'],
                       accepted_partitions=len(accepted),
                       partial_serving_requests=sum(r['partial'] for r in accepted),
                       assigned_D_recurrent_token_rows=assigned if accounted else 'NOT_MEASURED',
                       background_Prefill_recurrent_token_rows=denominator,
                       assigned_D_row_fraction=assigned / denominator if accounted and denominator else 'NOT_MEASURED',
                       queued_snapshot_KV_bytes=sum(r['KV_bytes'] for r in migrations) if migrations else 'NOT_MEASURED',
                       queued_snapshot_hidden_bytes=sum(r['hidden_bytes'] for r in migrations) if migrations else 'NOT_MEASURED',
                       queued_snapshot_fragments=sum(r['fragments'] for r in migrations) if migrations else 'NOT_MEASURED',
                       queued_snapshot_descriptors=sum(r['descriptors'] for r in migrations) if migrations else 'NOT_MEASURED',
                       FLOPs='NOT_MEASURED', GPU_kernel_invocations='NOT_MEASURED',
                       scope='Accepted boundaries plus completed requests and separate full-work gate. Heavy state runs are not clean latency. Rows are not FLOPs. Payload counts are queued native descriptors, not DMA durations.')
            for role in ('prefill', 'decode'):
                workers = [value for key, value in closed['worker_metrics'].items() if role in key]
                row[role + '_process_peak_torch_allocated_GiB'] = max(
                    (w['peak_allocated_bytes'] for w in workers), default=0) / 1024**3
                row[role + '_gross_model_submissions_including_warmup'] = sum(
                    w.get('issued_submissions', 0) for w in workers)
            rows.append(row)
        write(ROOT / 'method_work_and_payload_accounting.json', dict(
            results=rows, scope='Structural work and measured descriptor accounting. No unmeasured FLOP or exclusive GPU-time claim.'))
        if rows:
            with (ROOT / 'method_work_and_payload_accounting.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(dict(runs=len(rows))))


if __name__ == '__main__':
    main()
