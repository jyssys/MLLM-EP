"""Summarize additive host boundaries without calling them GPU compute."""
import json
import numpy as np
from .common import ROOT, write
from .figures import start_interval


def main():
    starts = json.loads((ROOT / 'analysis/Main_full_analysis.json').read_text())['per_start']
    fields = ('P_queue_wait_ms', 'P_first_prepare_to_ready_ms', 'Ready_to_D_prepare_ms',
              'D_prepare_to_first_delivery_ms', 'TTFT_ms', 'Ready_to_D_accepted_ms',
              'Ready_to_NIXL_receive_notification_ms', 'receive_notification_to_D_prepare_ms')
    rows = []
    for start in starts:
        if start['method'] != 'Native':
            continue
        paths = start['timeline']['paths']
        assert len(paths) == 200
        error = max(abs(sum(p[k] for k in fields[:4]) - p['TTFT_ms']) for p in paths)
        assert error < 0.01, error
        rows.append(dict(name=start['name'], QPS=start['offered_QPS'],
            means={k: float(np.mean([p[k] for p in paths])) for k in fields},
            maximum_additive_identity_error_ms=error,
            KV_allocation_refusals=start['timeline']['KV_allocation_refusals']))
    cells = []
    for q in sorted({r['QPS'] for r in rows}):
        part = [r for r in rows if r['QPS'] == q]
        cells.append(dict(QPS=q, independent_starts=len(part), metrics={k: dict(
            mean=float(np.mean([r['means'][k] for r in part])),
            CI95=start_interval([r['means'][k] for r in part])) for k in fields}))
    write(ROOT / 'analysis/Native_host_critical_path_start_statistics.json', dict(
        per_start=rows, cells=cells, evidence='TRACE_ANALYSIS',
        scope='The first four host intervals sum to request TTFT. P first-prepare to ready '
        'includes chunk scheduling and is not exclusive GPU compute. Ready-to-D-prepare '
        'includes transfer and admission. Means are additive, interval p95 values are not. '
        'Only independent starts are resampled for confidence intervals.'))


if __name__ == '__main__':
    main()
