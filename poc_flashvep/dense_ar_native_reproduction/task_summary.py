"""Keep task-level request fractions separate from output-token throughput."""
import csv
import json
import numpy as np
from .common import ROOT, write
from .figures import start_interval


def main():
    starts = json.loads((ROOT / 'analysis/Main_full_analysis.json').read_text())['per_start']
    fields = ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'mean_TBT_ms',
              'Joint_SLO_attainment', 'output_tokens_per_global_window_second')
    rows = []
    for q in sorted({s['offered_QPS'] for s in starts}):
        for method in ('Native', 'Agg'):
            part = [s for s in starts if s['offered_QPS'] == q and s['method'] == method]
            for task in sorted(part[0]['metrics']['by_task']):
                values = [s['metrics']['by_task'][task] for s in part]
                rows.append(dict(QPS=q, method=method, task=task, requests_per_start=40,
                    independent_starts=len(part), metrics={k: dict(
                        mean=float(np.mean([v[k] for v in values])),
                        CI95=start_interval([v[k] for v in values])) for k in fields}))
    write(ROOT / 'analysis/Main_task_start_level_statistics.json', dict(rows=rows,
        evidence='MEASURED_ONLINE', scope='Each task has 40 requests per server start. '
        'Confidence intervals resample independent server starts, not individual requests. '
        'Task output throughput uses the global measurement window, not a separate task window.'))
    with (ROOT / 'analysis/Main_task_cell_table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['QPS', 'method', 'task', 'independent_starts'] +
                        [c for k in fields for c in (k, k+'_CI95_low', k+'_CI95_high')])
        for row in rows:
            out = [row[k] for k in ('QPS', 'method', 'task', 'independent_starts')]
            for k in fields:
                value = row['metrics'][k]
                out.extend([value['mean'], *(value['CI95'] or [None, None])])
            writer.writerow(out)


if __name__ == '__main__':
    main()
