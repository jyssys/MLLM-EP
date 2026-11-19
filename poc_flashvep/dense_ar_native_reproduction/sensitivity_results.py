"""Summarize one-factor discovery without claiming a tuned capacity frontier."""
import csv
import json
from .common import ROOT, write
from .serving_analysis import analyze


def main():
    results = json.loads((ROOT / 'analysis/B_S_sensitivity_v1.json').read_text())['results']
    assert len(results) == 24, 'Finish the frozen sensitivity design before final aggregation'
    fields = ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms',
              'generated_output_tokens_per_second', 'Joint_SLO_attainment',
              'SLO_qualified_output_tokens_per_second')
    rows = []
    frozen = json.loads((ROOT / 'state/runtime_source_freeze.json').read_text())['files']
    for result in results:
        folder = ROOT / 'raw' / result['name']
        cfg = json.loads((folder / 'configuration.json').read_text())
        row = dict(name=result['name'], method=cfg['method'], QPS=cfg['offered_QPS'],
                   B=cfg['token_budget'], S=cfg['max_num_seqs'], valid=result['valid'])
        hashes = json.loads((folder / 'source_hashes.json').read_text())
        row['runtime_source_hashes'] = {k: hashes.get('poc_flashvep/dense_ar_native_reproduction/'+k)
                                        for k in frozen}
        row['runtime_files_different_from_Main_freeze'] = [k for k, v in frozen.items()
            if row['runtime_source_hashes'][k] != v]
        if result['valid']:
            actual = [json.loads(p.read_text()) for p in folder.glob('actual_runtime_*.json')]
            assert len(actual) == 2
            assert all((a['scheduler_B'], a['scheduler_S']) == (row['B'], row['S']) for a in actual)
            trace = analyze(result['name'])
            row.update(metrics={k: result['metrics'][k] for k in fields},
                       actual_runtime=[dict(role=a['role'], B=a['scheduler_B'], S=a['scheduler_S'],
                           chunked_prefill=a['chunked_prefill'], KV_blocks=a['KV_blocks'],
                           UUID=a['uuid']) for a in actual],
                       actual_geometry=trace['geometry'])
        else:
            row['error'] = result.get('error')
        rows.append(row)
    write(ROOT / 'analysis/B_S_final_discovery_summary.json', dict(rows=rows,
        independent_starts_per_cell=1, evidence='MEASURED_DISCOVERY_SENSITIVITY',
        scope='Same N40 pilot and arrival seed. One-factor B/S changes. Actual workers are checked. '
        'The moderate Native B8192/S1024 control predates the expanded read-only queue observer. '
        'Per-run runtime hashes expose this limitation. Small differences are diagnostic. '
        'No start-level CI, global optimality, independent tuned confirmation or sustainable capacity claim.'))
    with (ROOT / 'analysis/B_S_final_discovery_table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'QPS', 'B', 'S', 'valid', *fields])
        for row in rows:
            writer.writerow([row[k] for k in ('method', 'QPS', 'B', 'S', 'valid')] +
                            [row.get('metrics', {}).get(k) for k in fields])


if __name__ == '__main__':
    main()
