"""CPU analysis of completed mechanism runs. No latency model fills missing data."""
import csv
import json
import statistics
from .common import ROOT, setup, write


def read_rows(folder):
    path = folder/'requests.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if not json.loads(line)['warmup']]


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*q
    left = int(index)
    return values[left]+(values[min(left+1,len(values)-1)]-values[left])*(index-left)


def main():
    setup()
    runs = {}
    summary = []
    comparisons = []
    for folder in sorted((ROOT/'raw').iterdir()):
        status = folder/'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'COMPLETE':
            continue
        cfg = json.loads((folder/'rcc_contract.json').read_text())
        rows = read_rows(folder)
        if len(rows) != cfg['count']:
            raise RuntimeError('A complete run has an incomplete request file')
        runs[folder.name] = rows
        probes = []
        for path in folder.glob('fusion_probe_*.json'):
            probes.append(json.loads(path.read_text()))
        # Worker Request IDs are transport IDs, not coordinator IDs. Seeds identify measured rows.
        seeds = {cfg['seed']+index for index in range(cfg['count'])}
        # Warmup uses the same seeds. Host timestamps separate measured submissions.
        window = json.loads((folder/'window.json').read_text())
        batches = [b for p in probes for b in p['batches']
                   if b['host_begin'] >= window['start'] and all(s['seed'] in seeds for s in b['spans'])]
        mixed = [b for b in batches if {s['phase'] for s in b['spans']} >= {'decode', 'prefill'}]
        mixed_depths = [b for b in mixed if len({s['depth'] for s in b['spans']}) > 1]
        kv = [json.loads(line) for path in folder.glob('KV_checks-*.jsonl') for line in path.read_text().splitlines()]
        before, after = (json.loads((folder/(n+'.json')).read_text()) for n in ('before_measured', 'after_measured'))
        loop_rows = sum(value['issued_loop_rows']-before[name]['issued_loop_rows'] for name, value in after.items())
        expected = cfg['count']*len(rows[0]['input_ids'])*rows[0]['depth'] + sum((r['output_tokens']-1)*r['depth'] for r in rows)
        transfer_bytes = sum(value['transport_bytes_sent']-before[name]['transport_bytes_sent'] for name, value in after.items())
        itls = [1000*(e['time']-e['previous_time']) for r in rows for e in r['token_events'] if not e['first'] and e['new_tokens'] == 1]
        delivered_bundles = sum(e['new_tokens'] != 1 for r in rows for e in r['token_events'] if not e['first'])
        fragments = sum(value['transport_completed_count']-before[name]['transport_completed_count'] for name, value in after.items())
        row = dict(name=folder.name, model=cfg['model'], arm=cfg['arm'], gpus=','.join(map(str, cfg['gpus'])),
                   evidence=rows[0]['evidence'], requests=len(rows), input_tokens=cfg['chunk'],
                   output_tokens=sum(r['output_tokens'] for r in rows), makespan_ms=1000*window['seconds'],
                   mean_TTFT_ms=statistics.mean(r['ttft_ms'] for r in rows),
                   mean_TPOT_ms=statistics.mean(r['tpot_ms'] for r in rows),
                   mean_E2E_ms=statistics.mean(r['e2e_ms'] for r in rows),
                   mixed_batches=len(mixed), cross_depth_mixed_batches=len(mixed_depths),
                   issued_loop_rows=loop_rows, expected_loop_rows=expected, work_exact=loop_rows == expected,
                   transfer_bytes=transfer_bytes, transfer_fragments=fragments,
                   transfer_identity_passed=sum(k['passed'] for k in kv),
                   transfer_identity_checks=len(kv), peak_worker_HBM_bytes=max(v['peak_allocated_bytes'] for v in after.values()))
        for field, values in [('TTFT',[r['ttft_ms'] for r in rows]), ('TPOT',[r['tpot_ms'] for r in rows]),
                              ('E2E',[r['e2e_ms'] for r in rows]), ('ITL',itls)]:
            for label, quantile in (('P50',.5),('P95',.95),('P99',.99)):
                row[field+'_'+label+'_ms'] = percentile(values,quantile)
        row.update(ITL_observations=len(itls), bundled_delivery_events=delivered_bundles,
                   completed_requests_per_second=len(rows)/window['seconds'],
                   output_tokens_per_second=sum(r['output_tokens'] for r in rows)/window['seconds'],
                   SLO_goodput='NOT_MEASURED', sustainable_capacity='NOT_MEASURED')
        nvml = [json.loads(p.read_text()) for p in folder.glob('NVML-summary-*.json')]
        samples = [v for item in nvml for v in item.get('measured_cohort_peak_sampled_HBM_bytes',{}).values()]
        row['peak_NVML_sampled_HBM_bytes'] = max(samples) if samples else 'NOT_MEASURED'
        summary.append(row)
        if mixed:
            write(folder/'mixed_batch_examples.json', mixed[:10])
    for name, rows in runs.items():
        cfg = json.loads((ROOT/'raw'/name/'rcc_contract.json').read_text())
        for target_arm in ('PD_G1', 'A_sep'):
            target = name.replace('_'+cfg['arm']+'_', '_'+target_arm+'_')
            if target not in runs or target == name:
                continue
            other = {r['sample_id']:r for r in runs[target]}
            comparisons.append(dict(name=name, reference=target, N=len(rows),
                                    inputs_exact=sum(r['input_ids'] == other[r['sample_id']]['input_ids'] for r in rows),
                                    tokens_exact=sum(r['output_ids'] == other[r['sample_id']]['output_ids'] for r in rows),
                                    depths_exact=sum(r['exit_depths'] == other[r['sample_id']]['exit_depths'] for r in rows),
                                    stop_exact=sum(r['finish_reason'] == other[r['sample_id']]['finish_reason'] for r in rows),
                                    mismatches=[dict(sample_id=r['sample_id'],
                                                     first_output_difference=next((i for i, (x,y) in enumerate(zip(r['output_ids'], other[r['sample_id']]['output_ids'])) if x != y), 'length'))
                                                for r in rows if r['output_ids'] != other[r['sample_id']]['output_ids']]))
    for filename, values in (('full_results.csv', summary), ('correctness_matrix.json', comparisons)):
        if filename.endswith('.json'):
            write(ROOT/filename, values)
        elif values:
            with (ROOT/filename).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(values)
    print(json.dumps(dict(runs=summary, comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
