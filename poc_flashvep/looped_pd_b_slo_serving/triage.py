"""Separate correctness diagnostics. No result here qualifies a serving speedup."""
import json
import time

from .common import ROOT, write, remaining
from .jobs import configuration, run, parity


def main():
    findings = {}
    for model in ('ouro', 'huginn'):
        failed = ROOT / 'state' / (model + '_B_serving_parity_failure.json')
        if not failed.exists():
            continue
        failure = json.loads(failed.read_text())
        original_left, original_right = failure['left'], failure['right']
        contract = json.loads((ROOT / 'configs/jobs' / (original_left + '.json')).read_text())
        source = json.loads((ROOT / 'data' / (contract['dataset'] + '.json')).read_text())
        indices = [row['sample_id'] for row in failure['differences']]
        data_name = model + '_mismatch_diagnostic'
        write(ROOT / 'data' / (data_name + '.json'), [source[index] for index in indices])
        result = dict(original_failure=failure, selected_original_sample_ids=indices,
                      sequential_seed_rule='Both sequential arms use seed plus subset index. Huginn initial-state draws therefore need not equal the original sample-index draws.',
                      scope='Correctness only. Selected mismatch subset is not an accuracy or performance sample.')
        if remaining() >= 3000:
            # A repeated control tests whether natural batching changes its own output.
            repeat = configuration(model, 'PD_G1', contract['workload'], contract['count'],
                                   contract['rate'], contract['seed'], gpus=contract['gpus'],
                                   dataset=contract['dataset'], cap=contract['cap'],
                                   capacity=contract['capacity'], diagnostic=True,
                                   name=model + '_diagnostic_control_repeat', timeout=1800)
            run(repeat)
            result['control_repeat_vs_original_control'] = parity(original_left, repeat['name'])
            result['control_repeat_vs_original_B'] = parity(repeat['name'], original_right)
        if indices and remaining() >= 1800:
            jobs = {}
            for arm in ('PD_G1', 'B'):
                if remaining() < 900:
                    break
                job = configuration(model, arm, 'diagnostic', len(indices), 0,
                                    contract['seed'], gpus=(0, 1), dataset=data_name,
                                    cap=contract['cap'], capacity=1, diagnostic=True,
                                    name=f'{model}_diagnostic_sequential_{arm}', timeout=1200)
                run(job)
                jobs[arm] = job
            if len(jobs) == 2:
                result['sequential_matched_output'] = parity(jobs['PD_G1']['name'], jobs['B']['name'])
        if remaining() >= 1800:
            # Byte identity under concurrent arrivals is separate from sequence parity.
            job = configuration(model, 'B', 'diagnostic', 32, 2. if model == 'ouro' else .4,
                                contract['seed'], gpus=(0, 1), dataset='W1', cap=32,
                                capacity=8, diagnostic=True, timeout=1200,
                                env=dict(ABC_VERIFY_KV='1', ABC_REVERSE_FRAGMENTS='1'),
                                name=model + '_diagnostic_concurrent_KV_B')
            run(job)
            folder = ROOT / 'raw' / job['name']
            metric_file = folder / 'closed_stats.json'
            measured_tids = set()
            if metric_file.exists():
                measured_tids = {row['transfer_id'] for row in json.loads(metric_file.read_text()).get('metrics', [])
                                 if row['request_id'].startswith('r0_')}
            checks = [json.loads(line) for path in folder.glob('KV_checks-*.jsonl')
                      for line in path.read_text().splitlines()]
            checks = [row for row in checks if row['tid'] in measured_tids]
            result['concurrent_KV_and_hidden_identity'] = dict(
                N=len(checks), exact=sum(row['passed'] for row in checks), planned=32,
                evidence='HEAVY_INSTRUMENTED_BYTE_IDENTITY_NOT_OUTPUT_ACCURACY_OR_CLEAN_PERFORMANCE')
        result['time'] = time.time()
        result['B_performance_gate_remains_closed'] = True
        findings[model] = result
        write(ROOT / 'configs/mismatch_triage.json', findings)
    # Profiling remains a separate diagnostic even when the serving gate is closed.
    from .profile import main as profile
    profile()
    return findings


if __name__ == '__main__':
    main()
