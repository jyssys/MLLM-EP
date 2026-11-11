"""Apply frozen confirmation gates without turning a missing measurement into a pass."""
import argparse
from .common import *


def check_ratio(item, point_limit, upper_limit):
    if item is None or item.get('geometric_ratio') is None:
        return dict(status='NOT_MEASURED')
    point = item['geometric_ratio']
    interval = item.get('bootstrap_95_CI')
    if item['paired_starts'] < 3 or interval is None:
        return dict(status='INSUFFICIENT_INDEPENDENT_STARTS', point=point, interval=interval)
    passed = point <= point_limit and (interval[1] < upper_limit if point_limit < upper_limit
        else interval[1] <= upper_limit)
    return dict(status='PASS' if passed else 'FAIL', point=point, interval=interval,
        point_limit=point_limit, upper_limit=upper_limit, paired_starts=item['paired_starts'])


def compare(arm, rows, ratios, outputs):
    baseline = {row['arrival_seed']:row for row in rows if row['label'] == 'zero'}
    candidate = {row['arrival_seed']:row for row in rows if row['label'] == arm}
    result = dict(arm=arm, paired_seeds=sorted(set(baseline)&set(candidate)))
    lookup = {row['metric']:row for row in ratios if row['arm'] == arm and row['reference'] == 'zero'}
    limits = dict(TTFT_p95_ms=(.95,1.), E2E_p95_ms=(1.,1.),
        ITL_p99_ms=(1.05,1.05), TPOT_p95_ms=(1.05,1.05))
    result['relative_latency'] = {key:check_ratio(lookup.get(key), *limit)
        for key, limit in limits.items()}
    checks = []
    for seed in result['paired_seeds']:
        a, b = candidate[seed], baseline[seed]
        folder = ROOT/'raw'/a['actual_name']
        exits = [json.loads(path.read_text()) for path in folder.glob('native_worker_exit_*.json')]
        exit_pass = any(value['passed'] and {row['role'] for row in value['records']
            if row['exitcode'] == 0 and not row['forced_cleanup']} == {'prefill','decode'} for value in exits)
        cfg = json.loads((ROOT/'configs'/(a['actual_name']+'.json')).read_text())
        state = ROOT/'raw'/cfg.get('required_qualification','')/'qualification_with_frozen_reference.json'
        qualified = json.loads(state.read_text()) if state.exists() else {}
        checks.append(dict(seed=seed, actual_name=a['actual_name'],
            exact_output=outputs.get(a['actual_name'],{}).get('passed', False),
            state_and_lifetime_qualified=qualified.get('campaign_gate_passed',qualified.get('passed',False)),
            actual_worker_exit_pass=exit_pass, absolute_Decode_SLO=a['Decode_SLO_screen'],
            additional_frozen_ITL_threshold_exceedances=a['ITL_limit_misses']-b['ITL_limit_misses'],
            maximum_ITL_ms=a['maximum_ITL_ms'], zero_maximum_ITL_ms=b['maximum_ITL_ms'],
            offloaded_requests=a['offloaded_requests'], actual_D_prompt_recurrent_rows=a['actual_D_prompt_recurrent_rows'],
            goodput_difference=a['SLO_goodput']-b['SLO_goodput']))
    result['checks_by_seed'] = checks
    result['all_latency_gates'] = all(v['status'] == 'PASS' for v in result['relative_latency'].values())
    result['all_run_integrity_gates'] = len(checks) >= 3 and all(
        row['exact_output'] and row['state_and_lifetime_qualified'] and row['actual_worker_exit_pass']
        and row['absolute_Decode_SLO'] and row['additional_frozen_ITL_threshold_exceedances'] <= 0
        for row in checks)
    result['actual_migration_demonstrated'] = any(row['offloaded_requests'] > 0 for row in checks)
    result['P_critical_path_gate'] = 'REQUIRES_EXPLICIT_DEVICE_EVIDENCE'
    result['scope'] = ('These checks do not establish a scientific GO by themselves. '
        'The report must inspect maximum ITL, actual P critical-path evidence, goodput, '
        'individual completion economics, and the strongest Token control.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    args = parser.parse_args()
    folder = ROOT/'confirmation'
    rows, ratios, outputs = [json.loads((folder/(args.tag+'_'+name+'.json')).read_text())
        for name in ('serving_results','paired_start_ratios','output_exactness')]
    labels = sorted({row['label'] for row in rows}-{'zero','zerorepeat'})
    result = [compare(label, rows, ratios, {row['candidate']:row for row in outputs}) for label in labels]
    write(folder/(args.tag+'_scientific_gate_checks.json'), result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
