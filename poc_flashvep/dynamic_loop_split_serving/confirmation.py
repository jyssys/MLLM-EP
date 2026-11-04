"""Confirm only completed, exact, Decode-safe split candidates."""
import copy
import random
import statistics
from .common import *
from .run import execute


def paired_interval(values):
    values = list(values)
    result = dict(starts=len(values), differences=values,
        scope='Paired independent-start differences. Requests and tokens are not independent replications.')
    if values:
        result['mean_difference'] = statistics.mean(values)
    if len(values) in (2, 3):
        critical = {2: 12.706204736432095, 3: 4.302652729911275}[len(values)]
        half = critical * statistics.stdev(values) / len(values) ** .5
        result['t_interval_95'] = [result['mean_difference'] - half, result['mean_difference'] + half]
        result['limitation'] = 'Small-sample Student t interval. Normal start-level differences remain an assumption.'
    else:
        result['t_interval_95'] = 'NOT_ESTIMATED'
    return result


def trace(cfg, name, count, start):
    result = copy.deepcopy(cfg)
    pool = json.loads((ROOT / 'data' / cfg['dataset']).read_text())[:128]
    result.update(name=name, count=count, independent_start=start,
        arrival_seed=2026110401 + start,
        prompt_ids=[pool[index % len(pool)]['prompt_id'] for index in range(count)],
        request_caps=[cfg['cap']] * count,
        phase='PAIRED_INDEPENDENT_CONFIRMATION' if count == 128 else 'CYCLIC_128_RECORD_PERSISTENCE_CONTROL',
        data_scope=dict(pool_records=len(pool), measured_requests=count,
            distinct_input_records=min(count, len(pool)), cyclic_reuse=count > len(pool),
            scope='Native prefix caching remains disabled. Cyclic records are not independent texts.'))
    rng = random.Random(result['arrival_seed'])
    result['arrival_offsets'] = [0.]
    for _ in range(count - 1):
        result['arrival_offsets'].append(result['arrival_offsets'][-1] + rng.expovariate(result['arrival_rate']))
    result['timeout'] = max(cfg.get('timeout', 900), int(result['arrival_offsets'][-1] + 240) + 1)
    return result


def main():
    initialize()
    paths = [ROOT / 'state' / (name + '.json') for name in ('static_campaign', 'dynamic_campaign')]
    while time.time() < LONG_START_CUTOFF:
        if all(path.exists() for path in paths):
            ledgers = [json.loads(path.read_text()) for path in paths]
            if all(row['status'].endswith('_CAMPAIGN_FINISHED') for row in ledgers):
                break
        time.sleep(5)
    else:
        write(ROOT / 'state/confirmation.json', dict(status='NOT_RUN_TIME_BUDGET'))
        return
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .metrics import summarize
    from .work_accounting import summarize as accounting
    from .lifetime_gate import check as lifetime
    ledger = dict(status='IN_PROGRESS', models={})
    target = ROOT / 'state/confirmation.json'
    for model in ('ouro', 'huginn'):
        static, dynamic = (row['models'].get(model, {}) for row in ledgers)
        candidates = static.get('serving', []) + dynamic.get('serving', [])
        safe = [row for row in candidates if row.get('passed')
                and row.get('metrics', {}).get('Decode_SLO_screen')]
        chosen = None
        for load in ('medium', 'heavy', 'low'):
            baseline = next((row for row in safe if row.get('plan') == 'native' and row.get('load') == load), None)
            if baseline is None:
                continue
            axes = {}
            for axis in ('token', 'loop'):
                possibilities = []
                for row in safe:
                    if row.get('load') != load or row.get('plan') in ('native', 'staged_native', 'snapshot_full', 'token_full'):
                        continue
                    cfg = json.loads((ROOT / 'configs' / (row['name'] + '.json')).read_text())
                    work = row.get('accounting', row.get('work_and_payload', {}))
                    if cfg['axis'] != axis or work.get('completed_offloaded_serving_requests', 0) == 0:
                        continue
                    a, b = row['metrics'], baseline['metrics']
                    if a['joint_SLO_goodput'] >= b['joint_SLO_goodput'] and (
                            a['TTFT_p95_ms'] < b['TTFT_p95_ms'] or a['service_TTFT_ms_p95'] < b['service_TTFT_ms_p95']):
                        possibilities.append(row)
                if possibilities:
                    axes[axis] = max(possibilities, key=lambda row:
                        (row['metrics']['joint_SLO_goodput'], -row['metrics']['TTFT_p95_ms']))
            if axes:
                chosen = dict(load=load, baseline=baseline, axes=axes)
                break
        row = ledger['models'][model] = dict(results=[], selection=chosen,
            scope='Selection uses start 1. Starts 2 and 3 are separate paired confirmation traces. No new SLO or online future information.')
        write(target, ledger)
        if chosen is None:
            row['status'] = 'NO_PROMISING_EXACT_DECODE_SAFE_SPLIT'
            write(target, ledger)
            continue
        original = [('native', chosen['baseline'])] + list(chosen['axes'].items())
        all_passed = True
        for count, start in ((128, 2), (128, 3), (512, 4)):
            if time.time() >= LONG_START_CUTOFF or not all_passed:
                row['results'].append(dict(count=count, start=start,
                    status='NOT_RUN_TIME_BUDGET_OR_PREVIOUS_GATE'))
                continue
            reference = None
            for label, selected in original:
                base = json.loads((ROOT / 'configs' / (selected['name'] + '.json')).read_text())
                cfg = trace(base, f'l_{model}_{chosen["load"]}_{label}_start{start}_{count}', count, start)
                save(cfg)
                result = dict(name=cfg['name'], label=label, count=count, start=start,
                    complete=execute(cfg['name']), passed=False)
                if result['complete']:
                    with quiet_analysis(cfg['name'] + '_confirmation'):
                        result['output'] = evidence.compare(reference, cfg['name']) if reference else dict(passed=True)
                        result['work'] = evidence.full_work(cfg['name'])
                        result['lifetime'] = lifetime(cfg['name'])
                        result['metrics'] = summarize(cfg['name'])
                        result['accounting'] = accounting(cfg['name'])
                        result['passed'] = all(result[key]['passed'] for key in ('output', 'work', 'lifetime'))
                row['results'].append(result)
                write(target, ledger)
                if not result['passed']:
                    all_passed = False
                    break
                reference = reference or cfg['name']
                if not result['metrics']['Decode_SLO_screen']:
                    all_passed = False
            write(target, ledger)
        row['status'] = 'CONFIRMATION_EXECUTION_FINISHED_REVIEW_REQUIRED'
        row['paired_start_statistics'] = {}
        for axis, selected in chosen['axes'].items():
            paired = [(1, selected['metrics'], chosen['baseline']['metrics'])]
            for start in (2, 3):
                candidate = next((r for r in row['results'] if r.get('label') == axis
                                  and r.get('start') == start and r.get('passed')), None)
                native = next((r for r in row['results'] if r.get('label') == 'native'
                               and r.get('start') == start and r.get('passed')), None)
                if candidate and native:
                    paired.append((start, candidate['metrics'], native['metrics']))
            row['paired_start_statistics'][axis] = {field: dict(
                including_selection_start=paired_interval(a[field] - b[field] for _, a, b in paired),
                held_out_starts_only=paired_interval(a[field] - b[field] for start, a, b in paired if start > 1))
                for field in ('TTFT_p95_ms', 'service_TTFT_ms_p95', 'joint_SLO_goodput',
                              'TPOT_p95_ms', 'ITL_p99_ms')}
        write(target, ledger)
    ledger['status'] = 'CONFIRMATION_CAMPAIGN_FINISHED'
    write(target, ledger)


if __name__ == '__main__':
    main()
