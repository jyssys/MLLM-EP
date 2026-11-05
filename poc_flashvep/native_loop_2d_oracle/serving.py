"""Bounded equal-trace serving screens after exact N32 qualification."""
import copy
import random
import traceback
from .common import *
from .prepare import RATES
from .run import execute


def serving_configuration(source, name, count=128, load='medium', start=1, markers=True):
    cfg = copy.deepcopy(source)
    cfg.update(name=name, count=count, state_probe=False, verify_transfer=False,
        observe_service_timing=markers, load=load, independent_start=start,
        arrival_rate=RATES[cfg['model']][load], arrival_seed=2026110401 + start,
        phase='FINITE_EVENT_MARKED_SERVING' if markers else 'MATCHED_MARKER_FREE_SERVING',
        timeout=600, startup_timeout=240)
    pool = json.loads((ROOT / 'data' / cfg['dataset']).read_text())
    cfg['prompt_ids'] = [row['prompt_id'] for row in pool[:count]]
    cfg['request_caps'] = [cfg['cap']] * count
    rng = random.Random(cfg['arrival_seed'])
    cfg['arrival_offsets'] = [0.]
    for _ in range(count - 1):
        cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
    return cfg


def evaluate(cfg, reference=None):
    save(cfg)
    name = cfg['name']
    cached = ROOT / 'raw' / name / 'serving_gate.json'
    if cached.exists():
        return json.loads(cached.read_text())
    row = dict(name=name, model=cfg['model'], method=cfg['method'], load=cfg['load'],
               count=cfg['count'], independent_start=cfg['independent_start'],
               markers=cfg['observe_service_timing'], complete=execute(name), passed=False)
    if row['complete']:
        from ..looped_pd_atomic_prefill import evidence
        from ..dynamic_loop_split_serving.lifetime_gate import check
        from ..dynamic_loop_split_serving.metrics import summarize
        from ..fluidtoken_loop_split.static_screen import metrics
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        with quiet_analysis(name + '_serving_gate'):
            row['work'] = evidence.full_work(name)
            row['lifetime'] = check(name)
            row['output'] = evidence.compare(reference, name) if reference else dict(passed=True)
            row['metrics'] = summarize(name) if cfg['observe_service_timing'] else metrics(name)
            before = json.loads((ROOT / 'raw' / name / 'before_measured.json').read_text())
            after = json.loads((ROOT / 'raw' / name / 'after_measured.json').read_text())
            counters = {}
            for role in ('prefill', 'decode'):
                keys = [key for key in after if '-' + role + '-' in key]
                assert len(keys) == 1
                key = keys[0]
                counters[role] = {field: after[key][field] - before[key][field] for field in
                    ('issued_loop_rows', 'issued_submissions', 'transport_bytes_sent', 'transport_completed_count')}
            d_prompt = counters['decode']['issued_loop_rows'] - cfg['count'] * (cfg['cap'] - 1) * cfg['depth']
            frontier = cfg.get('native_frontier')
            expected = (cfg['count'] * (cfg['depth'] - frontier[0]) * (cfg['length'] - frontier[1]) if frontier
                else cfg['count'] * (cfg['depth'] - cfg['native_partial_cut']) * cfg['length']
                if cfg.get('native_partial_cut') is not None else 0)
            row['actual_work'] = dict(D_prompt_recurrent_rows=d_prompt, expected_D_prompt_rows=expected,
                completed_offloaded_serving_requests=cfg['count'] if expected else 0,
                counters=counters, passed=d_prompt == expected,
                scope='Actual P-to-D serving continuation. No independent D-local probes. Rows are not FLOPs. Transfer bytes are payload counters, not control-message or wire-header bytes.')
            row['passed'] = all(row[key]['passed'] for key in ('work', 'lifetime', 'output', 'actual_work'))
    write(cached, row)
    return row


def main():
    initialize()
    gate_file = ROOT / 'state/effective_qualification.json'
    while time.time() < GPU_DEADLINE - 900:
        if gate_file.exists():
            gate = json.loads(gate_file.read_text())
            if gate['status'] == 'QUALIFICATION_FINISHED':
                break
        time.sleep(5)
    else:
        write(ROOT / 'state/serving_campaign.json', dict(status='NOT_RUN_TIME_BUDGET'))
        return
    ledger = dict(status='SERVING_IN_PROGRESS', models={},
        scope='Discovery uses medium start 1. Additional starts retain their selection scope. No new online scheduler is implemented.')
    target = ROOT / 'state/serving_campaign.json'
    def persist():
        write(target, ledger)
    for model in ('ouro', 'huginn'):
        qualified = gate['models'][model].get('qualification', {})
        row = ledger['models'][model] = dict(results=[], selected={})
        native = qualified.get('native', {})
        if not native.get('passed'):
            row['status'] = 'BLOCKED_NATIVE_N32'
            persist()
            continue
        sources = {method: json.loads((ROOT / 'configs' / (entry['config'] + '.json')).read_text())
                   for method, entry in qualified.items() if entry.get('passed')}
        reference = None
        for method, source in sources.items():
            cfg = serving_configuration(source, f'c_{model}_{method}_medium_start1_128')
            cfg['method'] = method
            result = evaluate(cfg, reference)
            row['results'].append(result)
            persist()
            if method == 'native':
                if not result['passed']:
                    break
                reference = cfg['name']
        safe = [item for item in row['results'] if item.get('passed')
                and item.get('metrics', {}).get('Decode_SLO_screen')]
        for axis, predicate in (('loop', lambda name: name.startswith('loop')),
                                ('token', lambda name: name.startswith('token')),
                                ('2d', lambda name: name.startswith('2d_'))):
            candidates = [item for item in safe if predicate(item['method'])]
            if candidates:
                best = max(candidates, key=lambda item: (item['metrics']['joint_SLO_goodput'],
                           -item['metrics']['TTFT_p95_ms']))
                row['selected'][axis] = best['method']
        selected = list(dict.fromkeys(['native', 'zero'] + list(row['selected'].values())))
        # Normal load is a finite N32 mechanism control for both models.
        # Heavy load expands only candidates that pass the medium Decode screen.
        for load, count in (('low', 32), ('heavy', 128)):
            reference = None
            for method in selected:
                if method not in sources:
                    continue
                cfg = serving_configuration(sources[method], f'c_{model}_{method}_{load}_start1_{count}', count, load)
                cfg['method'] = method
                result = evaluate(cfg, reference)
                row['results'].append(result)
                persist()
                if method == 'native':
                    if not result['passed']:
                        break
                    reference = cfg['name']
        # Three independent paired starts for the fixed zero-offload gate and
        # the start-1 selected exact, Decode-safe partial candidates.
        for start in (2, 3):
            reference = None
            order = selected if start == 2 else ['native'] + list(reversed(selected[1:]))
            for method in order:
                if method not in sources:
                    continue
                cfg = serving_configuration(sources[method], f'c_{model}_{method}_medium_start{start}_128', start=start)
                cfg['method'] = method
                result = evaluate(cfg, reference)
                row['results'].append(result)
                persist()
                if method == 'native':
                    if not result['passed']:
                        break
                    reference = cfg['name']
        # Marker controls are matched to the first medium start. Service TTFT
        # stays NOT_MEASURED in marker-free runs.
        reference = None
        for method in selected:
            if method not in sources:
                continue
            cfg = serving_configuration(sources[method], f'd_{model}_{method}_medium_marker_off128', markers=False)
            cfg['method'] = method
            result = evaluate(cfg, reference)
            row['results'].append(result)
            persist()
            if method == 'native':
                if not result['passed']:
                    break
                reference = cfg['name']
        row['status'] = 'SERVING_FINISHED'
        persist()
    ledger['status'] = 'SERVING_FINISHED'
    persist()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        initialize()
        write(ROOT / 'state/serving_exception.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
