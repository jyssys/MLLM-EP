"""Bounded clean static controls after each model's exact connector gate."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def metrics(name):
    from ..looped_pd_headroom_recovery.gates import percentile
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    rows = [row for row in rows if not row.get('warmup')]
    slo = json.loads((ROOT / 'configs/frozen_SLO.json').read_text())[cfg['model'] + '_L' + str(cfg['length'])]
    itl = [1000 * (event['time'] - event['previous_time']) / event['new_tokens']
           for row in rows for event in row['token_events'] if not event['first']]
    span = max(row['finish'] for row in rows) - min(row['arrival'] for row in rows)
    joint = sum(row['ttft_ms'] <= slo['TTFT_ms'] and row['tpot_ms'] <= slo['TPOT_ms'] for row in rows)
    result = dict(completed=len(rows), offered=cfg['count'], response_span_s=span,
        requests_per_second=len(rows) / span, joint_SLO_goodput=joint / span,
        joint_SLO_attainment=joint / cfg['count'],
        total_tokens_per_second=sum(row['input_tokens'] + row['output_tokens'] for row in rows) / span,
        output_tokens_per_second=sum(row['output_tokens'] for row in rows) / span,
        maximum_ITL_ms=max(itl), frozen_SLO=slo,
        scope='Clean finite pilot. Host request latencies, not exclusive GPU durations or sustainable capacity.')
    for label, values in [('TTFT', [r['ttft_ms'] for r in rows]), ('TPOT', [r['tpot_ms'] for r in rows]),
                          ('E2E', [r['e2e_ms'] for r in rows]), ('ITL', itl)]:
        for p in (.5, .95, .99):
            result[f'{label}_p{int(p * 100)}_ms'] = percentile(values, p)
    result['Decode_SLO_screen'] = (result['TPOT_p95_ms'] <= .9 * slo['TPOT_ms'] and
                                  result['ITL_p99_ms'] <= slo['ITL_ms'])
    return result


def main():
    initialize()
    path = ROOT / 'state/conditional_split_qualification.json'
    target = ROOT / 'state/static_screen.json'
    while time.time() < LONG_START_CUTOFF:
        if path.exists():
            state = json.loads(path.read_text())
            if state['status'] != 'IN_PROGRESS':
                break
        time.sleep(5)
    else:
        write(target, dict(status='NOT_RUN_TIME_BUDGET'))
        return
    if state['status'] != 'QUALIFICATION_FINISHED':
        write(target, dict(status='BLOCKED_BY_HUGINN_GATE'))
        return
    from ..looped_pd_atomic_prefill import evidence
    ledger = []
    for model in ('huginn', 'ouro'):
        qualification = state['results'][model]
        if state.get('extended_results', {}).get(model, {}).get('status') != 'PASSED_32_REQUEST_STATE_GATE':
            ledger.append(dict(model=model, status='BLOCKED_32_REQUEST_STATE_GATE'))
            continue
        if len(qualification) != 5 or not all(row['passed'] for row in qualification):
            ledger.append(dict(model=model, status='BLOCKED_CONNECTOR_CORRECTNESS'))
            continue
        reference = None
        for qualified in qualification:
            if time.time() >= LONG_START_CUTOFF:
                ledger.append(dict(model=model, status='NOT_RUN_TIME_BUDGET'))
                break
            cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (qualified['name'] + '.json')).read_text()))
            source = json.loads(((ROOT / 'configs/recovery_huginn_admitted_probe_clean512.json') if model == 'huginn'
                else (PREVIOUS / 'configs/d_graph_ouro_state_verified_matched128.json')).read_text())
            # Retain identical preserved prompts, seeds, and arrival trace across all five controls.
            cfg.update(name=qualified['name'].replace('static_gate_v4_', 'static_screen_v2_')[:-1] + '32',
                count=32, dataset=source['dataset'], seed=source['seed'],
                state_probe=False, verify_transfer=False,
                phase='CLEAN_STATIC_SCREEN_NOT_FINAL_CAPACITY', timeout=1200)
            for key in ('request_caps', 'prompt_ids'):
                cfg[key] = source[key][:32]
            import random
            rng = random.Random(cfg['arrival_seed'])
            cfg['arrival_offsets'] = [0.]
            for _ in range(31):
                cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
            cfg['data_scope'] = dict(offered_count=32, explicit_cyclic_reuse=False,
                selection='First 32 preserved prompts. Correctness configuration supplies the common memory budget.')
            save(cfg)
            complete = execute(cfg['name'])
            row = dict(model=model, name=cfg['name'], complete=complete)
            if complete:
                row['metrics'] = metrics(cfg['name'])
                if reference:
                    row['output'] = evidence.compare(reference, cfg['name'])
                else:
                    reference = cfg['name']
                row['correct'] = row.get('output', {'passed': True})['passed']
            ledger.append(row)
            write(target, dict(status='IN_PROGRESS', results=ledger))
            if not complete or not row['correct']:
                break
    write(target, dict(status='STATIC_SCREEN_FINISHED', results=ledger,
        limitation='N32 exploratory controls only. Dynamic methods and final repeated serving are not included.'))


if __name__ == '__main__':
    main()
