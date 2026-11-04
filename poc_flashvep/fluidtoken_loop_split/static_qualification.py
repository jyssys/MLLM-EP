"""Qualify existing exact connectors before any method latency comparison."""
import argparse
import copy
import random
import time
from .common import *
from .run import execute
from .recovery_qualification import save
from .state_comparison import compare as compare_states


def prepare(model):
    initialize()
    if model == 'ouro':
        path = PREVIOUS / 'configs/d_graph_ouro_state_verified_matched128.json'
    else:
        path = ROOT / 'configs/recovery_huginn_graph_state8.json'
    base = json.loads(path.read_text())
    if model == 'huginn':
        recovered = json.loads((ROOT / 'state/huginn_final_recovery_gate.json').read_text())
        assert recovered['status'] == 'RECOVERY_READY'
        for key in ('D_slots', 'graph_rows', 'route_campaign_evidence_async',
                    'D_local_probe_output_pacing', 'arrival_rate', 'arrival_seed', 'arrival_offsets'):
            if key in recovered['qualified_configuration']:
                base[key] = recovered['qualified_configuration'][key]
    source_data = (PREVIOUS if model == 'ouro' else ROOT) / 'data' / base['dataset']
    data = json.loads(source_data.read_text())
    target = ROOT / 'data' / base['dataset']
    if not target.exists():
        write(target, data)
    else:
        assert json.loads(target.read_text()) == data
    for key in list(base):
        if key.startswith('D_probe_'):
            base.pop(key)
    base.update(count=8, state_probe=True, verify_transfer=True,
                route_campaign_evidence_async=True,
                phase='STATIC_CONNECTOR_CORRECTNESS_NOT_LATENCY',
                timeout=1200, estimated_seconds=300,
                D_token_budget=base['length'], D_prefill_graph_rows=base['length'],
                D_kv_blocks=26752 if model == 'huginn' else 16384,
                nsys=False, P_only_profiler=False,
                scheduler_mode='refill', dynamic=False)
    base['data_scope'] = dict(original_pool_records=len(data), offered_count=8,
                              explicit_cyclic_reuse=False, selection='First eight preserved records')
    for key in ('request_caps', 'arrival_offsets', 'prompt_ids'):
        base[key] = base[key][:8]
    plans = [
        ('atomic_native', dict(arm='PD_G1', wavefront=False, atomic_prompt=True,
                              handoff_mode='native_G1', split_evaluation=False)),
        ('staged_native', dict(arm='PD_G1', wavefront=True, atomic_prompt=False,
                              handoff_mode='native_G1', split_evaluation=True)),
        ('staged_snapshot_full', dict(arm='Loop_Static', axis='loop', cut=base['depth'])),
        ('loop_partial', dict(arm='Loop_Static', axis='loop', cut=3 if model == 'ouro' else 31)),
        ('token_partial', dict(arm='Token_Static', axis='token', cut=base['depth'],
                              prefix_tokens=base['length'] * (3 if model == 'ouro' else 31) // base['depth'])),
    ]
    names = []
    for label, values in plans:
        cfg = copy.deepcopy(base)
        cfg.update(wavefront=True, atomic_prompt=False, split_evaluation=True,
                   handoff_mode='snapshot')
        cfg.update(values)
        cfg['name'] = f'static_gate_v4_{model}_{label}8'
        save(cfg)
        names.append(cfg['name'])
    return names


def connector_gate(name):
    from ..looped_pd_atomic_prefill import evidence
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    if cfg['arm'] == 'PD_G1':
        transport = evidence.transfer(name)
    else:
        expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
        checks = [json.loads(line) for path in (ROOT / 'raw' / name).glob('KV_checks_*.jsonl')
                  for line in path.read_text().splitlines()]
        selected = [row for row in checks if row['seed'] in expected]
        observed = {row['seed'] for row in selected}
        transport = dict(expected=len(expected), compared=len(observed),
                         passed=observed == expected and all(row['passed'] for row in selected),
                         scope='Snapshot receiver fingerprint compared with the source fingerprint.')
    work = evidence.full_work(name)
    result = dict(transfer=transport, work=work, passed=transport['passed'] and work['passed'])
    write(ROOT / 'raw' / name / 'split_connector_gate.json', result)
    return result['passed']


def main(model, run):
    names = prepare(model)
    if not run:
        print(json.dumps(names))
        return
    from ..looped_pd_atomic_prefill import evidence
    ledger = []
    for name in names:
        complete = execute(name)
        row = dict(name=name, complete=complete, byte_and_work_gate=complete and connector_gate(name))
        if complete and name != names[0]:
            row['output'] = evidence.compare(names[0], name)
            row['prompt_state'] = compare_states(names[0], name)
        row['passed'] = bool(row['byte_and_work_gate']
                             and row.get('output', {'passed': True})['passed']
                             and row.get('prompt_state', {'exact': True})['exact'])
        ledger.append(row)
        write(ROOT / 'state' / f'static_qualification_{model}.json', ledger)
        if not row['passed']:
            break


def extended(model):
    """Check 32 requests on a new arrival seed before performance expansion."""
    from ..looped_pd_atomic_prefill import evidence
    original = json.loads((ROOT / 'state' / f'static_qualification_{model}.json').read_text())
    if len(original) != 5 or not all(row['passed'] for row in original):
        return dict(status='BLOCKED_EIGHT_REQUEST_GATE', results=[])
    source = json.loads(((ROOT / 'configs/recovery_huginn_admitted_probe_clean512.json') if model == 'huginn'
        else (PREVIOUS / 'configs/d_graph_ouro_state_verified_matched128.json')).read_text())
    ledger = []
    reference = None
    for label in ('atomic_native', 'loop_partial', 'token_partial'):
        if time.time() >= LONG_START_CUTOFF:
            return dict(status='NOT_RUN_TIME_BUDGET', results=ledger)
        cfg = json.loads((ROOT / 'configs' / f'static_gate_v4_{model}_{label}8.json').read_text())
        cfg.update(name=f'static_gate_v4_{model}_{label}32', count=32,
            dataset=source['dataset'], seed=source['seed'], arrival_seed=2026111801,
            timeout=1800, estimated_seconds=600,
            phase='EXACT_STATE_32_INDEPENDENT_ARRIVAL_NOT_LATENCY')
        for key in ('request_caps', 'prompt_ids'):
            cfg[key] = source[key][:32]
        rng = random.Random(cfg['arrival_seed'])
        cfg['arrival_offsets'] = [0.]
        for _ in range(31):
            cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
        cfg['data_scope'] = dict(offered_count=32, explicit_cyclic_reuse=False,
            selection='First 32 preserved records, with a new Poisson arrival seed. Heavy state checks are not latency evidence.')
        save(cfg)
        complete = execute(cfg['name'])
        row = dict(name=cfg['name'], complete=complete,
                   byte_and_work_gate=complete and connector_gate(cfg['name']))
        if complete and reference is not None:
            row['output'] = evidence.compare(reference, cfg['name'])
            row['prompt_state'] = compare_states(reference, cfg['name'])
        if reference is None:
            reference = cfg['name']
        row['passed'] = bool(row['byte_and_work_gate']
            and row.get('output', {'passed': True})['passed']
            and row.get('prompt_state', {'exact': True})['exact'])
        ledger.append(row)
        write(ROOT / 'state' / f'extended_state_qualification_{model}.json', ledger)
        if not row['passed']:
            receipt = ROOT / 'raw' / cfg['name'] / 'dispatcher_result.json'
            if receipt.exists() and json.loads(receipt.read_text()).get('status') == 'NOT_RUN_TIME_BUDGET':
                return dict(status='NOT_RUN_TIME_BUDGET', results=ledger)
            return dict(status='BLOCKED_CORRECTNESS', results=ledger)
    return dict(status='PASSED_32_REQUEST_STATE_GATE', results=ledger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=('ouro', 'huginn'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    main(args.model, args.run)
