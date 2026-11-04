"""Reproduce both original errors, then test the isolated origin candidate."""
import copy
from .common import *
from .run import execute


def template(model, label, count=8):
    initialize()
    path = PREVIOUS / 'configs' / f'static_gate_v4_{model}_{label}8.json'
    cfg = copy.deepcopy(json.loads(path.read_text()))
    cfg['parent_configuration'] = dict(path=str(path), SHA256=digest(path), publication=BASE)
    for key in ('extension', 'extension_confirmation', 'selection_scope', 'control_scope',
                'memory_control', 'rate_index', 'measurement_epoch', 'arrival_seed_base',
                'capacity_reference_file'):
        cfg.pop(key, None)
    data_path = PREVIOUS / 'data' / cfg['dataset']
    pool = json.loads(data_path.read_text())
    new_data = ROOT / 'data' / cfg['dataset']
    if new_data.exists():
        assert json.loads(new_data.read_text()) == pool
    else:
        write(new_data, pool)
    slo = json.loads((PREVIOUS / 'configs/frozen_SLO.json').read_text())
    destination = ROOT / 'configs/frozen_SLO.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == slo
    else:
        write(destination, slo)
    cfg.update(count=count, token_suffix_origin_candidate=label.startswith('token'),
               phase='TOKEN_ORIGIN_RECOVERY_EXACTNESS_NOT_LATENCY',
               gpus=[0, 1] if model == 'ouro' else [4, 5],
               data_scope=dict(pool_records=len(pool), measured_records=count,
                               cyclic_reuse=False, selection='Preserved pool prefix, with original token IDs and generation seeds.'))
    if count != 8:
        import random
        cfg.update(request_caps=[cfg['cap']] * count,
                   prompt_ids=[row['prompt_id'] for row in pool[:count]], arrival_seed=2026110401)
        rng = random.Random(cfg['arrival_seed'])
        cfg['arrival_offsets'] = [0.]
        for _ in range(count - 1):
            cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
    return cfg


def reusable_qualification(folder, cfg, reference):
    path = folder / 'qualification.json'
    receipt = folder / 'dispatcher_result.json'
    if not path.exists() or not receipt.exists():
        return None
    row = json.loads(path.read_text())
    if not (row.get('passed') and row.get('complete')
            and json.loads(receipt.read_text()).get('complete')):
        return None
    contract = folder / 'arc_contract.json'
    # The inherited worker replaces configuration.json with CLI arguments.
    # arc_contract.json retains the complete actual run contract.
    launch_config = contract if contract.exists() else folder / 'configuration.json'
    assert json.loads(launch_config.read_text()) == cfg, 'Immutable run configuration changed'
    if reference is not None and (row.get('output', {}).get('reference') != reference
                                 or row.get('state', {}).get('reference') != reference):
        return None
    return dict(row, reuse_scope='Previously completed immutable GPU run and its exact recorded qualification. No new GPU execution or new validation claim.')


def qualify(name, reference=None):
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    cached = reusable_qualification(ROOT / 'raw' / name, cfg, reference)
    if cached is not None:
        return cached
    from ..fluidtoken_loop_split.static_qualification import connector_gate
    from ..fluidtoken_loop_split.state_comparison import compare
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    complete = execute(name)
    row = dict(name=name, complete=complete, passed=False)
    if complete:
        with quiet_analysis(name + '_state_gate'):
            row['transfer_and_work'] = connector_gate(name)
            if reference:
                row['output'] = evidence.compare(reference, name)
                row['state'] = compare(reference, name)
            row['passed'] = bool(row['transfer_and_work']
                and row.get('output', {'passed': True})['passed']
                and row.get('state', {'exact': True})['exact'])
        write(ROOT / 'raw' / name / 'qualification.json', row)
    return row


def main():
    initialize()
    target = ROOT / 'state/initial_qualification.json'
    ledgers = {}
    # Keep both expected failures before activating the candidate in either model.
    for model in ('ouro', 'huginn'):
        cfg = template(model, 'token_partial')
        cfg.update(name=f'a_{model}_original_error8', token_suffix_origin_candidate=False,
                   phase='EXPECTED_GPU_ERROR_REPRODUCTION')
        save(cfg)
        complete = execute(cfg['name'])
        log = (ROOT / 'logs' / (cfg['name'] + '.log')).read_text()
        ledgers[model] = dict(reproduction=dict(name=cfg['name'], complete=complete,
            expected_error_reproduced=not complete and 'wavefront completed prompt chunks out of order' in log))
        write(target, dict(status='IN_PROGRESS', models=ledgers))
    for model in ('ouro', 'huginn'):
        rows = ledgers[model]
        for label in ('atomic_native', 'token_partial', 'loop_partial'):
            cfg = template(model, label)
            cfg.update(name=f'a_{model}_{label}_origin_v1_state8')
            save(cfg)
            row = qualify(cfg['name'], None if label == 'atomic_native'
                          else f'a_{model}_atomic_native_origin_v1_state8')
            rows[label] = row
            write(target, dict(status='IN_PROGRESS', models=ledgers))
            if not row['passed']:
                break
    write(target, dict(status='INITIAL_PHASE_FINISHED', models=ledgers,
        scope='Individual exactness gates, not clean latency or N32 qualification. A failed model does not skip the other model.'))


if __name__ == '__main__':
    main()
