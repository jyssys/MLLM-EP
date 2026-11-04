"""Run bounded online pilots only with measured costs and exact connector gates."""
import argparse
import copy
from .common import *
from .recovery_qualification import save
from .run import execute


def cost_table(model, axis, cfg):
    from .dynamic_adapter import SIGNATURE
    ledger = json.loads((ROOT / 'state' / f'method_calibration_{model}.json').read_text())
    assert ledger['status'] == 'CALIBRATION_COMPLETE'
    rows = [row for row in ledger['results'] if row['axis'] == axis]
    assert len(rows) == 2 and all(row['passed'] for row in rows)
    from .static_screen import metrics
    reference = f'static_screen_v2_{model}_atomic_native32'
    baseline = metrics(reference)
    assert baseline['Decode_SLO_screen'], 'The baseline must first satisfy its Decode contract'
    folder = ROOT / 'raw' / reference
    observations = [json.loads(line) for line in (folder / 'queue_observations.jsonl').read_text().splitlines()]
    peak = max(row['pending_phases'].get('decode', 0) for row in observations)
    assert peak > 0
    evidence = {str(folder / 'requests.jsonl'): digest(folder / 'requests.jsonl'),
                str(folder / 'queue_observations.jsonl'): digest(folder / 'queue_observations.jsonl')}
    costs = []
    for row in rows:
        calibration = json.loads((ROOT / 'configs' / (row['name'] + '.json')).read_text())
        key = calibration['cut'] if axis == 'loop' else calibration['prefix_tokens']
        value = row['cost']
        costs.append(dict(key=key, p_service_s=value['P_host_service_p95_s'],
            d_continuation_s=value['D_acceptance_to_first_token_max_s'] if row['partial'] else 0.,
            transfer_exposed_s=value['exposed_handoff_p95_s'], d_extra_blocks=0))
        raw = ROOT / 'raw' / row['name']
        for path in [raw / 'method_calibration.json', raw / 'requests.jsonl', raw / 'closed_stats.json',
                     ROOT / 'configs' / (row['name'] + '.json'), *raw.glob('device_execution_*.json')]:
            evidence[str(path)] = digest(path)
    slo = baseline['frozen_SLO']
    table = dict(qualified=True, axis=axis,
        execution_signature={key: cfg.get(key) for key in SIGNATURE},
        TTFT_limit_s=slo['TTFT_ms'] / 1000, TPOT_margin_s=.9 * slo['TPOT_ms'] / 1000,
        ITL_limit_s=slo['ITL_ms'] / 1000,
        safe_decode_batch=min(cfg['D_slots'], peak),
        baseline_token_interval_s=baseline['ITL_p95_ms'] / 1000,
        costs=costs, calibration_evidence=evidence,
        scope='Finite measured cost estimates. Not an oracle or a concurrent SLO guarantee.',
        capacity_rule='Reject admission at the observed baseline peak Decode count. Also apply current token age, recent intervals, and native full-KV reservation.',
        extra_blocks_scope='The current request already reserves its full native KV allocation. Common workspace and graph allocation remain in every arm.',
        D_cost_scope='Maximum isolated acceptance-to-first-token host enclosure, including Coda and delivery. This is a conservative finite estimate, not pure GPU time.',
        tuning_scope='One qualified partial boundary plus Full-PD per axis. This is not a complete cut-space optimization.')
    path = ROOT / 'configs' / f'measured_dynamic_cost_{model}_{axis}.json'
    if path.exists():
        assert json.loads(path.read_text()) == table
    else:
        write(path, table)
    return path


def main(model):
    initialize()
    from .conditional_split_qualification import recovery_ready
    assert recovery_ready(json.loads((ROOT / 'state/huginn_final_recovery_gate.json').read_text()))
    from .static_qualification import connector_gate
    from .state_comparison import compare
    from .static_screen import metrics
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    target = ROOT / 'state' / f'online_method_pilot_{model}.json'
    ledger = []
    for heavy in (True, False):
        for axis in ('token', 'loop'):
            prefix = 'static_gate_v4_' if heavy else 'static_screen_v2_'
            template = f'{prefix}{model}_{axis}_partial32'
            reference = f'{prefix}{model}_atomic_native32'
            cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (template + '.json')).read_text()))
            cfg.update(name=f'online_pilot_{model}_{axis}_{"state" if heavy else "clean"}32',
                dynamic=True, arm='Loop_Dynamic' if axis == 'loop' else 'Token_Dynamic',
                phase='ONLINE_EXACT_STATE_GATE' if heavy else 'ONLINE_CLEAN_N32_PILOT',
                state_probe=heavy, verify_transfer=heavy)
            with quiet_analysis(cfg['name'] + '_calibration_table'):
                path = cost_table(model, axis, cfg)
            cfg.update(dynamic_cost_table=str(path), dynamic_cost_table_SHA256=digest(path))
            save(cfg)
            complete = execute(cfg['name'])
            row = dict(name=cfg['name'], axis=axis, heavy=heavy, complete=complete, passed=False)
            if complete:
                with quiet_analysis(cfg['name'] + '_online_gate'):
                    row['output'] = evidence.compare(reference, cfg['name'])
                    row['work'] = evidence.full_work(cfg['name'])
                    if heavy:
                        row['connector'] = connector_gate(cfg['name'])
                        row['state'] = compare(reference, cfg['name'])
                    else:
                        row['metrics'] = metrics(cfg['name'])
                    accepted = [json.loads(line) for path in (ROOT / 'raw' / cfg['name']).glob('accepted_partitions_*.jsonl')
                                for line in path.read_text().splitlines()]
                    seeds = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
                    begin = json.loads((ROOT / 'raw' / cfg['name'] / 'measurement_begin.json').read_text())['time']
                    accepted = [record for record in accepted if record['seed'] in seeds and record['time'] >= begin]
                    row['accepted_requests'] = len(accepted)
                    row['actual_partial_requests'] = sum(record['partial'] for record in accepted)
                    row['assigned_D_recurrent_token_rows'] = sum(record['assigned_D_recurrent_token_rows'] for record in accepted)
                    row['passed'] = bool(row['output']['passed'] and row['work']['passed']
                        and len(accepted) == cfg['count']
                        and (not heavy or row['connector'] and row['state']['exact']))
            ledger.append(row)
            write(target, dict(status='IN_PROGRESS', results=ledger))
            if not row['passed']:
                write(target, dict(status='FAILED_OR_UNQUALIFIED', results=ledger))
                return
    write(target, dict(status='N32_ONLINE_PILOT_COMPLETE', results=ledger,
        limitation='One start per axis. Full fallback does not validate an unexecuted partial/fallback branch. No replicated serving advantage is established.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=('ouro', 'huginn'))
    main(parser.parse_args().model)
