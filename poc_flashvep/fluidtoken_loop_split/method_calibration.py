"""Measure isolated split costs before an optional online policy experiment."""
import argparse
import copy
from .common import *
from .recovery_qualification import save
from .run import execute


def summarize(name):
    from ..looped_pd_headroom_recovery.gates import percentile
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
    requests = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    requests = {row['seed']: row for row in requests if not row.get('warmup')}
    assert set(requests) == expected
    phases = json.loads((folder / 'closed_stats.json').read_text())['metrics']
    phases = {row['request_id']: row for row in phases}
    accepted = [json.loads(line) for path in folder.glob('accepted_partitions_*.jsonl')
                for line in path.read_text().splitlines()]
    begin = json.loads((folder / 'measurement_begin.json').read_text())['time']
    accepted = {row['seed']: row for row in accepted if row['seed'] in expected and row['time'] >= begin}
    assert set(accepted) == expected, 'Missing actual destination acceptance'
    p_service, exposed, d_enclosure = [], [], []
    for seed, request in requests.items():
        phase = phases[f"r{request['run']}_q{request['sample_id']}"]['timings']
        p_service.append(phase['prefill_compute_completed'] - phase['prefill_started'])
        exposed.append(phase['d_received'] - phase['prefill_compute_completed'])
        d_enclosure.append(request['first_token'] - accepted[seed]['time'])
    assert min(p_service) > 0 and min(exposed) >= 0 and min(d_enclosure) >= 0
    device_paths = list(folder.glob('device_execution_*.json'))
    assert device_paths, 'CUDA event evidence is mandatory for this calibration'
    devices = [json.loads(path.read_text()) for path in device_paths]
    assert {row['role'] for row in devices} == {'prefill', 'decode'}
    result = dict(name=name, samples=len(requests),
        P_host_service_p95_s=percentile(p_service, .95),
        exposed_handoff_p95_s=percentile(exposed, .95),
        D_acceptance_to_first_token_max_s=max(d_enclosure),
        P_host_service_samples_s=p_service, exposed_handoff_samples_s=exposed,
        D_acceptance_to_first_token_samples_s=d_enclosure,
        evidence='MEASURED_ISOLATED_HOST_ENCLOSURES_WITH_CUDA_EVENT_EVIDENCE',
        scope='P service excludes admission waiting but includes launch and scheduler gaps. The D upper envelope includes continuation, Coda, sampling, and delivery. These are not exclusive GPU times.',
        caveat='An isolated maximum is a finite measured cost estimate, not a bound on concurrent serving.')
    write(folder / 'method_calibration.json', result)
    return result


def main(model):
    initialize()
    from .conditional_split_qualification import recovery_ready
    assert recovery_ready(json.loads((ROOT / 'state/huginn_final_recovery_gate.json').read_text()))
    qualification = json.loads((ROOT / 'state/conditional_split_qualification.json').read_text())
    assert qualification['extended_results'][model]['status'] == 'PASSED_32_REQUEST_STATE_GATE'
    source_root = ROOT if model == 'huginn' else PREVIOUS
    source_name = 'recovery_huginn_admitted_probe_clean512' if model == 'huginn' else 'd_graph_ouro_state_verified_matched128'
    source = json.loads((source_root / 'configs' / (source_name + '.json')).read_text())
    data = json.loads((source_root / 'data' / source['dataset']).read_text())[96:104]
    assert len(data) == 8
    dataset = f'method_cost_{model}_pool96_104.json'
    target = ROOT / 'data' / dataset
    if target.exists():
        assert json.loads(target.read_text()) == data
    else:
        write(target, data)
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    ledger = []
    reference = None
    for axis, partial in (('loop', False), ('token', False), ('loop', True), ('token', True)):
        template = f'static_screen_v2_{model}_{axis}_partial32' if partial else f'static_screen_v2_{model}_staged_snapshot_full32'
        cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (template + '.json')).read_text()))
        cfg.update(name=f'method_cost_{model}_{axis}_{"partial" if partial else "full"}8',
            axis=axis, arm='Loop_Static' if axis == 'loop' else 'Token_Static',
            dataset=dataset, seed=2026112401, count=8, arrival_offsets=[0.] * 8,
            arrival_rate=None, arrival_seed=None, calibration_serial=True, request_caps=[source['cap']] * 8,
            prompt_ids=[row['prompt_id'] for row in data], state_probe=False, verify_transfer=False,
            observe_device_execution=True, phase='ISOLATED_METHOD_CALIBRATION_NOT_CLEAN_SERVING',
            timeout=900, estimated_seconds=180,
            data_scope='Eight preserved records 96:104, disjoint from the first-32 method screen. These records appeared in recovery, not an unseen dataset.')
        if not partial:
            cfg.update(cut=cfg['depth'], prefix_tokens=cfg['length'])
        save(cfg)
        complete = execute(cfg['name'])
        row = dict(name=cfg['name'], axis=axis, partial=partial, complete=complete, passed=False)
        if complete:
            with quiet_analysis(cfg['name'] + '_cost_check'):
                row['work'] = evidence.full_work(cfg['name'])
                row['output'] = evidence.compare(reference, cfg['name']) if reference else dict(passed=True)
                if row['work']['passed'] and row['output']['passed']:
                    row['cost'] = summarize(cfg['name'])
                    row['passed'] = True
        reference = reference or cfg['name']
        ledger.append(row)
        write(ROOT / 'state' / f'method_calibration_{model}.json', dict(status='IN_PROGRESS', results=ledger))
        if not row['passed']:
            write(ROOT / 'state' / f'method_calibration_{model}.json', dict(status='FAILED_OR_UNQUALIFIED', results=ledger))
            return
    write(ROOT / 'state' / f'method_calibration_{model}.json', dict(status='CALIBRATION_COMPLETE', results=ledger))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=('ouro', 'huginn'))
    main(parser.parse_args().model)
