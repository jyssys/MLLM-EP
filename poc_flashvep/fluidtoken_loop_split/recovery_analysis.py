"""Audit one shared interval without importing another run's GPU evidence."""
import argparse
from .common import *


def analyze(name):
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_autonomous_headroom.aligned_gates import one
    from ..looped_pd_autonomous_headroom.ready_queue_audit import audit
    from ..looped_pd_headroom_recovery import gates
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    # Preserve the previous matched reference as external, hashed evidence.
    reference_name = 'capacity_short_huginn_L2048_v2'
    reference_path = PREVIOUS / 'raw' / reference_name / 'cohorts.jsonl'
    source = json.loads((PREVIOUS / 'configs' / (reference_name + '.json')).read_text())
    for key in ('model', 'model_revision', 'length', 'depth', 'chunk',
                'P_prefill_graph_rows', 'token_budget'):
        assert source.get(key) == cfg.get(key), ('Unmatched service reference', key)
    cohorts = [json.loads(line) for line in reference_path.read_text().splitlines()]
    reference = max(row['job']['batch'] / row['prefill_host_service_s']
                    for row in cohorts if row.get('status') == 'COMPLETE')
    references = [dict(path=str(reference_path), SHA256=digest(reference_path),
                       requests_per_second=reference, scope='Prior finite matched reference')]
    refreshed = ROOT / 'raw/capacity_recovery_huginn_L2048/cohorts.jsonl'
    finished = refreshed.parent / 'dispatcher_result.json'
    if finished.exists() and json.loads(finished.read_text()).get('complete'):
        new_cfg = json.loads((ROOT / 'configs/capacity_recovery_huginn_L2048.json').read_text())
        for key in ('model', 'model_revision', 'length', 'depth', 'chunk',
                    'P_prefill_graph_rows', 'token_budget'):
            assert new_cfg.get(key) == cfg.get(key), ('Unmatched refreshed reference', key)
        values = [json.loads(line) for line in refreshed.read_text().splitlines()]
        new_rate = max(row['job']['batch'] / row['prefill_host_service_s']
                       for row in values if row.get('status') == 'COMPLETE')
        references.append(dict(path=str(refreshed), SHA256=digest(refreshed),
                               requests_per_second=new_rate, scope='Refreshed finite matched reference'))
        reference = max(reference, new_rate)
    if cfg.get('route_campaign_evidence_async'):
        name_ref = 'capacity_recovery_huginn_async_evidence_L2048'
        path_ref = ROOT / 'raw' / name_ref / 'cohorts.jsonl'
        done_ref = json.loads((path_ref.parent / 'dispatcher_result.json').read_text())
        assert done_ref.get('complete'), 'Updated runtime requires its completed finite reference'
        cfg_ref = json.loads((ROOT / 'configs' / (name_ref + '.json')).read_text())
        for key in ('model', 'model_revision', 'length', 'depth', 'chunk',
                    'P_prefill_graph_rows', 'token_budget', 'route_campaign_evidence_async'):
            assert cfg_ref.get(key) == cfg.get(key), ('Unmatched updated reference', key)
        records = [json.loads(line) for line in path_ref.read_text().splitlines()]
        rate = max(row['job']['batch'] / row['prefill_host_service_s']
                   for row in records if row.get('status') == 'COMPLETE')
        references.append(dict(path=str(path_ref), SHA256=digest(path_ref),
                               requests_per_second=rate, scope='Updated evidence-routing runtime finite reference'))
        reference = max(reference, rate)
    inherited = gates.actual_P_reference
    gates.actual_P_reference = lambda config: reference
    try:
        with quiet_analysis(name + '_joint_audit'):
            result = one(cfg)
            queue = audit(cfg, result)
    finally:
        gates.actual_P_reference = inherited
    probe_file = ROOT / 'raw' / name / 'D_probe_requests.jsonl'
    probes = [json.loads(line) for line in probe_file.read_text().splitlines()] if probe_file.exists() else []
    central = [row for row in probes if result['observation_start'] <= row['time']
               <= row['finish'] <= result['observation_end']]
    output = dict(gates=result, queue=queue, central_completed_probes=len(central),
                  reference=dict(sources=references, requests_per_second=reference,
                                 scope='Strongest recorded matched finite reference. Not sustainable serving capacity.'),
                  status='SCREEN_ONLY_CORRECTNESS_AND_REPETITIONS_PENDING')
    target = ROOT / 'raw' / name / 'recovery_joint_screen.json'
    if target.exists() and json.loads(target.read_text()) != output:
        import time
        target = target.with_name('recovery_joint_screen_' + str(time.time_ns()) + '.json')
    write(target, output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    print(json.dumps(analyze(parser.parse_args().name)))
