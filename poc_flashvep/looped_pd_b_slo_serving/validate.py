"""Audit saved evidence without changing inference, raw data, or source pins."""
import hashlib
import json
import math
import subprocess
from pathlib import Path
import zipfile
import numpy as np

from .common import ROOT, REPO, MODEL_PINS, write
from .jobs import output_rows
from .analyze import save


def main():
    frozen = json.loads((ROOT / 'configs/heldout_runtime_freeze.json').read_text())['files']
    records = []
    for path in sorted((ROOT / 'configs/jobs').glob('*.json')):
        job = json.loads(path.read_text())
        folder = ROOT / 'raw' / job['name']
        status_path = folder / 'status.json'
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())['status']
        rows = output_rows(job['name'])
        record = dict(run=job['name'], status=status, completed_rows=len(rows), planned=job['count'])
        record['execution_epoch'] = job.get('execution_epoch', 'original')
        record['explicit_runtime_override'] = job.get('env', {}).get('B_REPAIR_VARIANT', 'NONE')
        if record['explicit_runtime_override'] != 'NONE':
            overrides = [json.loads(p.read_text()) for p in folder.glob('repair_override_*.json')]
            record['override_metadata_available'] = bool(overrides) if status == 'COMPLETE' else 'PARTIAL_OR_FAILED'
            record['override_variant_matches_contract'] = all(value['variant'] == record['explicit_runtime_override']
                                                              for value in overrides)
        if job.get('reference'):
            record['scope'] = 'Official HF diagnostic. No RLT depth trace is required.'
        else:
            record['unique_sample_ids'] = len({row['sample_id'] for row in rows}) == len(rows)
            record['completed_count_matches_plan'] = len(rows) == job['count'] if status == 'COMPLETE' else 'PARTIAL_OR_FAILED'
            record['full_fixed_output_depth'] = all(len(row['exit_depths']) == row['output_tokens']
                                                   and all(value == MODEL_PINS[job['model']][2] for value in row['exit_depths'])
                                                   for row in rows)
            record['correct_model_revision'] = all(row['model_revision'] == MODEL_PINS[job['model']][1] for row in rows)
            record['TPOT_matches_observed_first_last_tokens'] = all(
                abs(row['tpot_ms'] - (1000 * (row['token_events'][-1]['time'] - row['token_events'][0]['time'])
                                     / (row['output_tokens'] - 1) if row['output_tokens'] >= 2 else 0.)) < 1e-6
                for row in rows if row.get('token_events'))
            expected_gpus = ','.join(map(str, job['gpus']))
            record['GPU_mapping_matches_contract'] = all(row['gpus'] == expected_gpus for row in rows)
            snapshot = folder / 'followup_sources.zip'
            if snapshot.exists():
                with zipfile.ZipFile(snapshot) as archive:
                    for name in ('worker.py', 'common.py'):
                        expected = frozen['poc_flashvep/looped_pd_b_slo_serving/' + name]
                        record[name + '_matches_heldout_freeze'] = hashlib.sha256(archive.read(name)).hexdigest() == expected
                # Qualification predates the last metadata-only freeze.
                record['freeze_test_applies'] = job.get('rate', 0) > 0 and job.get('workload') != 'calibration'
        telemetry = [json.loads(p.read_text()) for p in folder.glob('async_evidence_summary_*.json')]
        record['telemetry_processes_with_summary'] = len(telemetry)
        record['telemetry_error_free'] = all(value.get('error') is None for value in telemetry)
        record['telemetry_all_queued_records_written'] = all(value['enqueued_records'] == value['background_written_records'] for value in telemetry)
        records.append(record)
    save('evidence_validation.csv', records)
    issues = []
    for row in records:
        mandatory = ['unique_sample_ids', 'full_fixed_output_depth', 'correct_model_revision', 'GPU_mapping_matches_contract',
                     'TPOT_matches_observed_first_last_tokens',
                     'telemetry_error_free', 'telemetry_all_queued_records_written']
        if row['status'] == 'COMPLETE':
            mandatory.append('completed_count_matches_plan')
            if row.get('explicit_runtime_override') != 'NONE':
                mandatory += ['override_metadata_available', 'override_variant_matches_contract']
        if row.get('freeze_test_applies'):
            mandatory += ['worker.py_matches_heldout_freeze', 'common.py_matches_heldout_freeze']
        issues += [dict(run=row['run'], invariant=key) for key in mandatory if row.get(key) is False]
    current_source = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected for name, expected in frozen.items()}
    prior = json.loads((ROOT / 'configs/prior_source_hashes.json').read_text())
    prior_source = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected for name, expected in prior.items()}
    freeze = json.loads((ROOT / 'configs/heldout_runtime_freeze.json').read_text())
    patch = subprocess.check_output(['git', 'diff', '--binary'], cwd='/home/esjung/external/looped-pd-abc-rlt')
    patch_matches = hashlib.sha256(patch).hexdigest() == freeze['runtime_patch_SHA256']
    issues += [dict(run='GLOBAL', invariant='inference_source:' + name) for name, same in current_source.items() if not same]
    issues += [dict(run='GLOBAL', invariant='prior_source:' + name) for name, same in prior_source.items() if not same]
    if not patch_matches:
        issues.append(dict(run='GLOBAL', invariant='private_RLT_patch_matches_freeze'))
    slo_validation = {}
    slo_path = ROOT / 'configs/slo_lock.json'
    if slo_path.exists():
        locked = json.loads(slo_path.read_text())
        for model in ('ouro', 'huginn'):
            baselines = [output_rows(f'{model}_calibration_{arm}') for arm in ('DP_Mixed', 'PD_G1')]
            expected = dict(TTFT_ms=math.ceil(1.25 * max(np.percentile([r['ttft_ms'] for r in data], 90) for data in baselines)),
                            TPOT_ms=math.ceil(1.25 * max(np.percentile([r['tpot_ms'] for r in data], 90) for data in baselines)))
            same = all(locked[model][key] == value for key, value in expected.items())
            slo_validation[model] = dict(expected_from_disjoint_baseline_outputs=expected, matches_lock=same,
                                         recorded_creation_epoch=locked[model]['created'])
            if not same:
                issues.append(dict(run='GLOBAL', invariant=model + '_SLO_lock_matches_calibration_outputs'))
    write(ROOT / 'configs/final_evidence_validation.json', dict(
        run_count=len(records), issues=issues, inference_source_matches_freeze=current_source,
        prior_source_preserved=prior_source, private_RLT_patch_matches_freeze=patch_matches,
        SLO_validation=slo_validation,
        scope='Saved-evidence integrity and recorded fixed-depth checks. This does not prove original-model fidelity or answer accuracy.'))
    return issues


if __name__ == '__main__':
    main()
