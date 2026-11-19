"""Check completed work, preserved state and approved burner ownership."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .common import ROOT, SOURCE, REPO, QUIET_LOCK, PYTHON, digest, preserve, resources, write


def checked_json(relative):
    path = ROOT / relative
    return json.loads(path.read_text())


def validate_completed_work():
    for name in ('main_common_sweep_complete', 'followup_controls_complete',
                 'UCX_protocol_probe_complete', 'Native_clean_C1_confirmation_complete'):
        assert checked_json('state/' + name + '.json')['complete'], name
    assert not list((ROOT / 'state').glob('*hold*.json'))
    source = checked_json('state/runtime_source_freeze.json')['files']
    assert all(digest(SOURCE / name) == expected for name, expected in source.items())
    old = checked_json('state/prior_report_hashes.json')
    assert all(digest(REPO / 'poc_flashvep/reports' / name) == expected for name, expected in old.items())
    main = checked_json('analysis/Main_full_analysis.json')
    assert main['complete'] and len(main['per_start']) == 22
    assert all(row['metrics']['n'] == 200 for row in main['per_start'])
    assert all(all(c['outputs_equal'] and c['arrival_offsets_equal'] and c['dataset_equal']
                   for c in pair['checks']) for pair in main['paired_comparison'])
    assert checked_json('analysis/Main_runtime_audit.json')['all_B8192_S1024']
    sensitivity = checked_json('analysis/B_S_sensitivity_v1.json')['results']
    assert len(sensitivity) == 24
    assert all(r['valid'] for r in sensitivity if '_B8192_S1024_' in r['name'])
    profiles = [checked_json(f'analysis/Native_online_kernel_activity_q{q}_v1.json') for q in ('1', '2.35')]
    assert all(p['observer_gate_passed'] for p in profiles)
    assert checked_json('analysis/UCX_protocol_selection_and_teardown_audit.json')['all_recorded_context_errors_after_completion']
    assert digest(ROOT / 'data/longbench_v1_mixed200.json') == '1a9e6637bbd4f26e1338533d6735fa981c3ab258e0a59ac39c35b8b22d035553'
    return dict(main_server_starts=22, main_requests=4400, runtime_freeze_files=len(source),
                historical_report_hashes_preserved=len(old), sensitivity_cells=24,
                sensitivity_valid_cells=sum(r['valid'] for r in sensitivity),
                qualified_online_kernel_profiles=len(profiles))


def burners_checked():
    owner = resources()
    active = owner.processes()
    owned = owner.burners()
    assert all(os.getsid(r['pid']) == owned.get(r['gpu']) for r in active), active
    owner.restore(list(range(8)))
    samples = []
    for _ in range(3):
        time.sleep(5)
        owned = owner.burners()
        active = owner.processes()
        assert set(owned) == set(range(8))
        assert {r['gpu'] for r in active} == set(range(8))
        assert all(os.getsid(r['pid']) == owned[r['gpu']] for r in active)
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        values = [line.split(',') for line in output.splitlines()]
        assert len(values) == 8 and all(int(v[2].strip()) > 0 for v in values), output
        samples.append(dict(time=time.time(),owned_burners=owned,active=active,nvidia_smi=output))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-publication', action='store_true')
    args = parser.parse_args()
    preserve()
    checks = validate_completed_work()
    with QUIET_LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        samples = burners_checked()
        if args.post_publication:
            receipt = checked_json('state/publication_receipt.json')
            assert receipt['pushed'] and receipt['remote_matches']
            remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+receipt['branch']],
                                             cwd=REPO,text=True).split()[0]
            assert remote == receipt['SHA']
            write(ROOT / 'state/post_publication_final_check.json', dict(time=time.time(),
                checks=checks, remote_SHA=remote, approved_burner_samples=samples,
                checkout_index_external_runtime_preserved=True, passed=True))
        else:
            command = [str(PYTHON), '-m', 'unittest',
                       'poc_flashvep.dense_ar_native_reproduction.test_contracts',
                       'poc_flashvep.dense_ar_native_reproduction.test_transport_exactness',
                       'poc_flashvep.dense_ar_native_reproduction.test_online_profile']
            test = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
            assert test.returncode == 0, test.stderr
            shutil.copyfile(ROOT / 'burner.log', ROOT / 'state/burner_closeout_snapshot.log')
            write(ROOT / 'state/CAMPAIGN_CLOSED.json', dict(time=time.time(), status='CLOSED',
                checks=checks, tests=dict(command=command, returncode=test.returncode,
                    stdout=test.stdout, stderr=test.stderr), approved_burner_samples=samples,
                checkout_index_external_runtime_preserved=True,
                verdict='DATASET_PASS / MODEL_PATH_REPRODUCTION_PASS / CONCURRENT_TRANSPORT_PARTIAL / OFFLOADING_HOLD',
                scope='All scheduled Native/Agg jobs ended. No Layer/Token restart. Scientific holds remain explicit. Publication follows this resource closeout.'))
            write(ROOT / 'state/progress.json', dict(time=time.time(), status='GPU_CAMPAIGN_CLOSED_PUBLICATION_PENDING',
                  forbidden='Layer/Token and dynamic experiments remain on hold.'))
    preserve()
    print(json.dumps(dict(passed=True,checks=checks,post_publication=args.post_publication)))


if __name__ == '__main__':
    main()
