"""Freeze workers and retain the qualified exclusive GPU owner."""
import argparse
import subprocess
import zipfile
from .common import *


def execute(name):
    initialize()
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    try:
        for key in ('required_qualification', 'post_decode_reference'):
            if not cfg.get(key):
                continue
            gate = json.loads((ROOT / 'raw' / cfg[key] / 'qualification_with_frozen_reference.json').read_text())
            assert gate.get('campaign_gate_passed', gate['passed']), 'Required exact-state qualification did not pass'
            if cfg.get('child_exit_integrity_required'):
                receipts = [json.loads(path.read_text()) for path in
                    (ROOT/'raw'/cfg[key]).glob('native_worker_exit_*.json')]
                assert any(value['passed'] and {row['role'] for row in value['records']
                    if row['exitcode'] == 0 and not row['forced_cleanup']} == {'prefill', 'decode'}
                    for value in receipts), 'Required worker exit integrity did not pass'
    except (FileNotFoundError, AssertionError) as error:
        destination = ROOT / 'state/preflight_failures'
        destination.mkdir(exist_ok=True)
        write(destination / f'{name}_{time.time_ns()}.json', dict(name=name,
            status='NOT_RUN_REQUIRED_QUALIFICATION', error=repr(error), time=time.time(),
            configuration_sha256=digest(ROOT / 'configs' / (name+'.json'))))
        raise
    from ..native_split_deep_validation import run as inherited
    inherited.initialize = initialize
    original = subprocess.Popen

    def launch(command, *args, **kwargs):
        if isinstance(command, list) and 'poc_flashvep.looped_pd_headroom_recovery.worker' in command:
            if cfg.get('nsight_graph_enclosures'):
                command = [('--cuda-graph-trace=graph' if value == '--cuda-graph-trace=node' else value)
                    for value in command]
                command = [('--output='+str(ROOT / 'raw' / name / 'graph_enclosures')
                    if value.startswith('--output=') else value) for value in command]
            folder = ROOT / 'raw' / name
            snapshot = folder / 'worker_source'
            with zipfile.ZipFile(folder / 'new_campaign_source.zip') as archive:
                assert all(not p.startswith('/') and '..' not in p.split('/') for p in archive.namelist())
                archive.extractall(snapshot)
            reused = []
            for package in ('completion_aware_loop_pd', 'idle_intervention_dynamic_loop'):
                reused += sorted((REPO / 'poc_flashvep' / package).rglob('*.py'))
            with zipfile.ZipFile(folder / 'qualified_predecessor_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in reused:
                    archive.write(path, str(path.relative_to(REPO)))
                archive.extractall(snapshot)
            write(folder / 'qualified_predecessor_hashes.json', {str(p): digest(p) for p in reused})
            env = dict(kwargs['env'])
            connections = cfg.get('cuda_device_max_connections')
            if connections is not None:
                assert type(connections) is int and 1 <= connections <= 32
                env['CUDA_DEVICE_MAX_CONNECTIONS'] = str(connections)
            env['NET_BENEFIT_SNAPSHOT'] = str(snapshot)
            env['PYTHONPATH'] = str(snapshot / 'poc_flashvep/huginn_net_benefit/bootstrap') + os.pathsep + env['PYTHONPATH']
            kwargs['env'] = env
        return original(command, *args, **kwargs)

    subprocess.Popen = launch
    try:
        return inherited.execute(name)
    finally:
        subprocess.Popen = original
        if cfg.get('local_nsys'):
            from .profile_cleanup import cleanup
            cleanup(name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name')
    raise SystemExit(0 if execute(p.parse_args().name) else 1)
