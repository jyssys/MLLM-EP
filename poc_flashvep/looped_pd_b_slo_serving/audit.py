"""Read-only inventory. Write evidence only in the follow-up output tree."""
import hashlib
import subprocess
from pathlib import Path

from .common import ROOT, REPO, MODEL_PINS, START, HARD_STOP, GPU_STOP, setup, write


def main():
    setup()
    commands = {
        'GPU_list': ['nvidia-smi', '-L'],
        'topology': ['nvidia-smi', 'topo', '-m'],
        'GPU_processes': ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'],
        'GPU_clocks': ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,memory.total', '--format=csv,noheader'],
        'CPU': ['lscpu'],
        'RAM': ['free', '-b'],
    }
    data = {}
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True)
        data[name] = dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    for field, directory, command in [('RLT_patch', '/home/esjung/external/looped-pd-abc-rlt', ['git', 'diff', '--binary']),
                                       ('RLT_upstream', '/home/esjung/external/looped-pd-abc-rlt', ['git', 'rev-parse', 'HEAD'])]:
        value = subprocess.check_output(command, cwd=directory)
        data[field] = dict(SHA256=hashlib.sha256(value).hexdigest()) if field == 'RLT_patch' else value.decode().strip()
    data.update(model_pins={name: MODEL_PINS[name] for name in ('ouro', 'huginn')},
                budget=dict(start=START, hard_stop=HARD_STOP, GPU_stop=GPU_STOP),
                note='Snapshot after parallel qualification and during baseline calibration. It is not a claim of an otherwise idle shared host.')
    write(ROOT / 'configs/environment_inventory.json', data)
    prior = REPO / 'poc_flashvep/looped_pd_abc'
    write(ROOT / 'configs/prior_source_hashes.json', {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in prior.glob('*.py')})


if __name__ == '__main__':
    main()
