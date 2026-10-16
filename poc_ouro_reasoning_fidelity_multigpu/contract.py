"""Pins and paths for this independent experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ASSETS = Path('/home/esjung/external/ouro-reasoning-fidelity-assets')
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_reasoning_fidelity_real_workload_multigpu_poc_spec.md')
PYTHON = '/home/esjung/.venvs/cp-composition-vllm026/bin/python'
CPU = '/home/esjung/.venvs/llada2-mini-reference/bin/python'
REVISION = 'f1edd81e7ac41355db670500ceaf204e0f73af68'
MODEL = Path('/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-2.6B-Thinking/snapshots') / REVISION
GPUS = [4, 5, 6, 7]
UUIDS = ['6076e2f2-5b63-3761-5586-56ceb7df8139', 'a1a1cfcf-93a1-3544-9a5e-e58144b68730',
         'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28', '4cc26b88-19fc-1988-f9e0-17858aa7a99b']
PREVIOUS = PROJECT / 'poc_looped_exact_depth_chunk_pipeline'
SEEDS = [41015, 41016, 41017]

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(chunk)
    return result.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def read(path):
    return json.loads(Path(path).read_text())

def file_id(value):
    return value.replace('/', '__')
