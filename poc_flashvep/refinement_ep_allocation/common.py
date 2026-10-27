"""Frozen model identity and atomic experiment artifacts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'poc_flashvep/results/refinement_aware_ep_expert_allocation'
REVISION = 'dad945cac317da394b390f82c7b40691d8a881ed'
MODEL = Path('/home/esjung/.cache/huggingface/hub/models--inclusionAI--LLaDA2.0-mini/snapshots') / REVISION
REFERENCE = Path('/home/esjung/MLLM-EP-mini-repro/reproduction')
LEGACY_EP = Path('/home/esjung/MLLM-EP-dllm-comm-amplification')
DINFER = Path('/home/esjung/external/dinfer-llada2-flash-poc')
HF_PYTHON = '/home/esjung/.venvs/llada2-mini-reference/bin/python'
EP_PYTHON = '/home/esjung/.venvs/llada2-flash-sglang-053/bin/python'
UUIDS = ['GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d',
         'GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb',
         'GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e',
         'GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8',
         'GPU-6076e2f2-5b63-3761-5586-56ceb7df8139',
         'GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730',
         'GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28',
         'GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b']

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def snapshot_driver(path):
    path=Path(path)
    content=path.read_text()
    checksum=hashlib.sha256(content.encode()).hexdigest()
    target=ROOT/'sources'/f'{path.stem}.{checksum}.json'
    if not target.exists():
        write(target,dict(path=str(path),sha256=checksum,source_text=content))
    return dict(path=str(path),sha256=checksum,snapshot=str(target))

def reference_helpers():
    spec = importlib.util.spec_from_file_location('frozen_mini_reference', REFERENCE/'run_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def rows(manifest, allow_incomplete=False):
    lines=Path(manifest).read_text().splitlines(keepends=True)
    if allow_incomplete and lines and not lines[-1].endswith('\n'):
        lines=lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]

def setup():
    for name in ('state', 'logs', 'manifests', 'hf', 'ep4', 'sources', 'figures','archives'):
        (ROOT/name).mkdir(parents=True, exist_ok=True)
