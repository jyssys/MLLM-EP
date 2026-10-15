"""Pinned experiment scope. No candidate before the native measurement gate."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
GPUS=[4,5,6,7]
MASK='4,5,6,7'
UUIDS=['6076e2f2-5b63-3761-5586-56ceb7df8139','a1a1cfcf-93a1-3544-9a5e-e58144b68730',
    'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','4cc26b88-19fc-1988-f9e0-17858aa7a99b']
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_multigpu_kv_placement_poc_spec.md')
VLLM_PYTHON='/home/esjung/.venvs/cp-composition-vllm026/bin/python'
REFERENCE_PYTHON='/home/esjung/.venvs/llada2-mini-reference/bin/python'
ASSETS=Path('/home/esjung/external/looped-kv-placement-assets')
MODELS={
    'ouro':dict(repo='ByteDance/Ouro-2.6B-Thinking',revision='f1edd81e7ac41355db670500ceaf204e0f73af68'),
    'huginn':dict(repo='tomg-group-umd/huginn-0125',revision='bb6621b65e90b6a4b9b29ef88dc83866d450470c')}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def model_path(name):
    d=MODELS[name]
    return Path('/home/esjung/.cache/huggingface/hub')/('models--'+d['repo'].replace('/','--'))/'snapshots'/d['revision']
