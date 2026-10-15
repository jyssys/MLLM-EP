"""Pinned four-family experiment scope. No installed engine edits."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
ASSETS=Path('/home/esjung/external/looped-recurrence-tp-assets')
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_recurrence_tp_scaling_spec_v2_four_models.md')
PYTHON='/home/esjung/.venvs/cp-composition-vllm026/bin/python'
GPUS=[4,5,6,7]
UUIDS=['6076e2f2-5b63-3761-5586-56ceb7df8139','a1a1cfcf-93a1-3544-9a5e-e58144b68730',
       'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','4cc26b88-19fc-1988-f9e0-17858aa7a99b']
MODELS={
 'ouro':dict(repo='ByteDance/Ouro-2.6B-Thinking',revision='f1edd81e7ac41355db670500ceaf204e0f73af68'),
 'qwen3':dict(repo='Qwen/Qwen3-1.7B',revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
 'qwen25':dict(repo='Qwen/Qwen2.5-3B',revision='3aab1f1954e9cc14eb9509a215f9e5ca08227a9b')}
PREFIXES=[1024,16384,32704]
PARITY=dict(policy='BITWISE_IDENTICAL_SAME_TP_AND_R',logits_atol=0,KV_atol=0,
    fallback='CONTROL_BLOCKED if a genuine difference remains. No tolerance relaxation after timing.')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def model_path(name):
    value=MODELS[name]
    return Path('/home/esjung/.cache/huggingface/hub')/('models--'+value['repo'].replace('/','--'))/'snapshots'/value['revision']
