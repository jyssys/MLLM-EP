import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MASK = '2,3,6,7'
PYTHON = '/home/esjung/.venvs/flashvep-deepep-v020/bin/python'
TARGET_REV = 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
DRAFT_REV = 'c1899de289a04d12100db370d81485cdf75e47ca'
HUB = Path('/home/esjung/.cache/huggingface/hub')
TARGET = str(HUB/'models--Qwen--Qwen3-30B-A3B'/'snapshots'/TARGET_REV)
DRAFT = str(HUB/'models--Qwen--Qwen3-0.6B'/'snapshots'/DRAFT_REV)
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/speculative_decoding_moe_ep_communication_discovery_spec.md')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
