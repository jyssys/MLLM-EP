import json
import gzip
import os
from pathlib import Path

MASK = '0,1,4,5'
MLLM = '/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c'
LLM = '/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
ROOT = Path(__file__).resolve().parent
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/dispatch_combine_asymmetry_discovery_spec_gpu0145.md')


def load(path):
    """Published large accounting table is gzip; local generation stays JSON."""
    path=Path(path)
    if path.exists():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt') as stream:return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def geometry(cfg):
    cfg = cfg.get('text_config', cfg)
    keys = {'hidden_size': 2048, 'moe_intermediate_size': 768,
            'num_experts': 128, 'num_experts_per_tok': 8, 'num_hidden_layers': 48}
    for key, expected in keys.items():
        if cfg[key] != expected:
            raise ValueError((key, cfg[key], expected))
    return {key: cfg[key] for key in keys}
