"""Paths and immutable experiment constants. No model substitute is permitted."""
from pathlib import Path
import hashlib
import json
import os
import time

REPO = Path('/home/esjung/MLLM-EP-github')
DEFAULT_ROOT = REPO / 'poc_flashvep/reports/looped_moe_headroom'
ROOT = Path(os.environ.get('LOOPED_MOE_RESULTS',str(DEFAULT_ROOT)))
SPEC = REPO / 'poc_flashvep/reports/Looped_MoE_Expert_Efficiency Headroom_spec.md'
if not SPEC.exists():
    SPEC = REPO / 'poc_flashvep/reports/spec.md'
PRIOR = DEFAULT_ROOT if ROOT!=DEFAULT_ROOT else REPO / 'poc_flashvep/reports/ouro_huginn_adaptive_g_mixed_prefill_discovery'
ASSETS = Path('/home/esjung/looped-moe-headroom-assets')
PYTHON = Path('/home/esjung/.venvs/looped-moe-headroom-py312/bin/python')
XLLM = Path('/home/esjung/external/xllm-loop-ep-poc')
MODELS = {
    'ifm_ouro': 'IFM/LoopedLM-P1-moe-ouro-500b',
    'ifm_huginn': 'IFM/LoopedLM-P1-moe-huginn-500b',
    'foil1': 'ShourenWSR/how-to-loop-moe',
}
SEED = 20261026

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')

def record(event, **fields):
    append(ROOT / 'progress.jsonl', dict(time=time.time(), event=event, **fields))
