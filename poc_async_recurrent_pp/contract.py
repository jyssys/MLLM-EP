"""Immutable scope for the PARPP experiment."""
from pathlib import Path
from poc_looped_multigpu_kv_placement.contract import sha, write, GPUS, UUIDS
import json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ASSETS = Path('/home/esjung/external/async-recurrent-pp-assets')
REVISION = '3aaa2224253a92ca45cf2e3d427c360e1ef9c93d'
MODEL = Path('/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B-Thinking/snapshots') / REVISION
PYTHON = '/home/esjung/.venvs/llada2-mini-reference/bin/python'
MASK = '4,5,6,7'


def read(path):
    return json.loads(Path(path).read_text())
