"""Immutable primary geometry; no acceptance or placement changes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_intra_refinement_attn_moe_pipeline_spec.md')
from poc_dllm_ep_comm.contract import MASK, GPUS, UUIDS, MODEL, REVISION, DINFER, PYTHON


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
