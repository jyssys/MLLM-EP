"""Frozen device, model, and workload boundaries for this experiment."""
import gzip
import hashlib
import json
from pathlib import Path
from poc_dllm_prefix_canvas_cp.contract import MASK,GPUS,UUIDS,MODEL,REVISION,DINFER,PYTHON

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_multigpu_e2e_critical_path_spec.md')
PREFIXES=[512,4096,8192,16384]
BLOCK=32
OUTPUT=64
SEED=19411


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def rows(split='calibration'):
    with gzip.open(PROJECT/'poc_dllm_prefix_canvas_cp/results/frozen_tokens.json.gz','rt') as stream:
        return [r for r in json.load(stream) if r['split']==split]


def token_ids(row,prefix):
    ids=row['head']+row['background'][:prefix-len(row['head'])-len(row['tail'])]+row['tail']
    assert len(ids)==prefix and prefix+OUTPUT<=32768
    return ids
