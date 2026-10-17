"""Exact R4 discovery scope. No approximate recurrence or serving engine."""
from pathlib import Path
from poc_looped_multigpu_kv_placement.contract import GPUS,UUIDS,sha,write
from poc_async_recurrent_pp.contract import MODEL,REVISION,PYTHON
import json

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
ASSETS=Path('/home/esjung/external/recurrent-asymmetry-assets')
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/recurrent_memory_compute_asymmetry_discovery_spec.md')
MASK='4,5,6,7'
VLLM_PYTHON='/home/esjung/.venvs/cp-composition-vllm026/bin/python'


def read(path):
    return json.loads(Path(path).read_text())
