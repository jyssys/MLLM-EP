"""Decode-first DP discovery. Checkpoint and runtime identity are explicit."""
from pathlib import Path
import json
from poc_looped_multigpu_kv_placement.contract import GPUS,UUIDS,sha,write

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
ASSETS=Path('/home/esjung/external/decode-first-depth-dp-assets')
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/decode_first_depth_aware_dp_spec.md')
MODEL_ID='ByteDance/Ouro-1.4B'
REVISION='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
MODEL=Path('/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots')/REVISION
THINKING_REVISION='3aaa2224253a92ca45cf2e3d427c360e1ef9c93d'
REFERENCE_PYTHON='/home/esjung/.venvs/llada2-mini-reference/bin/python'
PYTHON='/home/esjung/.venvs/cp-composition-vllm026/bin/python'
CDB=Path('/home/esjung/external/decode-first-cdb-upstream')
CDB_REVISION='abdfae0adbdd12be8c6123ccb34eee98a38afb39'
RLT=Path('/home/esjung/external/decode-first-vllm-rlt-upstream')
RLT_REVISION='9d601ceaf9af63121c742df5f584467e23630a0b'
MASK='4,5,6,7'


def read(path):
    return json.loads(Path(path).read_text())
