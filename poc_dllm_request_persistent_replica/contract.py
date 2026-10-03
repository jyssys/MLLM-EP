import os
from pathlib import Path
from poc_dllm_ep_comm.contract import MASK,GPUS,PYTHON,DINFER,MODEL,REVISION,UUIDS,write,sha
from poc_dllm_block_local_replica.contract import EXPERT_BYTES,ROW_DC_BYTES,BUDGETS

PACKAGE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('DLLM_PERSIST_ARTIFACT_ROOT',str(PACKAGE)))
PROJECT=PACKAGE.parent
SHORT=PROJECT/'poc_dllm_block_local_replica'
COMM=PROJECT/'poc_dllm_ep_comm'
PARENT='90fcaa039d609e5c3f2c8e563448c54874673597'
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_request_persistent_replica_spec.md')
if not SPEC.exists():SPEC=PACKAGE/'references/authoritative_spec.md'


def new_json(path,data):
    if path.exists():raise FileExistsError(path)
    write(path,data)


def summary_path():
    latest=ROOT/'results/summary_v3.json'
    if latest.exists():return latest
    audited=ROOT/'results/summary_v2.json'
    return audited if audited.exists() else ROOT/'results/summary.json'
