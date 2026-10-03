import os
from pathlib import Path
from poc_dllm_ep_comm.contract import MASK, GPUS, PYTHON, DINFER, MODEL, REVISION, UUIDS, write, sha

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DLLM_REPLICA_ARTIFACT_ROOT', str(PACKAGE)))
PROJECT = PACKAGE.parent
SOURCE = PROJECT / 'poc_dllm_ep_comm'
SOURCE_COMMIT = 'dc539d3222193ab0902e0a342679edc3610f54bb'
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_block_local_ep_replication_spec.md')
if not SPEC.exists():
    SPEC = PACKAGE/'references/authoritative_spec.md'
EXPERT_BYTES = 6291456
ROW_DC_BYTES = 8192
BUDGETS = (.5, 1., 2., 4.)


def new_json(path, data):
    if path.exists():
        raise FileExistsError(path)
    write(path, data)


def final_summary_path():
    return ROOT/'results'/('summary_v2.json' if (ROOT/'results/summary_v2.json').exists() else 'summary.json')
