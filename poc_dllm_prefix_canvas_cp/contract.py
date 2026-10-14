import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
MASK='4,5,6,7'
GPUS=[4,5,6,7]
UUIDS=['6076e2f2-5b63-3761-5586-56ceb7df8139','a1a1cfcf-93a1-3544-9a5e-e58144b68730',
       'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','4cc26b88-19fc-1988-f9e0-17858aa7a99b']
PYTHON='/home/esjung/.venvs/llada2-flash-sglang-053/bin/python'
DINFER=Path('/home/esjung/external/dinfer-llada2-flash-poc')
REVISION='dad945cac317da394b390f82c7b40691d8a881ed'
MODEL=Path('/home/esjung/.cache/huggingface/hub/models--inclusionAI--LLaDA2.0-mini/snapshots')/REVISION
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_prefix_canvas_cp_poc_spec.md')
PREFIXES=[512,1024,2048,4096,8192,16384]
MAX_CONTEXT=32768
BLOCK=32
OUTPUT=64


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
