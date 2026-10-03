from pathlib import Path
from poc_speculative_moe_ep.contract import write, sha

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MASK = '2,3,6,7'
GPUS = (2,3,6,7)
PYTHON = '/home/esjung/.venvs/llada2-flash-sglang-053/bin/python'
DINFER = Path('/home/esjung/external/dinfer-llada2-flash-poc')
REVISION = 'dad945cac317da394b390f82c7b40691d8a881ed'
MODEL = Path('/home/esjung/.cache/huggingface/hub/models--inclusionAI--LLaDA2.0-mini/snapshots') / REVISION
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_moe_ep_comm_amplification_spec.md')
UUIDS = ('24200107-8a7f-de46-1bc8-b81f8d3af13e','17488c15-2d4c-5d9e-d503-29b0d959a8a8',
         'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','4cc26b88-19fc-1988-f9e0-17858aa7a99b')
