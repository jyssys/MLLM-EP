import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
UPSTREAM = Path('/home/esjung/external/Diffulex-multibd-poc')
UPSTREAM_SHA = '9d1a04e794351f18039b3234a10e8b07e15bcbbf'
MODEL_ID = 'SJTU-DENG-Lab/MBD-Code-LLaDA2-mini-16B'
MODEL_REVISION = '83434493ea9f026cbc139d681d7ae04ffefe052f'
PYTHON = '/home/esjung/.venvs/llada2-flash-sglang-053/bin/python'
# Explicit 2026-10-13 user override. Historical 2/3/6/7 evidence is retained;
# fresh matched confirmation is a separate hardware stratum.
MASK = '4,5,6,7'
GPUS = [4, 5, 6, 7]
UUIDS = ['6076e2f2-5b63-3761-5586-56ceb7df8139',
         'a1a1cfcf-93a1-3544-9a5e-e58144b68730',
         'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28',
         '4cc26b88-19fc-1988-f9e0-17858aa7a99b']
SOURCE_SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/multibd_moe_ep_poc_spec.md')
SPEC = SOURCE_SPEC if SOURCE_SPEC.exists() else ROOT/'authoritative_spec.md'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
