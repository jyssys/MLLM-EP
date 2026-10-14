import hashlib
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parent
MASK='4,5,6,7'
GPUS=[4,5,6,7]
UUIDS=['6076e2f2-5b63-3761-5586-56ceb7df8139','a1a1cfcf-93a1-3544-9a5e-e58144b68730',
       'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','4cc26b88-19fc-1988-f9e0-17858aa7a99b']
PYTHON='/home/esjung/.venvs/llada2-flash-sglang-053/bin/python'
MODEL='ShourenWSR/how-to-loop-moe'
REVISION='c1c857bab4665f00302f191fde5feb274ea97936'
SUBFOLDER='Foil-1'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
