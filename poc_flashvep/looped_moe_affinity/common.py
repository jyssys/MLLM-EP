from pathlib import Path
import os
import json
import tempfile
from ..looped_moe_headroom.common import REPO,PYTHON,XLLM,MODELS,SEED,digest,write,append
PRIOR=REPO/'poc_flashvep/reports/looped_moe_ep_coalescing_replication'
ROOT=Path(os.environ.get('LOOPED_AFFINITY_RESULTS',REPO/'poc_flashvep/reports/looped_moe_ep_affinity_placement'))
SPEC=REPO/'poc_flashvep/reports/looped_moe_ep_affinity_placement_poc_spec.md'
DATA=Path('/home/esjung/looped-moe-affinity-assets')
DEPS=Path('/home/esjung/looped-moe-affinity-deps/site')
MODEL_ASSETS=Path('/home/esjung/looped-moe-headroom-assets')

def write(path,value):
    """Give each concurrent writer a private temporary file, then publish atomically."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as file:
        json.dump(value,file,indent=2,ensure_ascii=False);file.write('\n');temporary=Path(file.name)
    os.replace(temporary,path)

def setup():
    for sub in ['runs/configs','runs/logs','raw','figures','state','archives','scripts','quality','sources']:
        (ROOT/sub).mkdir(parents=True,exist_ok=True)

