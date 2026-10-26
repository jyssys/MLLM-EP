from pathlib import Path
import os
from ..looped_moe_headroom.common import REPO,ASSETS,PYTHON,XLLM,MODELS,SEED,digest,write,append
PRIOR=REPO/'poc_flashvep/reports/looped_moe_headroom'
ROOT=Path(os.environ.get('LOOPED_EP_RESULTS',REPO/'poc_flashvep/reports/looped_moe_ep_coalescing_replication'))
SPEC=REPO/'poc_flashvep/reports/looped_moe_ep_coalescing_replication_poc_spec.md'

def setup():
    for name in ['configs','logs','raw','summary','figures','scripts','quality','oracle','state']:
        (ROOT/name).mkdir(parents=True,exist_ok=True)
