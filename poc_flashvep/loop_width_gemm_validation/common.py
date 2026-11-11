import json,time
from pathlib import Path
from datetime import datetime,timezone
from ..huginn_net_benefit import common as inherited

REPO=inherited.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/ouro_huginn_loop_width_vs_token_chunk_gemm_validation'
OLD_ROOT=REPO/'poc_flashvep/reports/huginn_autonomous_adaptive_loop_pd_net_benefit'
DIAG_ROOT=REPO/'poc_flashvep/reports/huginn_loop28_decode_tail_diagnosis_20261111'
BASE='4470f9e2a1c6ea1c1112c59ed751fa921bcaf2dd'
BRANCH='flashvep/loop-width-gemm-validation-20261111'
START=datetime(2026,11,11,7,38,55,tzinfo=timezone.utc).timestamp()
DEADLINE=START+6*3600
GPU_DEADLINE=DEADLINE-3600
LONG_START_CUTOFF=GPU_DEADLINE-20*60
write,digest=inherited.write,inherited.digest


def configure():
    for key in ('ROOT','BASE','BRANCH','START','DEADLINE','GPU_DEADLINE','LONG_START_CUTOFF'):
        setattr(inherited,key,globals()[key])
    inherited.SOURCE=REPO/'poc_flashvep/huginn_net_benefit'
    inherited.configure()
    from ..looped_pd_autonomous_headroom.common import resources
    resources().APPROVED_LOGS.update((OLD_ROOT/'burner.log',DIAG_ROOT/'burner.log'))


def initialize():
    configure()
    from ..looped_pd_autonomous_headroom.common import initialize as setup
    setup()
    for name in ('analysis','tools','data'):(ROOT/name).mkdir(exist_ok=True)


def save(cfg):
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg
    else:write(path,cfg)


if __name__=='__main__':
    initialize();print(ROOT)
