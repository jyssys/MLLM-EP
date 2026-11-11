import json
import time
from pathlib import Path
from datetime import datetime, timezone
from ..huginn_net_benefit import common as inherited

REPO = inherited.REPO
SOURCE = Path(__file__).parent
OLD_ROOT = REPO/'poc_flashvep/reports/huginn_autonomous_adaptive_loop_pd_net_benefit'
ROOT = REPO/'poc_flashvep/reports/huginn_loop28_decode_tail_diagnosis_20261111'
BASE = '21a067b8bf2242bb8a16fc8c59ed59193e481f54'
START = datetime(2026,11,11,5,5,19,tzinfo=timezone.utc).timestamp()
DEADLINE = START+150*60
GPU_DEADLINE = DEADLINE-30*60
LONG_START_CUTOFF = GPU_DEADLINE-20*60
write, digest = inherited.write, inherited.digest


def configure():
    # Freeze and load the unchanged qualified adapters. Only evidence routing
    # and the independent operational clock change in this process.
    for key in ('ROOT','BASE','START','DEADLINE','GPU_DEADLINE','LONG_START_CUTOFF'):
        setattr(inherited,key,globals()[key])
    inherited.BRANCH = 'flashvep/huginn-loop28-tail-diagnosis-20261111'
    inherited.SOURCE = REPO/'poc_flashvep/huginn_net_benefit'
    inherited.configure()
    from ..looped_pd_autonomous_headroom.common import resources
    resources().APPROVED_LOGS.add(OLD_ROOT/'burner.log')


def initialize():
    configure()
    from ..looped_pd_autonomous_headroom.common import initialize as setup
    setup()


def save(cfg):
    p = ROOT/'configs'/(cfg['name']+'.json')
    if p.exists():
        assert json.loads(p.read_text()) == cfg
    else:
        write(p,cfg)
