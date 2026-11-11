"""Keep the new campaign separate from published predecessors."""
import os,subprocess
from pathlib import Path
from datetime import datetime,timezone
from ..huginn_net_benefit import common as inherited

REPO=inherited.REPO
SOURCE=Path(__file__).parent
ROOT=REPO/'poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111'
WIDTH=REPO/'poc_flashvep/reports/ouro_huginn_loop_width_vs_token_chunk_gemm_validation'
BASE='30e1717b446e5097a0f523da0bcc890476af3ba1'
BRANCH='flashvep/prefill-heavy-decode-safe-20261111'
START=datetime(2026,11,11,12,27,21,tzinfo=timezone.utc).timestamp()
DEADLINE=START+10*3600
GPU_DEADLINE=DEADLINE-3600
LONG_START_CUTOFF=GPU_DEADLINE-20*60
write,digest=inherited.write,inherited.digest


def configure():
    for key in ('ROOT','BASE','BRANCH','START','DEADLINE','GPU_DEADLINE','LONG_START_CUTOFF'):
        setattr(inherited,key,globals()[key])
    inherited.SOURCE=REPO/'poc_flashvep/huginn_net_benefit'
    inherited.configure()
    from ..looped_pd_autonomous_headroom.common import resources
    resources().APPROVED_LOGS.add(WIDTH/'burner.log')


def initialize():
    configure()
    from ..looped_pd_autonomous_headroom.common import initialize as setup
    setup()
    for name in ('analysis','tools'):(ROOT/name).mkdir(exist_ok=True)
    budget=ROOT/'state/current_campaign_budget.json'
    if not budget.exists():write(budget,dict(start_epoch=START,hard_deadline_epoch=DEADLINE,
        GPU_stop_epoch=GPU_DEADLINE,long_job_start_cutoff_epoch=LONG_START_CUTOFF,
        maximum_wall_hours=10,scope='This campaign starts only after the prior GEMM campaign publishes and restores all eight approved burners. All phases and closeout fit within ten hours.'))


def save(cfg):
    initialize();path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert __import__('json').loads(path.read_text())==cfg,'Immutable configuration changed'
    else:write(path,cfg)


if __name__=='__main__':initialize();print(ROOT)
