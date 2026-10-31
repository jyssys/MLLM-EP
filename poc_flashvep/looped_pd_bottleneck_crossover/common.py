"""Campaign boundaries and immutable source references."""
import json
import os
import time
from pathlib import Path
from ..looped_pd_abc.common import REPO,RLT,PYTHON,UUIDS,environment,write,append,digest
from ..looped_pd_abc.common import MODEL_PINS as CATALOG

ROOT=REPO/'poc_flashvep/reports/looped_pd_bottleneck_crossover'
SOURCE=Path(__file__).parent
SPEC=REPO/'poc_flashvep/reports/looped_pd_prefill_decode_bottleneck_crossover_phase_abc_poc_spec.md'
START_EPOCH=1793428852.
START_UTC='2026-10-31T06:40:52+00:00'
BRANCH='flashvep/looped-pd-bottleneck-crossover-abc'
MODEL_PINS={k:CATALOG[k] for k in ('ouro','huginn')}
QWEN_PIN=('Qwen/Qwen3-1.7B','70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
OWNED=['poc_flashvep/looped_pd_bottleneck_crossover/','poc_flashvep/reports/looped_pd_bottleneck_crossover/',str(SPEC.relative_to(REPO))]
OWNED += ['poc_flashvep/reports/'+x for x in ('looped_pd_phase_a_independent_capacity.md',
    'looped_pd_phase_b_bottleneck_crossover.md','looped_pd_phase_c_dynamic_a_headroom.md',
    'looped_pd_prefill_decode_bottleneck_final_summary.md','looped_pd_prefill_decode_bottleneck_reproduction.md')]

def setup():
    for name in ('raw','configs','data','logs','state','figures'):(ROOT/name).mkdir(parents=True,exist_ok=True)

def config():return json.loads(Path(os.environ['ARC_CONFIG']).read_text())

def remaining():return START_EPOCH+43200-time.time()

def budget(**extra):
    setup();path=ROOT/'budget.json';value=json.loads(path.read_text()) if path.exists() else {}
    value.update(start_utc=START_UTC,start_epoch=START_EPOCH,hard_deadline_epoch=START_EPOCH+43200,
        no_long_job_after_epoch=START_EPOCH+39600,elapsed_hours=(time.time()-START_EPOCH)/3600,
        remaining_seconds=remaining(),last_update=time.time(),**extra)
    write(path,value);return value

def redirect():
    from ..adaptive_loop_token_pd_2gpu import common as prior
    prior.ROOT=ROOT;prior.setup=setup
    prior.redirect()

