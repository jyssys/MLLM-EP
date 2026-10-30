"""Independent evidence paths and a wall-clock campaign limit."""
import json
import time
from pathlib import Path
from ..looped_pd_abc.common import REPO,RLT,PYTHON,MODELS,MODEL_PINS,UUIDS,environment,write,append,digest

ROOT=REPO/'poc_flashvep/reports/adaptive_loop_token_pd_2gpu'
SOURCE=Path(__file__).parent
ARMS=['DP','DP_Chunked','PD_G1','Loop_Static','Loop_Dynamic','Token_Static','Token_Dynamic']
START_UTC='2026-10-30T17:37:00+00:00'
START_EPOCH=1793381820.0

def setup():
    for name in ('raw','data','configs','logs','state','figures'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)

def config():
    import os
    return json.loads(Path(os.environ['ARC_CONFIG']).read_text())

def redirect():
    """Redirect evidence writers only. Do not edit inherited source files."""
    from ..adaptive_recurrence_cut import common as old
    old.ROOT=ROOT;old.setup=setup;old.load_config=config;old.append=append
    from ..looped_pd_abc import common as abc
    abc.ROOT=ROOT;abc.setup=setup
    from ..looped_pd_b_numerical_diagnosis import common as numerical
    numerical.ROOT=ROOT

def remaining():
    return START_EPOCH+12*3600-time.time()

def budget(**extra):
    setup()
    path=ROOT/'budget.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    previous.update(start_utc=START_UTC,start_epoch=START_EPOCH,
                    elapsed_hours=(time.time()-START_EPOCH)/3600,
                    hard_deadline_epoch=START_EPOCH+43200,
                    no_long_job_after_epoch=START_EPOCH+39600,
                    closeout_reserve_seconds=3600,last_update_epoch=time.time())
    previous.setdefault('initial_monotonic',time.monotonic()-(time.time()-START_EPOCH))
    previous.update(extra);write(path,previous)
    return previous
