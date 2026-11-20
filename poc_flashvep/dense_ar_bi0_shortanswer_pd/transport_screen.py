"""Use one source snapshot for the frozen NIXL thread-count screen."""
import json
import time
import numpy as np
from .common import ROOT,write,save
from .qualification import configuration
from .qualification_analysis import main as state_gate
from .run import execute
from .metrics import summarize
from .quality import quality_gate


def main():
    while not (ROOT/'state/BI0_N8_state_jobs_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists()
        assert not (ROOT/'state/phase_B_hold.json').exists()
        time.sleep(3)
    assert state_gate(), 'Do not run performance after an unexplained structural/state failure'
    seed=2026112033;rate=2.35
    offsets=np.cumsum(np.random.default_rng(seed).exponential(1/rate,32));offsets-=offsets[0]
    design=ROOT/'configs/frozen_NIXL_thread_screen_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),dataset='qualification32.json',count=32,offered_QPS=rate,
            arrival_seed=seed,Native_thread_counts=[4,1,0],Agg_default=True,
            intervention='Only the official kv_connector_extra_config.num_threads field changes between Native arms.',
            scope='Diagnostic screen. Natural output quality and actual batches are reported. Not a sustainable-capacity comparison.'))
    results=[]
    for label,method,extra in (('Native_T4','Native',{}),('Native_T1','Native',dict(nixl_num_threads=1)),
                               ('Native_T0','Native',dict(nixl_num_threads=0)),('Agg','Agg',{})):
        name=f'diag_{label}_ShortAnswer_N32_q2.35_BI0_v1'
        cfg=configuration(name,method,'qualification32.json',count=32,serial=False,state_capture=False,
            correctness_only=False,arrival_offsets=offsets.tolist(),arrival_seed=seed,offered_QPS=rate,
            phase='DIAGNOSTIC_NIXL_THREAD_SCREEN',**extra)
        assert execute(save(cfg)),name
        quality=quality_gate(name)
        results.append(dict(name=name,metrics=summarize(name),quality=quality))
        write(ROOT/'analysis/NIXL_thread_screen_v1.json',dict(results=results))
    write(ROOT/'state/NIXL_thread_screen_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/transport_screen_hold.json',dict(time=time.time(),error=repr(error)))
        raise
