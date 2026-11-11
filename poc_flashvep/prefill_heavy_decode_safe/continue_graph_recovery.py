"""Retain failed capture evidence before a bounded corrective experiment."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .envelope_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/huginn_gap_cost_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    plan=ROOT/'state/graph_sync_18k_plan.json';cfg=json.loads(plan.read_text())
    if sequence(cfg['all_jobs'],'graph_sync_18k_controller'):
        value=analyze(plan);trace=value['trace']['prefill']
        screen(cfg['jobs'][0],window=(trace['left'],trace['right']))
    write(ROOT/'state/graph_sync_18k_controller_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
