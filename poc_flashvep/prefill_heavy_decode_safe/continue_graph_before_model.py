"""Test the collector ordering hypothesis after the current Native controls."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .envelope_analysis import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/matched_graph_capacity_recheck_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    path=ROOT/'state/graph_before_model_18k_plan.json';cfg=json.loads(path.read_text())
    if sequence(cfg['all_jobs'],'graph_before_model_18k_controller'):
        value=analyze(path);t=value['trace']['prefill']
        screen(cfg['jobs'][0],window=(t['left'],t['right']))
    write(ROOT/'state/graph_before_model_18k_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
