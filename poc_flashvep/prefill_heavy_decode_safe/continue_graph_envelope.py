"""Wait for the current owned probe, then test the new collector."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .envelope_analysis import analyze


def main():
    initialize()
    while not (ROOT/'state/source_slot_probe_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    plan=ROOT/'state/graph_bound_18k_plan.json';cfg=json.loads(plan.read_text())
    if sequence(cfg['all_jobs'],'graph_bound_18k_controller'):analyze(plan)
    write(ROOT/'state/graph_bound_18k_controller_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
