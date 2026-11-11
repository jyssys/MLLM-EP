"""Calibrate the actual context after repeated Native timeline controls."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence


def main():
    initialize()
    while not (ROOT/'state/huginn_physical_confirmation_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    jobs=json.loads((ROOT/'state/huginn_gap_cost_plan.json').read_text())['jobs']
    sequence(jobs,'huginn_exact_context_gap_costs')
    write(ROOT/'state/huginn_gap_cost_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
