"""Finish primitive calibration even when the online qualification gate fails."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence


def main():
    initialize()
    while not (ROOT/'state/ouro_18k_confirmation_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    sequence(json.loads((ROOT/'state/ouro_18k_gap_cost_plan.json').read_text())['jobs'],'ouro18k_gap_costs')
    write(ROOT/'state/ouro_18k_gap_cost_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
