"""Keep this control set consecutive after the current Huginn controls."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence


def main():
    initialize()
    while not (ROOT/'state/huginn_profile_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(10)
    plan=json.loads((ROOT/'state/bounded_quiescent_plan.json').read_text())
    if sequence([plan['state8'],*plan['jobs']],'bounded_quiescent_native_controller'):
        from .quiescent_analysis import main as analyze
        analyze(ROOT/'state/bounded_quiescent_plan.json')
    write(ROOT/'state/bounded_quiescent_controller_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
