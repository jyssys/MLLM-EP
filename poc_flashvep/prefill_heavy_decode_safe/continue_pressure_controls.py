"""Do not overlap the current Native controls or primitive qualification."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/extended_18k_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(10)
    plan=json.loads((ROOT/'state/pressure_quiescent_plan.json').read_text())
    if sequence(plan['all_jobs'],'pressure_native_control_controller'):
        from .quiescent_analysis import main as analyze
        analyze(ROOT/'state/pressure_quiescent_plan.json')
        screen(plan['clean_reference'])
    write(ROOT/'state/pressure_control_controller_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
