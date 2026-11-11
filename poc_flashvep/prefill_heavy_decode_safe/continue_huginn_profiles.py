"""Keep Huginn validation independent from the Ouro measurement search."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence


def main():
    initialize()
    while not (ROOT/'state/measurement_alternatives_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(10)
    plan=json.loads((ROOT/'state/huginn_3072_profile_plan.json').read_text())
    if sequence([plan['state8'],*plan['jobs']],'huginn_3072_profile_controller'):
        from .profile_controls import analyze
        analyze(ROOT/'state/huginn_3072_profile_plan.json')
    sequence([plan['state32']],'huginn_3072_native_state32_controller')
    write(ROOT/'state/huginn_profile_controller_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()
