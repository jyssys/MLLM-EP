"""Keep the generic Native probe separate from the final Loop comparison."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/graph_smoke_receipt.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    plan=json.loads((ROOT/'state/source_slot_probe_plan.json').read_text())
    if sequence(plan['jobs'],'source_slot_native_probe_controller'):
        cfg=json.loads((ROOT/'configs'/(plan['jobs'][-1]+'.json')).read_text())
        screen(plan['jobs'][-1],reference=cfg['comparison_reference'])
    write(ROOT/'state/source_slot_probe_controller_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
