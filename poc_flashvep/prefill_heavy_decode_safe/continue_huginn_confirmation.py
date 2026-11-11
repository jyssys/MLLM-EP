"""Continue repeated Native measurement without entering an unqualified policy."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .profile_controls import analyze
from .host_screen import screen


def main():
    initialize()
    while not (ROOT/'state/graph_bound_18k_controller_complete.json').exists():
        assert time.time()<LONG_START_CUTOFF;time.sleep(5)
    plan=json.loads((ROOT/'state/huginn_physical_confirmation_plan.json').read_text())
    if sequence([plan['jobs'][0]],'huginn_current_observer_qualification'):
        for i,path in enumerate(plan['plans']):
            cfg=json.loads(Path(path).read_text())
            if not sequence(cfg['jobs'],f'huginn_physical_confirmation_{i}'):break
            value=analyze(path)
            for row in value['observers']:
                if row['trace']:
                    trace=row['trace']['prefill']
                    screen(row['name'],window=(trace['left'],trace['right']))
    write(ROOT/'state/huginn_physical_confirmation_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
