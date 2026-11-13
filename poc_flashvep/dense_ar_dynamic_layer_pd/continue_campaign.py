"""Resume the next bounded stages without using held-out future results."""
from .common import *


def wait_for(path,seconds):
    deadline=time.monotonic()+seconds
    while not path.exists():
        assert time.monotonic()<deadline,'Missing prerequisite: '+str(path)
        time.sleep(5)


def main():
    initialize()
    wait_for(ROOT/'analysis/strong_baseline_results.json',5400)
    from .calibrate_costs import main as calibrate
    calibrate()
    assert (ROOT/'configs/calibrated_cost_model.json').exists(),'Cost calibration did not pass'
    from .online_pilots import main as pilots
    pilots()


if __name__=='__main__':main()
