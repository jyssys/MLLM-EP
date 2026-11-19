"""Wait for explicit pre-Main gates, then execute the frozen common-grid study."""
import json
import time
from .common import ROOT,write


def main():
    ready=ROOT/'analysis/B_S_pre_main_gate.json'
    while not ready.exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists()
        time.sleep(2)
    assert json.loads(ready.read_text())['passed']
    assert json.loads((ROOT/'analysis/Native_reproduction_pre_main_gate.json').read_text())['passed']
    # Use the existing campaign function. Its internal gate checks remain active.
    import sys
    from .campaign import main as campaign
    sys.argv=[sys.argv[0],'--main'];campaign()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/Main_driver_hold.json',dict(time=time.time(),error=repr(error),
            reason='Do not continue Main dispatch after a failed prerequisite or runtime/output gate.'))
        raise
