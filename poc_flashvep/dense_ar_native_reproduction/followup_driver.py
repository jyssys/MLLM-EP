"""Finish prespecified Native observation and scheduler sensitivity controls."""
import json
import sys
import time
from .common import ROOT,write


def main():
    while not (ROOT/'state/main_common_sweep_complete.json').exists():
        for file in ('STOP_REQUESTED.json','Main_driver_hold.json','campaign_hold.json'):
            assert not (ROOT/'state'/file).exists(),file
        time.sleep(5)
    from .online_profile import main as profile
    profile()
    # Reuse the completed moderate-load receipts. Dispatch only missing cells.
    from .sensitivity import main as sensitivity
    sys.argv=[sys.argv[0]];sensitivity()
    write(ROOT/'state/followup_controls_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/followup_driver_hold.json',dict(time=time.time(),error=repr(error)))
        raise
