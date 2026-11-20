"""Observe campaign transitions. Do not dispatch or stop GPU work."""
import argparse
import json
import time
from .common import ROOT


def main(seconds):
    end=time.monotonic()+seconds;last=None
    known={p.name for p in (ROOT/'state').glob('*.json')}
    while time.monotonic()<end:
        try:
            lease=json.loads((ROOT/'state/GPU_lease_current.json').read_text())
            state=(lease.get('name'),lease.get('status'))
            if state!=last:
                print(json.dumps(dict(time=time.time(),GPU_lease=lease)),flush=True);last=state
            current={p.name for p in (ROOT/'state').glob('*.json')}
            added=current-known
            for name in sorted(added):
                if any(word in name for word in ('hold','failed','mismatch','complete','CLOSED','STOP')):
                    print(json.dumps(dict(time=time.time(),new_state_file=name)),flush=True)
            known=current
            if 'CAMPAIGN_CLOSED.json' in current:return
        except (OSError,json.JSONDecodeError) as error:
            print(json.dumps(dict(transient_read_error=repr(error))),flush=True)
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=1800)
    main(parser.parse_args().seconds)
