"""Reuse exclusive quiet-lane ownership and immutable job cleanup."""
import argparse
from .common import *


def execute(name):
    initialize()
    assert not (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'This campaign is closed. Do not restart GPU jobs after burner restoration.'
    from ..native_split_deep_validation import run as inherited
    inherited.initialize=initialize
    import subprocess
    original=subprocess.Popen
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    def launch(command,*args,**kwargs):
        target='poc_flashvep.looped_pd_autonomous_headroom.operator_micro'
        if cfg.get('idle_worker') and isinstance(command,list) and target in command:
            command=list(command);command[command.index(target)]='poc_flashvep.adaptive_idle_loop_continuation.'+cfg['idle_worker']
        return original(command,*args,**kwargs)
    subprocess.Popen=launch
    try:return inherited.execute(name)
    finally:subprocess.Popen=original


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name');a=parser.parse_args()
    raise SystemExit(0 if execute(a.name) else 1)
