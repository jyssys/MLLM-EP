"""Run one immutable job through the qualified exclusive GPU owner."""
import argparse
import os
import subprocess
import zipfile
from .common import *


def execute(name):
    initialize()
    assert not (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    from ..native_split_deep_validation import run as inherited
    inherited.initialize=initialize
    original=subprocess.Popen
    def launch(command,*args,**kwargs):
        target='poc_flashvep.looped_pd_headroom_recovery.worker'
        if isinstance(command,list) and target in command:
            folder=ROOT/'raw'/name;snapshot=folder/'worker_source'
            with zipfile.ZipFile(folder/'new_campaign_source.zip') as archive:
                assert all(not p.startswith('/') and '..' not in p.split('/') for p in archive.namelist())
                archive.extractall(snapshot)
            env=dict(kwargs['env']);env['INTERVENTION_SNAPSHOT']=str(snapshot)
            env['PYTHONPATH']=str(snapshot/'poc_flashvep/idle_intervention_dynamic_loop/bootstrap')+os.pathsep+env['PYTHONPATH']
            kwargs['env']=env
        return original(command,*args,**kwargs)
    subprocess.Popen=launch
    try:return inherited.execute(name)
    finally:subprocess.Popen=original


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
    raise SystemExit(0 if execute(args.name) else 1)
