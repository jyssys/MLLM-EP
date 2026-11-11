"""Add one signed observer bootstrap to the unchanged exclusive dispatcher."""
import argparse
import os
import subprocess
import zipfile
from .common import *


def execute(name):
    initialize()
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    parent=cfg.get('diagnosis_parent')
    if parent:
        p=Path(parent['qualification'])
        assert digest(p)==parent['qualification_sha256']
        assert json.loads(p.read_text())['campaign_gate_passed']
    from ..huginn_net_benefit import run as reused
    reused.ROOT=ROOT;reused.initialize=initialize
    original=subprocess.Popen
    def launch(command,*args,**kwargs):
        if isinstance(command,list) and 'poc_flashvep.looped_pd_headroom_recovery.worker' in command:
            folder=ROOT/'raw'/name
            snapshot=folder/'worker_source'
            paths=sorted(list(SOURCE.rglob('*.py'))+list(SOURCE.rglob('*.cc')))
            with zipfile.ZipFile(folder/'diagnosis_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in paths:archive.write(p,str(p.relative_to(REPO)))
                archive.extractall(snapshot)
            write(folder/'diagnosis_source_hashes.json',
                {str(p.relative_to(REPO)):digest(p) for p in paths})
            env=dict(kwargs['env'])
            env['LOOP28_DIAG_SNAPSHOT']=str(snapshot)
            env['PYTHONPATH']=str(snapshot/'poc_flashvep/huginn_loop28_tail_diagnosis/bootstrap')+os.pathsep+env['PYTHONPATH']
            kwargs['env']=env
        return original(command,*args,**kwargs)
    subprocess.Popen=launch
    try:return reused.execute(name)
    finally:subprocess.Popen=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name')
    raise SystemExit(0 if execute(p.parse_args().name) else 1)
