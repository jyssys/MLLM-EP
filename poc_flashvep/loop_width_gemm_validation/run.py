import argparse,os,subprocess,zipfile
from .common import *


def execute(name):
    initialize()
    from ..huginn_net_benefit import run as reused
    reused.ROOT=ROOT;reused.initialize=initialize
    original=subprocess.Popen
    def launch(command,*args,**kwargs):
        if isinstance(command,list) and 'poc_flashvep.looped_pd_headroom_recovery.worker' in command:
            folder=ROOT/'raw'/name;snapshot=folder/'worker_source'
            paths=sorted(p for p in SOURCE.rglob('*') if p.suffix in ('.py','.cc'))
            with zipfile.ZipFile(folder/'width_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for p in paths:archive.write(p,str(p.relative_to(REPO)))
                archive.extractall(snapshot)
            write(folder/'width_source_hashes.json',{str(p.relative_to(REPO)):digest(p) for p in paths})
            env=dict(kwargs['env']);env['LOOP_WIDTH_SNAPSHOT']=str(snapshot)
            env['PYTHONPATH']=str(snapshot/'poc_flashvep/loop_width_gemm_validation/bootstrap')+os.pathsep+env['PYTHONPATH']
            kwargs['env']=env
        return original(command,*args,**kwargs)
    subprocess.Popen=launch
    try:return reused.execute(name)
    finally:subprocess.Popen=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');raise SystemExit(0 if execute(p.parse_args().name) else 1)
