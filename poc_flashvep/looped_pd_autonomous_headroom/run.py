"""Reuse the owned-session guard and qualified runner in an isolated namespace."""
import argparse
import zipfile
from .common import *


def environment(gpus):
    activate()
    from ..looped_pd_headroom_recovery.common import environment as inherited
    env=inherited(gpus)
    env['PYTHONPATH']=str(SOURCE/'bootstrap')+os.pathsep+env['PYTHONPATH']
    env['AUTONOMOUS_HEADROOM_WORKER']='1'
    cfg=json.loads(Path(os.environ['AUTONOMOUS_LAUNCH_CONFIG']).read_text())
    folder=ROOT/'raw'/cfg['name']
    hashes={str(p):digest(p) for p in SOURCE.rglob('*.py')}
    env['AUTONOMOUS_SOURCE_HASHES']=json.dumps(hashes)
    write(folder/'campaign_source_hashes.json',hashes)
    with zipfile.ZipFile(folder/'campaign_source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in SOURCE.rglob('*.py'):archive.write(p,str(p.relative_to(REPO)))
    if cfg.get('worker') in ('operator_micro', 'collector_selftest'):
        env['AUTONOMOUS_OPERATOR']='1'
    return env


def execute(name,keep_quiet=True):
    initialize()
    cfg_path=ROOT/'configs'/(name+'.json')
    cfg=json.loads(cfg_path.read_text())
    assert time.time()<GPU_DEADLINE-180,'GPU closeout boundary reached'
    assert time.time()+cfg.get('timeout',300)+cfg.get('startup_timeout',240)<GPU_DEADLINE,'Job cannot safely fit GPU budget'
    activate()
    from ..looped_pd_headroom_recovery import run as inherited
    inherited.ROOT=ROOT;inherited.resources=resources;inherited.preserve=preserve;inherited.environment=environment
    os.environ['AUTONOMOUS_LAUNCH_CONFIG']=str(cfg_path)
    if cfg.get('worker') in ('operator_micro', 'collector_selftest'):
        # The inherited guard owns the Popen session. Only its module argument
        # changes; cleanup and ownership checks remain unchanged.
        original=inherited.subprocess.Popen
        def launch(command,*args,**kwargs):
            if isinstance(command,list) and 'poc_flashvep.looped_pd_headroom_recovery.worker' in command:
                command=list(command);command[command.index('poc_flashvep.looped_pd_headroom_recovery.worker')]='poc_flashvep.looped_pd_autonomous_headroom.'+cfg['worker']
            return original(command,*args,**kwargs)
        inherited.subprocess.Popen=launch
        try:return inherited.execute(name,keep_quiet)
        finally:inherited.subprocess.Popen=original
    return inherited.execute(name,keep_quiet)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--restore',action='store_true');a=p.parse_args()
    raise SystemExit(0 if execute(a.name,not a.restore) else 1)
