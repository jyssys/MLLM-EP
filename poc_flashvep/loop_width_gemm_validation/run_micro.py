"""Exclusive timing and verified owned-process cleanup for projection experiments."""
import argparse,fcntl,os,signal,subprocess,zipfile
from .common import *


def execute(name):
    initialize();cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());folder=ROOT/'raw'/name
    assert not folder.exists(),'Use a new immutable run namespace'
    from ..looped_pd_autonomous_headroom.common import resources,preserve
    from ..looped_pd_abc.common import environment,PYTHON
    owner=resources();child=None
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);preserve();owner.stop(list(range(8)));folder.mkdir()
        write(folder/'isolation.json',dict(time=time.time(),active=owner.processes(),gpus=cfg['gpus']))
        with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in SOURCE.rglob('*'):
                if p.suffix in ('.py','.cc'):archive.write(p,str(p.relative_to(REPO)))
        command=[PYTHON,'-u','-m','poc_flashvep.loop_width_gemm_validation.micro',name]
        started=time.monotonic();code=None;timeout=False
        try:
            with (ROOT/'logs'/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=environment(cfg['gpus']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(folder/'command.json',dict(command=command,pid=child.pid,configuration_SHA256=digest(ROOT/'configs'/(name+'.json')),time=time.time()))
                print('START',name,child.pid,flush=True)
                try:code=child.wait(cfg['timeout'])
                except subprocess.TimeoutExpired:
                    timeout=True
                    assert os.getsid(child.pid)==child.pid
                    os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        finally:
            leftovers=[r for r in owner.processes() if child and os.getsid(r['pid'])==child.pid]
            if leftovers:
                os.killpg(child.pid,signal.SIGTERM)
                until=time.monotonic()+15
                while time.monotonic()<until and any(r for r in owner.processes() if os.getsid(r['pid'])==child.pid):time.sleep(.2)
            complete=code==0 and (folder/'projection_results.json').exists()
            write(folder/'dispatcher_result.json',dict(complete=complete,returncode=code,timeout=timeout,elapsed_seconds=time.monotonic()-started))
            preserve()
            if not owner.processes():owner.restore(list(range(8)))
        print('END',name,code,complete,flush=True)
        return complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');raise SystemExit(0 if execute(p.parse_args().name) else 1)
