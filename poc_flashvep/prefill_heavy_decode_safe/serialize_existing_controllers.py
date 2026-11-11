"""Serialize waiting controllers without suspending an active GPU worker."""
import os,signal,time,json
from .common import *


def words(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    initialize();active=2967363;waiting=[2958101,2952390,2975991,2977614];paused=[];events=[]
    path=ROOT/'state/controller_serialization.json'
    try:
        assert alive(active) and 'poc_flashvep.prefill_heavy_decode_safe.sequence' in words(active)
        for pid in waiting:
            proc=Path('/proc')/str(pid);assert proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid
            assert any(w in ('poc_flashvep.prefill_heavy_decode_safe.sequence','poc_flashvep.prefill_heavy_decode_safe.continue_mixed') for w in words(pid))
            children=(proc/'task'/str(pid)/'children').read_text().split()
            assert not children,('Do not suspend a controller with an active child',pid,children)
            os.kill(pid,signal.SIGSTOP);paused.append(pid)
            events.append(dict(action='SERIALIZE_WAITING_CONTROLLER',PID=pid,time=time.time(),active_GPU_child=False))
        write(path,dict(active=active,paused=paused,events=events,closed=False,
            scope='Only idle owned controllers wait. The active GPU worker and its watchdog continue. Serial closeout avoids an unlocked legacy burner restoration race. No performance window is paused.'))
        for pid in [active,*waiting]:
            if pid in paused:
                os.kill(pid,signal.SIGCONT);paused.remove(pid)
                events.append(dict(action='RESUME_CONTROLLER',PID=pid,time=time.time()))
                write(path,dict(active=pid,paused=list(paused),events=events,closed=False))
            while alive(pid):
                assert time.time()<GPU_DEADLINE,'Stop new controller work at the GPU deadline'
                time.sleep(5)
            events.append(dict(action='CONTROLLER_EXIT',PID=pid,time=time.time()))
    finally:
        for pid in paused:
            if alive(pid):os.kill(pid,signal.SIGCONT)
        write(path,dict(active=None,paused=[],events=events,closed=True))


if __name__=='__main__':main()
