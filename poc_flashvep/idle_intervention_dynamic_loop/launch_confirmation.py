"""Replace an idle legacy owner before the frozen safety-gated confirmation."""
import fcntl
import signal
import subprocess
from .common import *
from .qualification import qualify
from .analyze import metrics


def main():
    assert RESUMPTION is not None
    names=[f'final_calibration_{model}_{arm}128' for model in ('ouro','huginn') for arm in ('native','zero')]
    while time.time()<LONG_START_CUTOFF:
        receipts=[ROOT/'raw'/name/'dispatcher_result.json' for name in names]
        if all(p.exists() for p in receipts):break
        time.sleep(5)
    assert all(p.exists() and json.loads(p.read_text()).get('complete') for p in receipts)
    for name in names:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        assert qualify(name,cfg.get('comparison_reference'))['passed']
        assert metrics(name)['Decode_SLO_screen']
    path=ROOT/'state/quiet_lane.lock'
    while time.time()<LONG_START_CUTOFF:
        with path.open('a+') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                active=json.loads((ROOT/'state/active.json').read_text())
                assert active.get('status')=='IDLE',active
                pending=[n for n in json.loads((ROOT/'state/queue.json').read_text())
                    if not (ROOT/'raw'/n/'dispatcher_result.json').exists()]
                assert not pending,pending
                old=json.loads((ROOT/'state/queue_owner.json').read_text())
                for pid in (old['pid'],old['guard']):
                    proc=Path('/proc')/str(pid)
                    words=(proc/'cmdline').read_bytes().decode().split('\0')
                    assert proc.stat().st_uid==os.getuid() and 'poc_flashvep.idle_intervention_dynamic_loop.owner' in words
                    os.kill(pid,signal.SIGTERM)
                write(ROOT/'state/confirmation_owner_handoff.json',dict(time=time.time(),previous=old,
                    scope='No scientific worker was active. Reload the owner to enforce prior-start safety gates.'))
                break
        time.sleep(2)
    else:raise RuntimeError('The final start cutoff prevents confirmation launch')
    from .confirmation import main as freeze
    freeze()
    from .context_controls import main as contexts
    contexts()
    with (ROOT/'logs/confirmation_owner.log').open('a') as output:
        child=subprocess.Popen(['/home/esjung/.venvs/looped-pd-abc/bin/python','-m',
            'poc_flashvep.idle_intervention_dynamic_loop.owner'],cwd=REPO,env=dict(os.environ),
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/confirmation_owner_launch.json',dict(time=time.time(),pid=child.pid,
        GPU_deadline=GPU_DEADLINE,final_deadline=DEADLINE))
    print(json.dumps(dict(new_owner=child.pid,confirmation_frozen=True)),flush=True)


if __name__=='__main__':main()
