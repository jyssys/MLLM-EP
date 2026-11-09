"""Close scientific controllers only after every declared cell has a receipt."""
import fcntl
import signal
from .common import *


def stop_verified(pid,module):
    proc=Path('/proc')/str(pid)
    if not proc.exists():return 'ALREADY_EXITED'
    words=(proc/'cmdline').read_bytes().decode().split('\0')
    if not words or not words[0]:return 'ALREADY_EXITED'
    assert proc.stat().st_uid==os.getuid() and module in words,(pid,words)
    os.kill(pid,signal.SIGTERM)
    return 'SIGTERM_VERIFIED_CONTROLLER'


def main():
    configure()
    while time.time()<DEADLINE:
        queue=json.loads((ROOT/'state/queue.json').read_text())
        pending=[name for name in queue if not (ROOT/'raw'/name/'dispatcher_result.json').exists()]
        active=json.loads((ROOT/'state/active.json').read_text())
        if not pending and active.get('status')=='IDLE':break
        time.sleep(3)
    assert not pending and active.get('status')=='IDLE'
    while True:
        with (ROOT/'state/quiet_lane.lock').open('a+') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:
                assert json.loads((ROOT/'state/active.json').read_text()).get('status')=='IDLE'
                record=dict(time=time.time(),status='SCIENTIFIC_QUEUE_CLOSED',declared_cells=len(queue),
                    GPU_deadline=GPU_DEADLINE,final_deadline=DEADLINE,
                    scope='Every declared cell has a retained completion, failure, or not-run receipt. No new scientific work remains scheduled. Burner operation continues separately.')
                write(ROOT/'state/CAMPAIGN_CLOSED.json',record)
                owner=json.loads((ROOT/'state/queue_owner.json').read_text())
                stopped={str(pid):stop_verified(pid,'poc_flashvep.idle_intervention_dynamic_loop.owner')
                    for pid in (owner['pid'],owner['guard'])}
                monitor=json.loads((ROOT/'state/monitor_owner.json').read_text())
                stopped[str(monitor['pid'])]=stop_verified(monitor['pid'],'poc_flashvep.idle_intervention_dynamic_loop.monitor')
                write(ROOT/'state/controller_closure.json',dict(time=time.time(),controllers=stopped))
                break
        time.sleep(2)
    from .publish import preserved
    preserved()
    print(json.dumps(dict(closed=True,declared_cells=len(queue),preservation_passed=True)),flush=True)


if __name__=='__main__':main()
