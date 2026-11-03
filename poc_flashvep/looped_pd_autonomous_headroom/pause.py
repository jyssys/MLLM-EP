"""Checkpoint a user-requested pause. Preserve incomplete evidence separately."""
import signal
from .common import *


def command(pid):
    proc=Path('/proc',str(pid))
    if proc.stat().st_uid!=os.getuid():raise RuntimeError('Not an owned process')
    return (proc/'cmdline').read_bytes().decode().split('\0')


def main():
    state=json.loads((ROOT/'state/campaign.json').read_text())
    active=json.loads((ROOT/'state/active.json').read_text())
    controllers=[];guards=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:words=command(int(proc.name))
        except (OSError,RuntimeError):continue
        if 'poc_flashvep.looped_pd_autonomous_headroom.campaign' in words:controllers.append(int(proc.name))
        if 'poc_flashvep.looped_pd_autonomous_headroom.deadline_guard' in words:guards.append(int(proc.name))
    # Stop the controller first. It cannot start another namespace during cleanup.
    for pid in controllers:os.kill(pid,signal.SIGSTOP)
    for pid in controllers:
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    for pid in guards:os.kill(pid,signal.SIGTERM)
    pid=active.get('pid');stopped=[]
    if pid:
        try:
            words=command(pid)
            assert 'poc_flashvep.looped_pd_headroom_recovery.worker' in words
            assert str(ROOT/'configs'/(active['name']+'.json')) in words
            assert os.getsid(pid)==pid
            os.killpg(pid,signal.SIGTERM);stopped.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
    until=time.monotonic()+25
    owner=resources()
    while any(os.path.exists('/proc/'+str(row['pid'])) and os.getsid(row['pid']) in stopped for row in owner.processes()) and time.monotonic()<until:
        time.sleep(.5)
    for row in owner.processes():
        if row['pid'] and os.getsid(row['pid']) in stopped:
            assert Path('/proc',str(row['pid'])).stat().st_uid==os.getuid()
            os.killpg(os.getsid(row['pid']),signal.SIGKILL)
    until=time.monotonic()+15
    while owner.processes() and time.monotonic()<until:time.sleep(.5)
    interrupted=ROOT/'raw'/active['name'] if active.get('name') else None
    if interrupted:
        write(interrupted/'user_pause.json',dict(time=time.time(),status='INTERRUPTED_USER_PAUSE',
            scope='Incomplete request evidence is preserved. Do not count this start as a correctness pass or a completed measurement. Resume in a new namespace.'))
    checkpoint=dict(status='PAUSED_BY_USER',time=time.time(),previous_campaign_state=state,previous_active=active,
        completed_operator_models=['ouro','huginn'],controllers_stopped=controllers,guards_stopped=guards,
        worker_sessions_stopped=stopped,budget_deadline_unchanged=DEADLINE,
        resume='Inspect partial evidence. Use a new namespace for the interrupted gate. Preserve both exact actual-weight operator passes. Re-arm the deadline guard only after checking the remaining authorized budget.')
    write(ROOT/'state/PAUSED_BY_USER.json',checkpoint)
    write(ROOT/'state/active.json',dict(status='PAUSED_BY_USER',time=time.time()))
    write(ROOT/'state/campaign.json',checkpoint)
    preserve()
    if not owner.processes():owner.restore(list(range(8)))
    write(ROOT/'state/pause_resource_status.json',dict(time=time.time(),approved_burners=owner.burners(),active=owner.processes()))
    print('PAUSED. EVIDENCE_PRESERVED. APPROVED_BURNERS',sorted(owner.burners()))


if __name__=='__main__':main()
