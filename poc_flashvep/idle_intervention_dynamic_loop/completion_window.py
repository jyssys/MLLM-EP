"""Finish declared bounded jobs without extending GPU or publication deadlines."""
import fcntl
import signal
import subprocess
from .common import *


def main():
    assert RESUMPTION is not None
    record=ROOT/'state/bounded_completion_window.json'
    assert not record.exists()
    write(record,dict(created=time.time(),GPU_deadline=GPU_DEADLINE,final_deadline=DEADLINE,
        original_long_start_cutoff=LONG_START_CUTOFF,
        maximum_controller_cutoff=GPU_DEADLINE-180,
        required_startup_timeout_cleanup_margin_seconds=60,
        allowed_prefixes=['confirm_','context_'],
        scope='Internal start-window amendment within unchanged hard budgets. Only already declared cells can run. Startup timeout plus run timeout plus 60 seconds must fit before the original GPU deadline. No scientific configuration, SLO, seed, inclusion rule, or model-runtime guard changes.'))
    old=json.loads((ROOT/'state/queue_owner.json').read_text())
    while time.time()<GPU_DEADLINE-180:
        proc=Path('/proc')/str(old['pid'])
        alive=proc.exists() and '\nState:\tZ' not in (proc/'status').read_text()
        if not alive and json.loads((ROOT/'state/active.json').read_text()).get('status')=='IDLE':break
        time.sleep(3)
    else:return
    with (ROOT/'state/quiet_lane.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        proc=Path('/proc')/str(old['guard'])
        if proc.exists():
            words=(proc/'cmdline').read_bytes().decode().split('\0')
            assert proc.stat().st_uid==os.getuid() and 'poc_flashvep.idle_intervention_dynamic_loop.owner' in words and '--guard' in words
            os.kill(old['guard'],signal.SIGTERM)
    env=dict(os.environ,IDLE_LOOP_COMPLETION_WINDOW=str(record))
    with (ROOT/'logs/bounded_completion_owner.log').open('a') as output:
        child=subprocess.Popen(['/home/esjung/.venvs/looped-pd-abc/bin/python','-m',
            'poc_flashvep.idle_intervention_dynamic_loop.owner'],cwd=REPO,env=env,
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/bounded_completion_owner_launch.json',dict(time=time.time(),pid=child.pid,
        previous_owner=old,GPU_deadline=GPU_DEADLINE,final_deadline=DEADLINE))
    print(json.dumps(dict(bounded_completion_owner=child.pid)),flush=True)


if __name__=='__main__':main()
