"""Complete the user's short comparison before resuming discovery."""
import json,time
from .common import *


def main():
    initialize();receipt=ROOT/'state/observer_controller_complete.json'
    assert receipt.exists() and json.loads(receipt.read_text())['returncode']==0
    plan=json.loads((ROOT/'state/priority_width_comparison_plan.json').read_text())
    write(ROOT/'state/user_priority_interruption.json',dict(time=time.time(),
        reason='The user requested Ouro and larger Huginn Token-width comparisons before workload discovery.',
        deferred_controller_PID=2813487,active_GPU_job_interrupted=False,
        scope='The waiting discovery controller stopped before it launched a child. All four observer checks completed. The ten-hour campaign deadline remains unchanged.'))
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.prefill_heavy_decode_safe.sequence',*plan['jobs']],cwd=REPO).returncode
    write(ROOT/'state/priority_width_controller_complete.json',dict(time=time.time(),returncode=code))


if __name__=='__main__':main()
