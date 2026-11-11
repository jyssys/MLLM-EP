"""Close GPU work before analysis and verify every approved Burner."""
import json,subprocess,time
from .common import *


def main():
    initialize()
    done=ROOT/'state/native_three_slots_3072_complete.json'
    assert done.exists(),'Wait for the final bounded Native probe'
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[s for s in processes.splitlines() if ' -m poc_flashvep.looped_pd_headroom_recovery.worker ' in s]
    assert not active,active
    marker=ROOT/'state/CAMPAIGN_CLOSED.json'
    if not marker.exists():
        write(marker,dict(time=time.time(),elapsed_seconds=time.time()-START,
            reason='No model/workload passed every Phase 1 gate. The last bounded Native probe completed. Stop GPU search without relaxing measurement, context, or Decode criteria.',
            Phase_1=dict(ouro='INCONCLUSIVE_MEASUREMENT',huginn='INCONCLUSIVE_MEASUREMENT'),
            Phase_2='NOT_RUN: Phase 1 entry gate did not pass',
            scope='This closes GPU discovery. Final analysis, publication, and the queued Loop-specific audit remain.'))
    from ..completion_aware_loop_pd import restore_burners
    restore_burners.ROOT=ROOT;restore_burners.configure=configure
    restore_burners.main()


if __name__=='__main__':main()
