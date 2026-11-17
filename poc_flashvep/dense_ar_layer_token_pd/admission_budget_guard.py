"""Close new admissions before the report and publication reserve."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,SOURCE,write


def main():
    deadline=json.loads((ROOT/'state/progress.json').read_text())['deadline_epoch']-4200
    heartbeat=0.
    while time.time()<deadline and not (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
        if time.time()-heartbeat>=60:
            heartbeat=time.time();print('ADMISSION_BUDGET_GUARD_ACTIVE',round(deadline-time.time()),flush=True)
        time.sleep(min(30,max(1,deadline-time.time())))
    if (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
        return
    write(ROOT/'state/CAMPAIGN_CLOSED.json',dict(time=time.time(),status='ADMISSIONS_CLOSED_DRAINING',
        reason='Reserve the final70 minutes for safe drain, evidence verification, reports and publication. This marker does not mark incomplete research gates as passed.'))
    allowed={'confirmation','confirmation_primary_first','output64_refinement',
        'heldout_baselines','baseline_budget_v2','matched_confirmation',
        'matched_policy_controls','adaptive_token_frontier','output64_frontier',
        'adaptive_source_flow','adaptive_output64_flow','flow_low_headroom_followup',
        'mixed_oracle_validation','stall_diagnosis','event_controls'}
    stopped=[]
    for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
        parts=line.split()
        if len(parts)<4 or '-m' not in parts:
            continue
        module=parts[parts.index('-m')+1]
        if not module.startswith('poc_flashvep.dense_ar_layer_token_pd.') or module.rsplit('.',1)[1] not in allowed:
            continue
        pid=int(parts[0]);proc=Path(f'/proc/{pid}')
        try:
            if proc.stat().st_uid!=os.getuid() or (proc/f'task/{pid}/children').read_text().strip():
                continue
            cmd=(proc/'cmdline').read_bytes()
            if module.encode() not in cmd:
                continue
            os.kill(pid,signal.SIGTERM);stopped.append(dict(pid=pid,module=module))
        except (ProcessLookupError,FileNotFoundError,PermissionError):
            continue
    write(ROOT/'state/budget_guard_waiting_controllers.json',dict(time=time.time(),stopped=stopped,
        scope='Stop only UID-owned campaign controllers with no child process. Do not interrupt an active GPU coordinator. Existing dispatcher cleanup and wall timeout remain active.'))
    print('ADMISSIONS_CLOSED_FOR_CLOSEOUT',stopped,flush=True)


if __name__=='__main__':
    main()
