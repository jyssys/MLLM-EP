"""Reuse audited bounded GPU transaction, overriding its historical mask."""
import os
import subprocess
from pathlib import Path
from poc_native_lp_ep4.scripts import gpu_session as shared
from poc_speculative_moe_ep.contract import MASK

if __name__=='__main__':
    shared.MASK=MASK
    shared.PROJECT_ROOT=Path(__file__).resolve().parents[2]
    # Shared helper uses this solely as a status/watchdog path, not LP semantics.
    os.environ['NATIVE_LP_OUTPUT']=os.environ['SD_EP_OUTPUT']
    from poc_speculative_moe_ep.contract import write
    def snapshot():
        return dict(gpus=subprocess.check_output(['nvidia-smi','--id='+MASK,
            '--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem',
            '--format=csv,noheader'],text=True),processes=subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))
    # Logs reside beside versioned raw evidence, not in prior PoC directories.
    evidence=Path(os.environ['SD_EP_OUTPUT']+'.gpu_session.json')
    before=snapshot()
    try:
        shared.main()
    finally:
        write(evidence,dict(mask=MASK,before=before,after=snapshot(),
            note='post-run includes restored burner; foreign GPUs are read-only audit'))
