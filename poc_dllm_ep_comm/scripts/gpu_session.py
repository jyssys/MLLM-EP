"""Reuse scoped stop/watchdog/restore; never invoke burner with its default mask."""
import os
from poc_native_lp_ep4.scripts import gpu_session as shared
from poc_dllm_ep_comm.contract import MASK,PROJECT
if __name__=='__main__':
    shared.MASK=MASK;shared.PROJECT_ROOT=PROJECT
    os.environ['NATIVE_LP_OUTPUT']=os.environ['DLLM_COMM_OUTPUT']
    shared.main()
