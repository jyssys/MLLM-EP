"""Validated authorized-mask burner/watchdog/finally guard, reused unchanged."""
import os
from poc_dllm_request_persistent_replica.contract import MASK,PROJECT
import poc_native_lp_ep4.scripts.gpu_session as shared
shared.MASK=MASK;shared.PROJECT_ROOT=PROJECT
os.environ['NATIVE_LP_OUTPUT']=os.environ['DLLM_PERSIST_OUTPUT']
original_stop=shared.stop_burner


def guarded_stop():
    try:original_stop()
    except BaseException:
        if not shared.compute_pids():shared.restore()
        raise


shared.stop_burner=guarded_stop
if __name__=='__main__':shared.main()
