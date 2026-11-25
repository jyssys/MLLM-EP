"""Record counter-tool and privilege facts without starting GPU profiling."""
import os
from pathlib import Path
import re
import shutil
import time

from .common import ROOT,digest,preserve,write


def privilege_facts(driver,status):
    admin=re.search(r'^RmProfilingAdminOnly:\s*(\d+)\s*$',driver,re.MULTILINE)
    capabilities=re.search(r'^CapEff:\s*([0-9a-fA-F]+)\s*$',status,re.MULTILINE)
    value=int(capabilities.group(1),16) if capabilities else None
    return dict(RmProfilingAdminOnly=int(admin.group(1)) if admin else None,
        CapEff_hex=capabilities.group(1) if capabilities else None,
        effective_CAP_SYS_ADMIN=bool(value&(1<<21)) if value is not None else None)


def main():
    preserve();target=ROOT/'analysis/gpu_counter_capability_audit_v0.json';assert not target.exists()
    driver_path=Path('/proc/driver/nvidia/params');status_path=Path('/proc/self/status')
    driver=driver_path.read_text();status=status_path.read_text();facts=privilege_facts(driver,status)
    write(target,dict(time=time.time(),uid=os.getuid(),tools={key:shutil.which(key) for key in ['ncu','nsys','dcgmi']},
        driver_parameter_source=str(driver_path),driver_parameter_SHA256=digest(driver_path),**facts,
        hardware_counter_session_started=False,driver_or_privilege_change=False,
        active_serving_profiler_insertion=False,SM_active='NOT_MEASURED',
        Tensor_Core_activity='NOT_MEASURED',HBM_bandwidth='NOT_MEASURED',
        existing_CUPTI_scope='Kernel, copy and memset activity intervals. Not the three hardware-counter measurements.',
        scope='Read-only current capability facts. No claim that every other collector is unavailable. '
            'No privileged probe or extra measurement was inserted into active Native policy timing.'))
    preserve();print('GPU_COUNTER_CAPABILITY_FACTS_RECORDED',facts,flush=True)


if __name__=='__main__':main()
