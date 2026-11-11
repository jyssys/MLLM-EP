"""Measure conservative idle subsets without tracing Decode kernels."""
import json
from .common import *


def main():
    initialize();parent='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230';jobs=[]
    for mode in ('events','quiet_epochs','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('memory_','quiet_profile_')+'_'+mode
        cfg.update(name=name,comparison_reference=parent,
            phase='PARTIAL_COVERAGE_QUIESCENT_KERNEL_OBSERVER_CONTROL',
            discovery_event_observer=mode!='repeat',discovery_activity=mode=='quiet_epochs',
            activity_quiescent_decode=True,activity_delayed_start=True,
            activity_disable_after_outputs=True,activity_device_buffers=True,activity_memcpy=True,
            activity_window_fraction=.35,activity_window_seconds=8.,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
        save(cfg);jobs.append(name)
    write(ROOT/'state/quiescent_capture_plan.json',dict(clean_reference=parent,jobs=jobs,
        hypothesis='Native Decode tracing perturbs ITL. Trace every P kernel in the observation window. Trace D only during scheduler-quiescent epochs and disable before any ModelRunner execution. Actual captured kernels, not queue emptiness, determine idle within those epochs.',
        limitation='This method cannot measure whole-window D busy percentage or the complete idle distribution. Its sum of observed inactivity is a conservative subset, not total capacity. A post-Serving witness checks collector operation.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
