"""Avoid the measured mid-serving CUPTI disable stall."""
import json
from .common import *


def main():
    initialize();parent='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230';jobs=[]
    for mode in ('events','late_capture','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('memory_','late_profile_')+'_'+mode
        cfg.update(name=name,comparison_reference=parent,
            phase='LATE_ACTIVATION_WITHOUT_TIMED_DISABLE_NATIVE_CONTROL',
            discovery_event_observer=mode!='repeat',discovery_activity=mode=='late_capture',
            activity_delayed_start=True,activity_disable_after_outputs=True,
            activity_device_buffers=True,activity_memcpy=True,
            activity_window_fraction=.75,activity_window_seconds=7.,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
        save(cfg);jobs.append(name)
    write(ROOT/'state/late_capture_plan.json',dict(clean_reference=parent,jobs=jobs,
        hypothesis='The five-second collector measured 366.113 ms in P disable and 5.744 ms in D disable. Activate late and defer disabling until all outputs. Actual tracing extends beyond the exported window. All interference gates remain unchanged.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
