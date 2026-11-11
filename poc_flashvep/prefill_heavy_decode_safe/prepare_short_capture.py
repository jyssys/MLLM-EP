"""Limit actual tracing, not only the exported interval."""
import json
from .common import *


def main():
    initialize();jobs=[]
    parent='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230'
    for mode in ('events','short_capture','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
        name=parent.replace('memory_','short_profile_')+'_'+mode
        cfg.update(name=name,comparison_reference=parent,
            phase='ACTUALLY_BOUNDED_CONCURRENT_KERNEL_CAPTURE_CONTROL',
            discovery_event_observer=mode!='repeat',discovery_activity=mode=='short_capture',
            activity_delayed_start=True,activity_device_buffers=True,activity_memcpy=True,
            activity_window_seconds=5.,activity_window_fraction=.35,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
        save(cfg);jobs.append(name)
    write(ROOT/'state/short_capture_plan.json',dict(clean_reference=parent,jobs=jobs,
        hypothesis='Full-run instrumentation amplifies queueing. Preallocate buffers at startup, but enable concurrent kernel tracing only during five seconds. Local Decode tail must still pass the unchanged interference gate.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
