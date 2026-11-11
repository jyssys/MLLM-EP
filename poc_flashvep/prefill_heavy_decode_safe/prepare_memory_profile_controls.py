"""Test the recovered Native envelope before any migration experiment."""
import json
from .common import *


def main():
    initialize();jobs=[]
    parent='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230'
    for mode in ('events','large_buffers','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
        name=parent.replace('memory_','profile_')+'_'+mode
        cfg.update(name=name,comparison_reference=parent,
            phase='LONG_CONTEXT_NATIVE_OBSERVER_CONTROLS',
            discovery_event_observer=mode!='repeat',discovery_activity=mode=='large_buffers',
            activity_device_buffers=mode=='large_buffers',activity_memcpy=True,
            activity_window_seconds=8.,activity_window_fraction=.35,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
        save(cfg);jobs.append(name)
    write(ROOT/'state/long_native_profile_plan.json',dict(clean_reference=parent,jobs=jobs,
        scope='Same requests, arrivals, slots, KV pools, and frozen SLO. Events and kernel tracing require separate local-tail interference checks.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
