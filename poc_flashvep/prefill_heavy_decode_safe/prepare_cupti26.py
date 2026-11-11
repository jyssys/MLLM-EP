"""Pin the loaded CUPTI, not only the compile-time headers and RUNPATH."""
import json
from .common import *


def main():
    initialize();library=Path('/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.12').resolve()
    controls=[]
    prior=json.loads((ROOT/'state/anchor_controls_v2.json').read_text())['jobs']
    for name in prior:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        cfg.update(name=name.replace('_v2_N64_','_cupti26_N64_'),discovery_cupti_preload=str(library),
            discovery_cupti_preload_SHA256=digest(library),phase='CUPTI26_PINNED_NATIVE_PROFILER_ABA_CONTROL')
        save(cfg);controls.append(cfg['name'])
    cfg=json.loads((ROOT/'configs'/'native_huginn_early_profile_state8.json').read_text())
    cfg.update(name='native_huginn_cupti26_asyncprobe_state8',discovery_cupti_preload=str(library),
        discovery_cupti_preload_SHA256=digest(library),required_qualification='native_huginn_clean_recovery_state32',
        phase='CUPTI26_PINNED_WITH_ORIGINAL_ASYNCHRONOUS_FULL_STATE_OBSERVER')
    save(cfg)
    write(ROOT/'state/cupti26_controls.json',dict(jobs=controls,Huginn_state8=cfg['name'],
        hypothesis='The bridge compiles with CUPTI API 26, but the observed workers loaded pip CUPTI API 28. Pin API 26 before Torch imports and keep the original observer unchanged. This is not an established root cause.'))
    print(' '.join(controls[:3]+[cfg['name']]+controls[3:]))


if __name__=='__main__':main()
