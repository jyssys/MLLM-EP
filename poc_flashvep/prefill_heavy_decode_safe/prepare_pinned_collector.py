"""Test the documented host-pinned alternative against the actual default."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for model in ('ouro','huginn'):
        parent=f'native_{model}_devicebuffers_observer_state8'
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
        name=f'native_{model}_pinned_internal_observer_state8'
        cfg.update(name=name,activity_internal_host_pinned=True,
            phase='HOST_PINNED_CUPTI_INTERNAL_BUFFER_STATE_QUALIFICATION')
        save(cfg);jobs.append(name)
    state_jobs=list(jobs);parent='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230'
    serving=[]
    for mode in ('events','host_buffers','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('memory_','pinned_profile_')+'_'+mode
        cfg.update(name=name,comparison_reference=parent,required_qualification=state_jobs[0],
            phase='HOST_PINNED_INTERNAL_BUFFER_NATIVE_PROFILER_CONTROL',
            discovery_activity=mode=='host_buffers',discovery_event_observer=mode!='repeat',
            activity_internal_host_pinned=True,activity_device_buffers=False,activity_memcpy=True,
            activity_window_seconds=8.,activity_window_fraction=.35,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
        save(cfg);serving.append(name)
    write(ROOT/'state/pinned_collector_qualification_plan.json',dict(jobs=state_jobs,
        hypothesis='The actual API26 default is device profiling storage. The installed official header describes host-pinned storage as a performance alternative. This observer changes storage only, not model execution.'))
    write(ROOT/'state/pinned_profile_plan.json',dict(clean_reference=parent,jobs=serving,
        scope='Same legal Native geometry and immutable arrivals. Do not infer performance from the state tests. Both whole-run and local-tail interference gates remain mandatory.'))
    print(' '.join(jobs+serving))


if __name__=='__main__':main()
