"""Test a matched profiler on the improved legal Huginn Native envelope."""
import json
from .common import *


def main():
    initialize();reference='reserve_huginn_L3072_native_reference_state8'
    cfg=json.loads((ROOT/'configs'/(reference+'.json')).read_text())
    state8='profile_huginn_L3072_pinned_events_state8'
    cfg.update(name=state8,campaign_state_reference=reference,required_qualification=reference,
        qualification_fresh_native=False,discovery_activity=True,discovery_event_observer=True,
        activity_internal_host_pinned=True,activity_device_buffers=False,
        activity_memcpy=True,activity_window_seconds=5.,
        state_hash_defer_until_close=False,state_pinned_slots=6,
        phase='HUGINN_3072_PINNED_AND_EVENT_OBSERVER_EXACT_STATE',
        discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
        discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
    save(cfg)
    state32='reserve_huginn_L3072_native_reference_state32'
    cfg=json.loads((ROOT/'configs'/(reference+'.json')).read_text());pool=json.loads((ROOT/'data'/cfg['dataset']).read_text())
    cfg.update(name=state32,count=32,prompt_ids=[r['prompt_id'] for r in pool[:32]],request_caps=[16]*32,
        arrival_offsets=[0.]*32,state_hash_defer_until_close=False,state_pinned_slots=6,
        qualification_state_subset_reference=reference,
        required_qualification=reference,phase='HUGINN_3072_NATIVE_N32_FULL_STATE_REFERENCE')
    save(cfg)
    parent='reserve_huginn_L3072_O16_P2_mu115_N64_a2026111230';jobs=[]
    for mode in ('events','pinned_capture','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('reserve_','profile_')+'_'+mode
        cfg.update(name=name,comparison_reference=parent,required_qualification=state8 if mode=='pinned_capture' else reference,
            discovery_activity=mode=='pinned_capture',discovery_event_observer=mode!='repeat',
            activity_internal_host_pinned=True,activity_device_buffers=False,activity_memcpy=True,
            activity_window_seconds=8.,activity_window_fraction=.35,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808',
            phase='HUGINN_3072_MATCHED_NATIVE_PROFILER_CONTROLS')
        save(cfg);jobs.append(name)
    write(ROOT/'state/huginn_3072_profile_plan.json',dict(clean_reference=parent,jobs=jobs,state8=state8,state32=state32,
        scope='Legal matched pools. Positive host queue growth is not physical saturation. N8 observer qualification precedes tracing. N32 Native reference precedes any N128 policy expansion. No SLO or model change.'))
    print(state8,*jobs,state32)


if __name__=='__main__':main()
