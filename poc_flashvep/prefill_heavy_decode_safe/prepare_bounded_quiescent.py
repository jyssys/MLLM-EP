"""Bound actual P tracing and retire D tracing before GPU preparation."""
import json
from .common import *


def main():
    initialize();reference='async_ouro_L16384_native_reference_state8'
    cfg=json.loads((ROOT/'configs'/(reference+'.json')).read_text());state='bounded_quiet_ouro_L16384_observer_state8'
    cfg.update(name=state,campaign_state_reference=reference,required_qualification=reference,
        qualification_fresh_native=False,discovery_event_observer=True,discovery_activity=True,
        activity_quiescent_decode=True,activity_delayed_start=True,activity_device_buffers=True,
        activity_disable_after_outputs=False,activity_quiet_stop_before_submit=True,
        activity_window_fraction=.35,activity_window_seconds=8.,
        phase='BOUNDED_PARTIAL_COVERAGE_OBSERVER_EXACT_STATE',
        discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
        discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
    save(cfg);parent='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230';jobs=[]
    for mode in ('events','quiet_epochs','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('memory_','bounded_quiet_profile_')+'_'+mode
        cfg.update(name=name,comparison_reference=parent,required_qualification=state,
            phase='ACTUALLY_BOUNDED_P_WITH_QUIESCENT_D_OBSERVER_CONTROL',
            discovery_event_observer=mode!='repeat',discovery_activity=mode=='quiet_epochs',
            activity_quiescent_decode=True,activity_delayed_start=True,activity_device_buffers=True,
            activity_disable_after_outputs=False,activity_quiet_stop_before_submit=True,
            activity_memcpy=True,activity_window_fraction=.35,activity_window_seconds=8.,
            discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
            discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
        save(cfg);jobs.append(name)
    write(ROOT/'state/bounded_quiescent_plan.json',dict(clean_reference=parent,state8=state,jobs=jobs,
        hypothesis='The first partial collector passed local ITL screens but failed whole-run TTFT. P tracing continued after its exported window. Limit actual P collection to eight seconds and close D epochs before GPU metadata preparation. No threshold changes.',
        limitation='Partial D coverage remains a measured lower bound only. It does not establish the whole-window D idle percentage or complete idle distribution.'))
    print(state,*jobs)


if __name__=='__main__':main()
