"""Test a conservative activity bound without calling graphs kernel unions."""
import json
from .common import *


def main():
    initialize()
    receipt=json.loads((ROOT/'state/graph_smoke_receipt.json').read_text())
    assert receipt['passed'],'Qualify graph and eager coverage first'
    reference='extended_ouro_L18432_native_reference_state8'
    common=dict(discovery_D_execution_events=False,discovery_natural_readiness=True,
        observed_source_cuts=[3],activity_D_graph_enclosures=True,
        activity_internal_host_pinned=True,activity_device_buffers=False,
        activity_quiescent_decode=False,activity_delayed_start=False,
        activity_disable_after_outputs=True,activity_memcpy=True,
        activity_window_fraction=.35,activity_window_seconds=8.,
        discovery_cupti_preload='/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so.2025.1.1',
        discovery_cupti_preload_SHA256='fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808')
    state='graph_bound_ouro_L18432_observer_state8'
    cfg=json.loads((ROOT/'configs'/(reference+'.json')).read_text())
    cfg.update(common);cfg.update(name=state,campaign_state_reference=reference,
        qualification_fresh_native=False,required_qualification=reference,
        discovery_activity=True,discovery_event_observer=True,
        state_hash_defer_until_close=False,state_pinned_slots=6,
        phase='GRAPH_ENVELOPE_AND_EAGER_OBSERVER_EXACT_STATE')
    save(cfg)
    parent='extended_ouro_L18432_O16_P2_mu130_pilot64_a2026111272'
    jobs=[]
    for mode in ('clean','capture','repeat'):
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('extended_','graph_bound_')+'_'+mode
        cfg.update(common);cfg.update(name=name,
            discovery_activity=mode=='capture',discovery_event_observer=mode=='capture',
            required_qualification=state if mode=='capture' else reference,
            comparison_reference=parent,phase='GRAPH_ENVELOPE_MATCHED_NATIVE_CONTROLS')
        save(cfg);jobs.append(name)
    write(ROOT/'state/graph_bound_18k_plan.json',dict(state8=state,all_jobs=[state,*jobs],
        clean_reference=jobs[0],jobs=jobs[1:],
        scope='P uses actual concurrent kernels. D uses graph GPU enclosures plus eager kernels and copies. The D complement is a conservative inactivity bound, not the exact kernel-idle percentage. All whole-run and local-tail screens remain unchanged.'))
    print(' '.join([state,*jobs]))


if __name__=='__main__':main()
