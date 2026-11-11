"""Keep each instrumented start paired with the same Native arrival trace."""
import json
from .common import *


def main():
    initialize();native_plan=json.loads((ROOT/'state/huginn_native128_plan.json').read_text())
    old='profile_huginn_L3072_pinned_events_state8'
    cfg=json.loads((ROOT/'configs'/(old+'.json')).read_text());state='confirm_huginn_L3072_observer_state8_v2'
    cfg.update(name=state,phase='CURRENT_COLLECTOR_HUGINN_EXACT_STATE_RECHECK')
    save(cfg);plans=[];jobs=[state]
    for parent in native_plan['jobs']:
        local=[]
        for mode in ('capture','repeat'):
            cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent+'_'+mode
            cfg.update(name=name,comparison_reference=parent,
                required_qualification=state if mode=='capture' else 'reserve_huginn_L3072_native_reference_state32',
                discovery_activity=mode=='capture',discovery_event_observer=mode=='capture',
                activity_D_graph_enclosures=False,activity_internal_host_pinned=True,
                activity_device_buffers=False,activity_memcpy=True,activity_window_seconds=8.,
                activity_window_fraction=.35,phase='INDEPENDENT_NATIVE_N128_PHYSICAL_CONFIRMATION')
            save(cfg);local.append(name);jobs.append(name)
        p=ROOT/'state'/(parent+'_profiler_plan.json')
        write(p,dict(clean_reference=parent,jobs=local,
            scope='Each trace has its own paired marker-free A/A controls. Do not substitute another arrival seed for an instrumentation control.'))
        plans.append(str(p))
    write(ROOT/'state/huginn_physical_confirmation_plan.json',dict(jobs=jobs,plans=plans))
    print(' '.join(jobs))


if __name__=='__main__':main()
