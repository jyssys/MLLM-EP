"""Qualify long-context primitives without entering online migration."""
import json
from .common import *


def main():
    initialize();jobs=[];reference='async_ouro_L16384_native_reference_state8'
    source=json.loads((ROOT/'configs'/(reference+'.json')).read_text())
    for cut,tile in ((3,None),(3,1024),(2,None)):
        suffix='full' if tile is None else f'tile{tile}'
        state_name=f'cost_ouro_L16384_loop{cut}_{suffix}_state8'
        cfg=dict(source);cfg.update(name=state_name,method='loop',axis='loop',cut=cut,
            native_partial_cut=cut,native_frontier=None,D_prefill_loop_cap=1,
            D_prefill_token_cap=tile,D_prefill_cap_includes_prelude=True,
            campaign_state_reference=reference,required_qualification=reference,
            qualification_fresh_native=False,phase='LONG_CONTEXT_LOOP_PRIMITIVE_EXACT_STATE',
            priority_serial_comparison=False,require_realized_quantum_contract=True)
        cfg.pop('qualification_output_reference',None);save(cfg);jobs.append(state_name)
        cfg=dict(cfg);name=f'cost_ouro_L16384_loop{cut}_{suffix}_serial8'
        cfg.update(name=name,state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
            async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
            observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
            required_qualification=state_name,calibration_serial_output_reference=reference,
            serial_output_reference=reference,serial_baseline='cost_ouro_L16384_zero_serial8',
            load='serial',arrival_pattern='SERIAL_CALIBRATION',phase='LONG_CONTEXT_LOOP_PRIMITIVE_COST_NOT_SERVING',
            discovery_activity=False,discovery_event_observer=False,observe_P_submission_readiness=False,
            clean_confirmation=False,host_cost_trace=False)
        save(cfg);jobs.append(name)
    cfg=dict(source);name='cost_ouro_L16384_zero_serial8'
    cfg.update(name=name,state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
        async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
        observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
        required_qualification=reference,serial_output_reference=reference,serial_baseline=name,
        load='serial',arrival_pattern='SERIAL_CALIBRATION',phase='LONG_CONTEXT_NATIVE_SERIAL_REFERENCE',
        discovery_activity=False,discovery_event_observer=False,observe_P_submission_readiness=False,
        clean_confirmation=False,host_cost_trace=False)
    save(cfg);jobs.insert(0,name)
    write(ROOT/'state/long_loop_cost_plan.json',dict(jobs=jobs,
        scope='N8 exact state and separate serial CUDA Event enclosures. This is primitive calibration, not Phase 2 online migration or Decode-safe qualification.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
