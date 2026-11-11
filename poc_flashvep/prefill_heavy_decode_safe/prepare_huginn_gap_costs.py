"""Measure exact full-loop and tiled costs at the actual 3072-token context."""
import json
from .common import *


def main():
    initialize();reference='reserve_huginn_L3072_native_reference_state8'
    source=json.loads((ROOT/'configs'/(reference+'.json')).read_text());jobs=[]
    baseline='gapcost_huginn_L3072_zero_serial8'
    cfg=dict(source);cfg.update(name=baseline,state_probe=False,verify_transfer=False,
        async_prompt_state_probe=False,async_verify_handoff=False,
        calibration_serial=True,observe_device_execution=True,observe_service_timing=True,
        observe_core_graph_instance=True,priority_serial_comparison=True,
        required_qualification=reference,serial_output_reference=reference,serial_baseline=baseline,
        load='serial',arrival_pattern='SERIAL_CALIBRATION',phase='EXACT_CONTEXT_SERIAL_REFERENCE',
        discovery_activity=False,discovery_event_observer=False,observe_P_submission_readiness=False,
        clean_confirmation=False,host_cost_trace=False)
    save(cfg);jobs.append(baseline)
    for cut,tile in ((31,None),(31,128),(31,256),(31,512),(28,None)):
        suffix='full' if tile is None else f'tile{tile}'
        name=f'gapcost_huginn_L3072_loop{cut}_{suffix}_state8'
        cfg=dict(source);cfg.update(name=name,method='loop',axis='loop',cut=cut,
            native_partial_cut=cut,native_frontier=None,atomic_prompt=False,
            D_prefill_loop_cap=1,D_prefill_token_cap=tile,
            D_prefill_cap_includes_prelude=True,campaign_state_reference=reference,
            required_qualification=reference,qualification_fresh_native=False,
            require_realized_quantum_contract=True,state_hash_defer_until_close=False,state_pinned_slots=6,
            phase='EXACT_CONTEXT_FULL_AND_TILED_LOOP_STATE')
        cfg.pop('qualification_output_reference',None);save(cfg);jobs.append(name)
        serial=name.replace('_state8','_serial8');cfg=dict(cfg)
        cfg.update(name=serial,state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
            async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
            observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
            required_qualification=name,serial_output_reference=reference,serial_baseline=baseline,
            calibration_serial_output_reference=reference,load='serial',arrival_pattern='SERIAL_CALIBRATION',
            phase='EXACT_CONTEXT_LOOP_QUANTUM_COST_NOT_SERVING',discovery_activity=False,
            discovery_event_observer=False,observe_P_submission_readiness=False,
            clean_confirmation=False,host_cost_trace=False)
        save(cfg);jobs.append(serial)
    write(ROOT/'state/huginn_gap_cost_plan.json',dict(jobs=jobs,
        scope='Serial CUDA Event enclosures after N8 exact state. These costs do not establish online Decode safety or feasible handoff completion.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
