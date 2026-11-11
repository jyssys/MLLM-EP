"""Use a Split contract, not the Native whole-depth assertion."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for length,reference in ((16384,'async_ouro_L16384_native_reference_state8'),
        (24576,'extended_ouro_L24576_native_reference_state8')):
        source=json.loads((ROOT/'configs'/(reference+'.json')).read_text())
        if length==24576:
            baseline=dict(source);base=f'cost_ouro_L{length}_zero_serial8'
            baseline.update(name=base,state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
                async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
                observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
                required_qualification=reference,serial_output_reference=reference,serial_baseline=base,
                load='serial',arrival_pattern='SERIAL_CALIBRATION',phase='SERIAL_NATIVE_24K_COST_REFERENCE',
                discovery_activity=False,discovery_event_observer=False,observe_P_submission_readiness=False,
                clean_confirmation=False,host_cost_trace=False)
            save(baseline);jobs.append(base)
        name=f'recovered_cost_ouro_L{length}_loop3_full_state8'
        cfg=dict(source);cfg.update(name=name,method='loop',axis='loop',cut=3,
            native_partial_cut=3,native_frontier=None,atomic_prompt=False,
            D_prefill_loop_cap=1,D_prefill_token_cap=None,D_prefill_cap_includes_prelude=True,
            campaign_state_reference=reference,required_qualification=reference,qualification_fresh_native=False,
            phase='EXACT_FULL_PROMPT_PARTIAL_DEPTH_PRIMITIVE',
            require_realized_quantum_contract=True,priority_serial_comparison=False,
            contract_reason='The earlier configuration retained atomic_prompt=True. Its Native-only assertion requires all R4 on P and correctly rejected a three-loop P batch during warm-up. A Split uses full-work, exact state, per-request D rows, and native lifetime gates instead.')
        cfg.pop('qualification_output_reference',None);save(cfg);jobs.append(name)
        serial=dict(cfg);name=f'recovered_cost_ouro_L{length}_loop3_full_serial8'
        serial.update(name=name,state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
            async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
            observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
            required_qualification=cfg['name'],serial_output_reference=reference,
            serial_baseline=f'cost_ouro_L{length}_zero_serial8',
            load='serial',arrival_pattern='SERIAL_CALIBRATION',phase='SERIAL_PARTIAL_DEPTH_COST_NOT_SERVING',
            discovery_activity=False,discovery_event_observer=False,observe_P_submission_readiness=False,
            clean_confirmation=False,host_cost_trace=False)
        save(serial);jobs.append(name)
    write(ROOT/'state/recovered_loop_primitive_plan.json',dict(jobs=jobs,
        scope='The failed namespace remains unchanged. No runtime assertion or Native guard is removed. Correct Split configuration precedes exact N8 state and serial cost. These forced primitives are not Phase 2 Serving.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
