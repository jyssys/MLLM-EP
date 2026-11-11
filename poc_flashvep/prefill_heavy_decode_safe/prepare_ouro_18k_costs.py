"""Keep new-context primitive evidence separate from online migration."""
import json
from .common import *


def main():
    initialize();reference='extended_ouro_L18432_native_reference_state8'
    source=json.loads((ROOT/'configs'/(reference+'.json')).read_text())
    baseline='gapcost_ouro_L18432_zero_serial8'
    common=dict(state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
        async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
        observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
        serial_output_reference=reference,serial_baseline=baseline,
        load='serial',arrival_pattern='SERIAL_CALIBRATION',discovery_activity=False,
        discovery_event_observer=False,observe_P_submission_readiness=False,
        clean_confirmation=False,host_cost_trace=False,phase='OURO_18432_SERIAL_ENCLOSURE_NOT_SERVING')
    cfg=dict(source);cfg.update(common);cfg.update(name=baseline,required_qualification=reference)
    save(cfg)
    state='gapcost_ouro_L18432_loop3_full_state8';cfg=dict(source)
    cfg.update(name=state,method='loop',axis='loop',cut=3,native_partial_cut=3,native_frontier=None,
        atomic_prompt=False,D_prefill_loop_cap=1,D_prefill_token_cap=None,
        D_prefill_cap_includes_prelude=True,campaign_state_reference=reference,
        qualification_fresh_native=False,required_qualification=reference,
        require_realized_quantum_contract=True,state_hash_defer_until_close=False,state_pinned_slots=6,
        phase='OURO_18432_LOOP3_EXACT_STATE_NOT_ONLINE_MIGRATION')
    cfg.pop('qualification_output_reference',None);save(cfg)
    name=state.replace('_state8','_serial8');cfg=dict(cfg);cfg.update(common)
    cfg.update(name=name,required_qualification=state,calibration_serial_output_reference=reference)
    save(cfg);write(ROOT/'state/ouro_18k_gap_cost_plan.json',dict(jobs=[baseline,state,name]))
    print(baseline,state,name)


if __name__=='__main__':main()
