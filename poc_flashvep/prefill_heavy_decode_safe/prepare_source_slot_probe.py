"""Test a generic Native admission improvement, not a Loop benefit."""
import json
from .common import *


def main():
    initialize();reference='reserve_huginn_L3072_native_reference_state8'
    cfg=json.loads((ROOT/'configs'/(reference+'.json')).read_text());state='source_slots3_huginn_L3072_native_state8'
    cfg.update(name=state,campaign_state_reference=reference,required_qualification=reference,
        qualification_fresh_native=False,discovery_P_transfer_slots=3,
        state_hash_defer_until_close=False,state_pinned_slots=6,
        phase='GENERIC_NATIVE_SOURCE_TRANSFER_SLOT_EXACT_STATE_PROBE',
        selection_reason='Synchronous Huginn usually submits one prompt at a time. Its coordinator currently allows only two source transfer leases. Test three leases with the same two compute slots. Reject this option if it weakens the strongest Native performance.')
    save(cfg)
    parent='reserve_huginn_L3072_O16_P2_mu115_N64_a2026111230'
    cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('reserve_','source_slots3_')
    cfg.update(name=name,discovery_P_transfer_slots=3,required_qualification=state,
        comparison_reference=parent,phase='GENERIC_NATIVE_SOURCE_TRANSFER_SLOT_MARKER_FREE_PROBE')
    save(cfg);write(ROOT/'state/source_slot_probe_plan.json',dict(jobs=[state,name],
        scope='No offload, no delayed batching, no extra compute slot, and no relaxed SLO. This is a generic Native control. A regression cannot establish the requested Prefill-heavy condition.'))
    print(state,name)


if __name__=='__main__':main()
