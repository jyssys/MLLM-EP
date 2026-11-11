"""Use legal KV capacity to separate compute pressure from credit waiting."""
import json
from .common import *


def main():
    initialize();states=[];jobs=[]
    for model,length in (('ouro',16384),('huginn',3900),('huginn',3072)):
        original=f'coarse_{model}_L{length}_O16_P2_mu115_N64_a2026111230'
        cfg=json.loads((ROOT/'configs'/(original+'.json')).read_text());name=original.replace('coarse_','memory_')
        cfg.update(name=name,kv_blocks=24832 if model=='huginn' else 16384,
            D_kv_blocks=18432 if model=='ouro' else 24832,
            phase='LEGAL_KV_CAPACITY_RECOVERY_NATIVE_ZERO',comparison_reference=original,
            selection_reason='P2 pilots frequently report KV refusals. Increase only legal pool capacity before classifying the growing queue. No model, SLO, arrival, slot, or arithmetic change.')
        state=dict(cfg);state_name=f'memory_{model}_L{length}_native_reference_state8'
        template=json.loads((ROOT/'configs'/f'native_{model}_{"early_profile" if model=="ouro" else "clean_recovery"}_state8.json').read_text())
        for key in ('state_probe','verify_transfer','async_prompt_state_probe','async_verify_handoff','state_probe_measured_only',
            'state_hash_defer_until_close','qualification_ignore_observer_tail_latch'):
            if key in template:state[key]=template[key]
        state.update(name=state_name,count=8,prompt_ids=cfg['prompt_ids'][:8],request_caps=[16]*8,
            arrival_offsets=[0.]*8,state_pinned_slots=10,discovery_activity=False,discovery_event_observer=False,
            qualification_fresh_native=True,qualification_output_reference=original,
            phase='FRESH_NEW_LENGTH_NATIVE_FULL_STATE_REFERENCE',timeout=1200,startup_timeout=300)
        save(state);states.append(state_name)
        cfg['required_qualification']=state_name;save(cfg);jobs.append(name)
    write(ROOT/'state/memory_recovery_plan.json',dict(qualification=states,jobs=jobs,
        scope='Read-only admission logs identify P and D KV refusal separately. Native references qualify before the corresponding pilot. Fresh state bytes never use a different-length golden hash.'))
    print(' '.join([item for pair in zip(states,jobs) for item in pair]))


if __name__=='__main__':main()
