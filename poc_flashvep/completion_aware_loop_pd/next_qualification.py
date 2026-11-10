"""Expand exact state coverage only after the corresponding N8 test passes."""
from .calibrate import *
from .analyze import qualify_frozen_reference


def main():
    initialize();jobs=[]
    for model in ('ouro','huginn'):
        for method in ('loop','token'):
            name=f'q1_{model}_{method}_state8'
            checked=qualify_frozen_reference(name)
            assert checked['passed'] and checked['actual_dynamic_continuation_tested'],(name,checked)
            first=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            cfg=new_config(model,f'q2r2_{model}_{method}_state32',method,count=32)
            for k in ('state_probe','verify_transfer','async_prompt_state_probe','async_verify_handoff',
                'state_probe_measured_only','state_hash_defer_until_close','qualification_ignore_observer_tail_latch'):
                cfg[k]=first[k]
            cfg.update(state_pinned_slots=34,completion_deadline_ms=20000.,
                arrival_offsets=[(6. if model=='ouro' else 10.)*(i//8) for i in range(32)],
                arrival_seed=None,arrival_rate=None,arrival_pattern='FOUR_EIGHT_REQUEST_STATE_CLUSTERS',
                phase='N32_NEW_EXACT_STATE_QUALIFICATION_NOT_PERFORMANCE',
                timeout=1800,startup_timeout=300,required_qualification=name)
            cfg['native_transport_limits']=dict(cfg.get('native_transport_limits',{}),shutdown_timeout=300.)
            cfg['shutdown_scope']='Full-byte state observer finalization only. No serving deadline or Decode SLO changes. Prior attempt completed all outputs and all 32 state hashes before a 30-second worker shutdown deadline.'
            save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/N32_jobs.json',jobs);print(json.dumps(jobs))


if __name__=='__main__':main()
