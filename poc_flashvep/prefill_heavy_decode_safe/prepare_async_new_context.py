"""Qualify the existing exact asynchronous observer at the new context."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for length,count in ((4096,8),(16384,8),(16384,32)):
        source=f'native_ouro_early_profile_state8' if length==4096 else 'memory_ouro_L16384_native_reference_state8'
        cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text())
        name=f'async_ouro_L{length}_native_reference_state{count}'
        cfg.update(name=name,async_prompt_state_probe=True,async_verify_handoff=True,
            verify_transfer=False,state_probe=True,state_probe_measured_only=True,
            state_hash_defer_until_close=False,state_pinned_slots=6,discovery_activity=False,
            discovery_event_observer=False,qualification_fresh_native=True,
            phase='EXACT_ASYNC_OBSERVER_QUALIFICATION_NOT_SERVING',count=count,
            arrival_offsets=[0.]*count,request_caps=[cfg['cap']]*count,timeout=1200)
        if length==4096:
            cfg['campaign_state_reference']=source
        elif count==8:
            cfg['campaign_state_reference']=source
            cfg['required_qualification']='async_ouro_L4096_native_reference_state8'
        else:
            parent=json.loads((ROOT/'configs'/'memory_ouro_L16384_O16_P2_mu115_N64_a2026111230.json').read_text())
            cfg['prompt_ids']=parent['prompt_ids'][:count]
            cfg.pop('campaign_state_reference',None)
            cfg['required_qualification']='async_ouro_L16384_native_reference_state8'
            cfg['qualification_output_reference']='memory_ouro_L16384_O16_P2_mu115_N64_a2026111230'
        save(cfg);jobs.append(name)
    write(ROOT/'state/async_new_context_plan.json',dict(jobs=jobs,
        scope='Read-only full-byte copies and hashes. N8 compares both new observers to exact synchronous references. N32 checks full state coverage and the existing Native output subset. No correctness bytes are omitted. Observer timing is not Serving latency.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
