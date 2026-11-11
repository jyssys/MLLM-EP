"""Separate Native state checks from a failed collector/observer combination."""
import json
from .common import *


def main():
    initialize();jobs=[]
    for count in (8,32):
        cfg=json.loads((ROOT/'configs'/f'native_huginn_early_profile_state{count}.json').read_text())
        name=f'native_huginn_clean_recovery_state{count}'
        cfg.update(name=name,discovery_activity=False,phase='FRESH_NATIVE_STATE_GATE_WITHOUT_ACTIVITY_COLLECTOR',
            required_qualification='native_huginn_clean_recovery_state8' if count==32 else None)
        save(cfg);jobs.append(name)
    cfg=json.loads((ROOT/'configs'/'native_huginn_early_profile_state8.json').read_text())
    cfg.update(name='native_huginn_profile_syncprobe_state8',discovery_activity=True,
        async_prompt_state_probe=False,async_verify_handoff=False,verify_transfer=True,
        phase='COLLECTOR_WITH_SYNCHRONOUS_FULL_STATE_OBSERVER_DIAGNOSTIC',
        required_qualification='native_huginn_clean_recovery_state32')
    save(cfg);jobs.append(cfg['name'])
    old=json.loads((ROOT/'state/initial_anchor_controls.json').read_text())['jobs'];controls=[]
    for name in old:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        cfg.update(name=name.replace('_N64_','_v2_N64_'),
            required_qualification='native_huginn_clean_recovery_state32' if cfg['model']=='huginn' else 'native_ouro_early_profile_state32',
            phase='MATCHED_EARLY_COLLECTOR_NATIVE_A_B_A_V2_AFTER_STATE_OBSERVER_STALL')
        save(cfg);controls.append(cfg['name'])
    write(ROOT/'state/anchor_controls_v2.json',dict(jobs=controls,
        scope='Same frozen workloads and arrivals as the unrun initial anchors. Native state qualification is separate from intrusive state-hash/CUPTI interaction. No SLO changes.'))
    write(ROOT/'state/recovery_execution_plan.json',dict(jobs=controls[:3]+jobs+controls[3:],
        rationale='Run the qualified Ouro controls first. Recheck Huginn without CUPTI, then isolate its asynchronous-state observer. All GPU jobs remain serial.'))
    print(' '.join(controls[:3]+jobs+controls[3:]))


if __name__=='__main__':main()
