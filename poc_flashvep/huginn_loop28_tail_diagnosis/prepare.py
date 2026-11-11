"""Prepare short diagnostic controls. Do not change the prior experiment."""
import shutil
from .common import *

SOURCES = dict(zero='lifetime3_deferred2_zero_seed2026110940',
    full='match18_loop28_full_seed2026110940',
    tiled='match18_loop28_tile256_seed2026110940',
    loop31='quantum6_loop31_512_seed2026110940')


def main():
    initialize()
    for name in ('heldout_huginn_L2048.json',):
        shutil.copy2(OLD_ROOT/'data'/name,ROOT/'data'/name)
    shutil.copy2(OLD_ROOT/'configs/frozen_SLO.json',ROOT/'configs/frozen_SLO.json')
    jobs=[]
    for label, source in SOURCES.items():
        original=json.loads((OLD_ROOT/'configs'/(source+'.json')).read_text())
        parent=OLD_ROOT/'raw'/original['required_qualification']/'qualification_with_frozen_reference.json'
        assert json.loads(parent.read_text())['campaign_gate_passed']
        for mode in ('clean','events'):
            cfg=dict(original)
            name=f'tail_{label}_{mode}_n64'
            cfg.update(name=name,count=64,phase='DIAGNOSTIC_NOT_NEW_PERFORMANCE_BASELINE',
                required_qualification=None,timeout=420,startup_timeout=240,
                request_caps=original['request_caps'][:64],prompt_ids=original['prompt_ids'][:64],
                arrival_offsets=original['arrival_offsets'][:64],
                diagnosis_events=mode=='events',diagnosis_serialize=False,
                diagnosis_parent=dict(configuration=source,qualification=str(parent),
                    qualification_sha256=digest(parent)),
                clean_confirmation=False,matched_diagnostic_control=f'tail_{label}_clean_n64')
            if 'data_scope' in cfg:
                cfg['data_scope']=dict(cfg['data_scope'],distinct_prompt_ids=64,
                    scope='First 64 original controlled prompts. Short diagnosis, not the frozen N128 comparison.')
            save(cfg);jobs.append(name)
    cfg=json.loads((ROOT/'configs/tail_full_events_n64.json').read_text())
    cfg.update(name='tail_full_serial_events_n64',diagnosis_serialize=True,
        phase='SERIALIZED_DEVICE_COUNTERFACTUAL_NOT_OPTIMIZATION')
    save(cfg)
    # Add a device-ordering exact-state diagnostic before performance controls.
    original=json.loads((OLD_ROOT/'configs/match18_loop28_full_state8.json').read_text())
    original.update(name='tail_serial_state8',required_qualification=None,
        diagnosis_events=True,diagnosis_serialize=True,timeout=420,startup_timeout=240)
    save(original)
    repeat=json.loads((ROOT/'configs/tail_zero_clean_n64.json').read_text())
    repeat['name']='tail_zero_repeat_clean_n64';save(repeat)
    order=['tail_serial_state8','tail_zero_clean_n64','tail_full_clean_n64',
        'tail_tiled_clean_n64','tail_loop31_clean_n64','tail_zero_repeat_clean_n64',
        'tail_zero_events_n64','tail_full_events_n64','tail_tiled_events_n64',
        'tail_loop31_events_n64','tail_full_serial_events_n64']
    write(ROOT/'state/diagnostic_plan.json',dict(base=BASE,jobs=order,start=START,
        scope='Single matched Arrival trace. N64 controls quantify instrumentation sensitivity. Device serialization is a counterfactual, not a Serving fix. Prior measurements and SLOs remain frozen.'))
    print(' '.join(order))


if __name__=='__main__':main()
