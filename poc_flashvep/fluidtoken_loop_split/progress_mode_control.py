"""Test one supported progress setting after exact native state validation."""
import copy
from .common import *
from .run import execute
from .recovery_qualification import save
from .recovery_analysis import analyze
from .state_comparison import compare as compare_states
from .confirm_recovery import operational


def main():
    initialize()
    from ..looped_pd_autonomous_headroom.campaign import gate
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_headroom_recovery.profile_analysis import analyze as parse
    cfg = json.loads((ROOT / 'configs/recovery_huginn_graph_state8.json').read_text())
    cfg.update(name='recovery_huginn_manual_progress_state8', nixl_enable_progress_thread=False)
    save(cfg)
    complete = execute(cfg['name'])
    result = dict(complete=complete, change='NIXL enable_prog_thread=False only')
    if complete:
        result.update(native_state_gate=gate(cfg['name']),
                      state=compare_states('recovery_huginn_graph_state8', cfg['name']),
                      output=evidence.compare('recovery_huginn_graph_state8', cfg['name']))
    write(ROOT / 'state/manual_progress_control.json', result)
    if not (complete and result['native_state_gate'] and result['state']['exact'] and result['output']['passed']):
        return
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_extended1_clean512.json').read_text()))
    cfg.update(name='recovery_huginn_manual_progress_clean512', nixl_enable_progress_thread=False)
    save(cfg)
    if not execute(cfg['name']):
        return
    clean = cfg['name']
    profile = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_extended1_profile512.json').read_text()))
    profile.update(name='recovery_huginn_manual_progress_profile512', matched_clean=clean,
                   nixl_enable_progress_thread=False, native_stack_diagnostic=True)
    save(profile)
    result['profile_complete'] = execute(profile['name'])
    if result['profile_complete']:
        with quiet_analysis(profile['name'] + '_parse'):
            result['trace'] = parse(profile['name'], clean)
        result['screen'] = analyze(profile['name'])
        result['operational_pass'] = operational(result['screen'])
    result['scope'] = 'One supported native setting control. A successful start does not prove the historical root cause.'
    write(ROOT / 'state/manual_progress_control.json', result)


if __name__ == '__main__':
    main()
