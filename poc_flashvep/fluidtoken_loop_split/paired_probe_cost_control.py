"""Use the identical arrival trace to measure the local-probe intervention cost."""
import copy
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    from .conditional_split_qualification import recovery_ready
    gate = json.loads((ROOT / 'state/huginn_final_recovery_gate.json').read_text())
    assert recovery_ready(gate)
    source_name = gate['results'][0]['clean']
    source = json.loads((ROOT / 'configs' / (source_name + '.json')).read_text())
    cfg = copy.deepcopy(source)
    cfg.update(name='recovery_huginn_paced8_v5_s4_paired_no_probe512',
               phase='EXACT_ARRIVAL_PAIRED_NO_PROBE_CONTROL')
    for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'):
        cfg.pop(key, None)
    changes = {key for key in source.keys() | cfg.keys() if source.get(key) != cfg.get(key)}
    assert changes == {'name', 'phase', 'D_probe_tokens', 'D_probe_count', 'D_probe_interval_s'}
    assert cfg['D_probe_warmup_tokens'] == source['D_probe_warmup_tokens']
    save(cfg)
    complete = execute(cfg['name'])
    result = dict(name=cfg['name'], probe_run=source_name, complete=complete,
                  config_differences=sorted(changes), same_arrival_offsets=True,
                  scope='One paired clean start. Same prompts, arrival trace, memory, and execution geometry. Not a replicated causal estimate.')
    if complete:
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from ..looped_pd_atomic_prefill import evidence
        from .static_screen import metrics
        with quiet_analysis(cfg['name'] + '_paired_analysis'):
            result.update(no_probe_metrics=metrics(cfg['name']), probe_metrics=metrics(source_name),
                          output=evidence.compare(source_name, cfg['name']), work=evidence.full_work(cfg['name']))
    result['correct'] = bool(complete and result['output']['passed'] and result['work']['passed'])
    write(ROOT / 'state/paired_probe_cost_control.json', result)


if __name__ == '__main__':
    main()
