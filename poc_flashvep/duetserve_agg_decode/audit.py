"""Check independent starts, frozen settings, arrivals, and publication eligibility."""
import collections
import json

from .common import ROOT, digest, preserve, write


def main():
    preserve()
    groups = collections.defaultdict(list)
    failures = []
    completed = []
    for path in sorted((ROOT / 'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if not isinstance(cfg, dict) or 'arrival_offsets' not in cfg or 'name' not in cfg:
            continue
        folder = ROOT / 'raw' / cfg['name']
        receipt = folder / 'dispatcher_result.json'
        if not receipt.exists():
            continue
        value = json.loads(receipt.read_text())
        if not value['complete']:
            failures.append(dict(name=cfg['name'], receipt=value))
            continue
        completed.append(cfg['name'])
        if cfg['name'].startswith('confirm_'):
            family = 'FRESH_CONFIRMATION'
        elif cfg['name'].startswith(('sweep_', 'concurrency_')):
            family = 'DISCOVERY'
        else:
            continue
        key = (family, cfg['arm'], cfg['token_budget'], cfg['D_slots'], cfg['cap'], cfg['arrival_seed'])
        groups[key].append(cfg)
    fixed = []
    keys = ('token_budget', 'D_slots', 'chunk', 'placement', 'ar_cudagraph_mode',
            'ar_async_scheduling', 'ar_compile_mode', 'fixed_KV_bytes',
            'explicit_cudagraph_capture_sizes', 'gpu_phase_mask_norm', 'direct_phase_norm')
    for key, configs in groups.items():
        settings = {json.dumps({k: c[k] for k in keys}, sort_keys=True) for c in configs}
        fixed.append(dict(group=key, starts=[c['name'] for c in configs],
                          settings_identical=len(settings) == 1, settings=json.loads(next(iter(settings)))))
    arrivals = collections.defaultdict(list)
    for rows in groups.values():
        for cfg in rows:
            key = (cfg['dataset'], cfg['cap'], cfg['arrival_seed'], cfg['arrival_rate'])
            arrivals[key].append(cfg)
    comparisons = []
    for key, cfgs in arrivals.items():
        unique = {json.dumps((c['arrival_offsets'], c['request_caps'])) for c in cfgs}
        comparisons.append(dict(group=key, starts=[c['name'] for c in cfgs], identical=len(unique) == 1))
    passed = all(v['settings_identical'] for v in fixed) and all(v['identical'] for v in comparisons)
    write(ROOT / 'analysis/frozen_settings_and_arrival_audit.json', dict(passed=passed,
        complete_starts=len(completed), completed=completed, failed_starts=failures,
        fixed_settings=fixed, arrival_comparisons=comparisons,
        source_spec_SHA256=digest(ROOT.parent / 'duetserve_agg_dp2_tp2_tbt_crossover_reproduction_spec.md'),
        evidence='TRACE_ANALYSIS',
        scope='Configurations, not achieved service rates, define the intended arrivals. Graph/concurrency sensitivity groups remain distinct. Failed starts remain listed and are never silently replaced.'))
    assert passed
    print('FROZEN_SETTINGS_AND_ARRIVALS_PASS', len(completed), 'FAILED_PRESERVED', len(failures))


if __name__ == '__main__':
    main()
