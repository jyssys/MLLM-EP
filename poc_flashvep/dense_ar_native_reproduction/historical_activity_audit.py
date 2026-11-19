"""Verify the older censored observation without changing historical evidence."""
import json
from pathlib import Path
from .common import REPO, ROOT, digest, write


def main():
    old = REPO / 'poc_flashvep/reports/dense_ar_layerwise_pd_feasibility_20261112'
    name = 'eager_physical_qwen8_L8000_O16_N128_s25_quiet_v23'
    raw = old / 'raw' / name
    paths = dict(screen=old / 'analysis/eager_physical_s25_screen_v23.json',
                 configuration=raw / 'configuration.json',
                 source_hashes=raw / 'source_hashes.json')
    screen = json.loads(paths['screen'].read_text())
    cfg = json.loads(paths['configuration'].read_text())
    hashes = json.loads(paths['source_hashes'].read_text())
    verified = []
    for path, expected in hashes.items():
        p = Path(path)
        if p.parent.name == 'dense_ar_layerwise_pd' and p.name in (
                'native.py', 'extension.py', 'run.py'):
            observed = digest(p)
            assert observed == expected, path
            verified.append(dict(path=path, SHA256=observed))
    assert len(verified) == 3
    assert screen['trace_qualified'] and screen['interference_pass']
    assert cfg['length'] == 8000 and cfg['P_slots'] == 4 and cfg['D_slots'] == 16
    write(ROOT / 'analysis/historical_L8000_slots4_16_activity_audit.json', dict(
        evidence='HISTORICAL_TRACE_ANALYSIS', name=name,
        files={k: dict(path=str(p.relative_to(REPO)), SHA256=digest(p)) for k, p in paths.items()},
        verified_runtime_sources=verified,
        window_seconds=screen['P_activity']['window_seconds'],
        P_kernel_busy_percent=screen['P_activity']['kernel_busy_percent'],
        D_observed_kernel_inactive_lower_bound_percent=screen['D_quiet_activity']['physical_inactive_lower_bound_percent'],
        D_full_kernel_busy='NOT_MEASURED',
        observer_ratios=screen['observer_ratios'],
        settings={k: cfg[k] for k in ('length', 'cap', 'P_slots', 'D_slots', 'D_admission_batch',
            'token_budget', 'ar_enforce_eager', 'ar_async_scheduling', 'cohort_barrier')},
        scope='One qualified eager start. D capture is censored to fenced empty-engine epochs. '
              'It is not a full D busy ratio, a safe window, or a matched A/B with the new continuous scheduler.'))


if __name__ == '__main__':
    main()
