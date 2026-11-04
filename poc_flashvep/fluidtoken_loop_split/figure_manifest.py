"""List real figures and missing requested evidence without placeholder curves."""
from .common import *


def main():
    configure()
    rows = []
    historical = PREVIOUS / 'figures/final_duetserve_style_controls.png'
    for model in ('ouro', 'huginn'):
        rows.append(dict(number=1, model=model, topic='QPS motivation controls',
            status='HISTORICAL_REFERENCE_NOT_NEW_SWEEP',
            path=str(historical.relative_to(REPO)) if historical.exists() else None,
            source_commit=BASE,
            limitation='No new Aggregated-DP2 QPS sweep is supplied by this recovery and split campaign. Historical settings must not be relabeled as current controls.'))
        if model == 'huginn':
            target = ROOT / 'figures/recovery_huginn_paced8_v5_dual_activity512_joint_device.png'
            gate_path = ROOT / 'state/dual_device_paced_recovery_control.json'
            status = json.loads(gate_path.read_text())['status'] if gate_path.exists() else 'NOT_MEASURED'
        else:
            target = PREVIOUS / 'figures/persistence_ouro_nsys512_same_window.png'
            status = 'HISTORICAL_MEASURED_REFERENCE'
        rows.append(dict(number=2, model=model, topic='Same-run P pressure and D headroom',
            status=status if target.exists() else 'NOT_MEASURED',
            path=str(target.relative_to(REPO)) if target.exists() else None,
            limitation='Independent local probes are not migrated serving work. Kernel union is not SM occupancy.'))
        target = ROOT / 'figures' / f'{model}_static_method_screen.png'
        rows.append(dict(number=3, model=model, topic='Method comparison',
            status='STATIC_EXPLORATORY_SCREEN' if target.exists() else 'NOT_MEASURED',
            path=str(target.relative_to(REPO)) if target.exists() else None,
            limitation='Only completed controls appear. This static figure does not substitute for missing dynamic measurements.'))
        rows.append(dict(number=4, model=model, topic='Split-ratio sweep', status='NOT_MEASURED',
            path=None, limitation='A single partial boundary and transport controls are not a tuned split-ratio sweep.'))
        rows.append(dict(number=5, model=model, topic='Loop-specific advantage',
            status='NO_CONFIRMED_LOOP_ADVANTAGE', path=None,
            limitation='No advantage figure is fabricated. Consult exactness, padding controls, and measured method results.'))
    write(ROOT / 'figure_manifest.json', dict(figures=rows,
        scope='Every path points to an existing figure. Missing comparisons retain explicit status.'))
    print(str(ROOT / 'figure_manifest.json'))


if __name__ == '__main__':
    main()
