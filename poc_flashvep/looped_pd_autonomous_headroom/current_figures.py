"""Plot completed evidence without converting diagnostic traces into clean timing."""
import csv
from .common import *


def main():
    activate()
    from ..looped_pd_atomic_prefill import analyze
    analyze.ROOT = ROOT
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    components = []
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if not isinstance(cfg, dict) or 'name' not in cfg or cfg.get('capacity_mode'):
            continue
        folder = ROOT/'raw'/cfg['name']
        if (folder/'supervisor_recovery.json').exists():
            continue
        result = folder/'dispatcher_result.json'
        if not result.exists() or not json.loads(result.read_text()).get('complete'):
            continue
        if not (folder/'requests.jsonl').exists():
            continue
        row, requests, shares = analyze.serving_one(cfg)
        if row:
            row['P_prefill_graph_rows'] = cfg.get('P_prefill_graph_rows', 0)
            row['D_probe_tokens'] = cfg.get('D_probe_tokens', 0)
            row['collector_active'] = any(cfg.get(k) for k in (
                'nsys', 'P_only_profiler', 'cuda_only_profile', 'bounded_native_capture_seconds',
                'autonomous_capture_seconds', 'postwarm_capture'))
            rows.append(row)
            components.extend(shares)
    def table(name, values):
        keys = list(dict.fromkeys(k for row in values for k in row))
        with (ROOT/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in values:
                writer.writerow({k: 'NOT_MEASURED' if v is None else json.dumps(v) if isinstance(v, (dict, list)) else v
                                 for k, v in row.items()})
    table('completed_serving_current.csv', rows)
    table('completed_host_e2e_component_shares_current.csv', components)
    profiles = []
    for path in (ROOT/'raw').glob('*/profile_analysis.json'):
        if (path.parent/'supervisor_recovery.json').exists():
            continue
        value = json.loads(path.read_text())
        if value.get('status') != 'PARSED':
            continue
        cfg = json.loads((ROOT/'configs'/(path.parent.name+'.json')).read_text())
        if cfg.get('capture_role', 'prefill') != 'prefill':
            continue
        value.update(model=cfg['model'], length=cfg['length'], output_cap=cfg['cap'])
        profiles.append(value)
    table('completed_P_profile_current.csv', profiles)
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, model in zip(axes, ('ouro', 'huginn')):
        selected = [r for r in profiles if r['model'] == model]
        for i, row in enumerate(selected):
            valid = row.get('eligible_for_matched_clean', False)
            ax.scatter(i, 100*row['kernel_busy_fraction'], marker='o' if valid else 'x',
                       color='#157f3b' if valid else '#ae6c20')
        ax.axhline(80, color='black', linestyle='--', label='Frozen kernel-busy screen')
        ax.set_xticks(range(len(selected)), [r['name'].replace('search_', '').replace('deep_', '') for r in selected],
                      rotation=75, ha='right', fontsize=7)
        ax.set(title=model, ylabel='Measured kernel union (%)', ylim=(0, 100))
        ax.legend(fontsize=8)
    fig.suptitle('P capture windows: green circle = qualified, orange cross = perturbed\nKernel union is not SM occupancy or proof of compute saturation', fontsize=11)
    path = ROOT/'figures/current_P_kernel_evidence.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, model in zip(axes, ('ouro', 'huginn')):
        selected = [r for r in rows if r['model'] == model and r['D_probe_tokens'] and
                    not r['collector_active'] and 'reference' not in r['name']]
        for i, row in enumerate(selected):
            ax.scatter(i, row['request_TPOT_ms_p95']/row['TPOT_threshold_ms'], color='#1769aa', label='TPOT p95 / SLO' if i == 0 else None)
            ax.scatter(i, row['token_ITL_ms_p99']/row['ITL_threshold_ms'], marker='s', color='#a33928', label='ITL p99 / SLO' if i == 0 else None)
        ax.axhline(1, color='black', linestyle='--')
        ax.set_xticks(range(len(selected)), [r['name'] for r in selected], rotation=60, ha='right', fontsize=7)
        ax.set(title=model, ylabel='Whole-run latency / frozen SLO', ylim=(0, None))
        if selected:
            ax.legend(fontsize=8)
    fig.suptitle('Actual D-local prefill probes\nIndependent prompts, not migrated continuation or an E2E speedup', fontsize=11)
    path = ROOT/'figures/current_D_probe_SLO.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))
    write(ROOT/'current_figure_manifest.json', dict(time=time.time(), figures=figures,
        scope='Interim completed evidence. Different runs remain separate. No joint saturation verdict.'))
    print(json.dumps(dict(completed_serving=len(rows), profiles=len(profiles), figures=figures)))


if __name__ == '__main__':
    import fcntl
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
