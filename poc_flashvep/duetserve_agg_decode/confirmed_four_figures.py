"""Save the four start-level confirmation plots next to the user's discovery plots."""
import csv
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from .common import QUIET_LOCK, REPO, ROOT, digest, resources, write
from .statistics import mean_ci


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    revised = (ROOT / 'state/user_primary_scope_revision_v2.json').exists()
    receipt = 'primary_confirmation_complete_v2.json' if revised else 'fresh_confirmation_complete.json'
    assert json.loads((ROOT / 'state' / receipt).read_text())['status'] == 'COMPLETE'
    existing = REPO / 'poc_flashvep/duetserve_qps_four_metrics_confirmed_o200_metadata.json'
    if existing.exists():
        manifest = json.loads(existing.read_text())
        assert all(digest(path) == sha for path, sha in manifest['output_SHA256'].items())
        assert digest(manifest['data_csv']) == manifest['data_csv_SHA256']
        assert digest(ROOT / 'configs/frozen_SLO_regimes.json') == manifest['SLO_SHA256']
        for row in csv.DictReader(Path(manifest['data_csv']).open()):
            metric = json.loads((ROOT / 'raw' / row['run_name'] / 'metrics.json').read_text())
            for key in ('realized_arrival_QPS', 'mean_TTFT_ms', 'mean_TBT_ms', 'output_tokens_s'):
                assert float(row[key]) == metric[key]
            assert float(row['joint_SLO_attainment_percent']) == 100 * metric['SLO_regimes']['Balanced']['Joint_attainment']
        print('EXISTING_CONFIRMED_PLOTS_VERIFIED_WITHOUT_OVERWRITE', flush=True)
        return
    methods = (
        ('Agg-DP2 B8192 / S1024', 'confirm_DP2_B8192_S1024_FULL_N128_O200_q{}_s{}_v1', '#d55e00', 'o'),
        ('Agg-DP2 B512 / S16', 'confirm_DP2_B512_S16_FULL_N128_O200_q{}_s{}_v1', '#0072b2', 's'),
        ('Native PD 1P1D', 'confirm_Native_N128_O200_q{}_s{}_v1', '#009e73', '^'),
    )
    if revised:
        methods = tuple(row for row in methods if 'B512' not in row[0])
    seeds = (2026111622, 2026111623, 2026111624)
    destination = REPO / 'poc_flashvep'
    rows, points = [], {}
    for label, pattern, _, _ in methods:
        for qps in range(2, 11):
            selected = []
            for seed in seeds:
                name = pattern.format(qps, seed)
                path = ROOT / 'raw' / name / 'metrics.json'
                metric = json.loads(path.read_text())
                assert metric['complete'] and metric['output_gate']['passed']
                selected.append(metric)
                rows.append(dict(method=label, offered_QPS=qps, arrival_seed=seed, run_name=name,
                    realized_arrival_QPS=metric['realized_arrival_QPS'],
                    mean_TTFT_ms=metric['mean_TTFT_ms'], mean_TBT_ms=metric['mean_TBT_ms'],
                    joint_SLO_attainment_percent=100 * metric['SLO_regimes']['Balanced']['Joint_attainment'],
                    output_tokens_s=metric['output_tokens_s'], metrics_SHA256=digest(path)))
            points[label, qps] = {key: mean_ci(m[key] for m in selected)
                for key in ('mean_TTFT_ms', 'mean_TBT_ms', 'output_tokens_s')}
            points[label, qps]['joint_SLO_attainment_percent'] = mean_ci(
                100 * m['SLO_regimes']['Balanced']['Joint_attainment'] for m in selected)
    plots = (
        ('joint_SLO_attainment_percent', 'Joint SLO attainment (%)', 'SLO attainment', 'duetserve_qps_slo_confirmed_o200'),
        ('mean_TTFT_ms', 'Mean TTFT (ms)', 'Mean TTFT', 'duetserve_qps_mean_ttft_confirmed_o200'),
        ('mean_TBT_ms', 'Mean TBT (ms)', 'Mean TBT', 'duetserve_qps_mean_tbt_confirmed_o200'),
        ('output_tokens_s', 'Generated output tokens/s', 'Generation throughput', 'duetserve_qps_output_tokens_s_confirmed_o200'),
    )
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9, 'pdf.fonttype': 42})
    x = list(range(2, 11))
    def draw(ax, key, ylabel, title):
        for label, _, color, marker in methods:
            values = [points[label, qps][key] for qps in x]
            ax.plot(x, [v['mean'] for v in values], color=color, marker=marker,
                    linewidth=1.8, markersize=5, label=label)
            ax.fill_between(x, [v['CI95'][0] for v in values], [v['CI95'][1] for v in values],
                            color=color, alpha=.15)
        ax.set(xlabel='Offered QPS (requests/s)', ylabel=ylabel, title=title, xticks=x)
        ax.set_xlim(1.8, 10.2)
        ax.set_ylim(bottom=0)
        if key == 'joint_SLO_attainment_percent':
            ax.set_ylim(-2, 102)
        ax.grid(alpha=.25)
    title = 'Qwen3-8B | 2 x H100 | Input8000 / Output200 | Frozen fresh-prompt confirmation'
    footnote = 'N128 x 3 independent server starts/QPS. Shading: empirical start-level bootstrap95% CI (27 resamples).'
    scope = 'Balanced SLO: TTFT <=5000ms, every ITL <=34ms, TPOT <=33ms. Output-only finite-cohort throughput, not sustainable capacity.'
    outputs = []
    def save(fig, stem):
        for suffix in ('.png', '.pdf'):
            path = destination / (stem + suffix)
            assert not path.exists(), 'Preserve the existing figure: ' + str(path)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs.append(str(path))
        plt.close(fig)
    for key, ylabel, panel, stem in plots:
        fig, ax = plt.subplots(figsize=(8.2, 5.3))
        draw(ax, key, ylabel, panel)
        ax.legend(loc='best')
        fig.suptitle(title, fontsize=9.5, y=.99)
        fig.text(.025, .035, footnote, fontsize=7.5)
        fig.text(.025, .012, scope, fontsize=6.9)
        fig.tight_layout(rect=(0, .08, 1, .95))
        save(fig, stem)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    for ax, (key, ylabel, panel, _) in zip(axes.flat, plots):
        draw(ax, key, ylabel, panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=3, fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.text(.025, .041, footnote, fontsize=9)
    fig.text(.025, .015, scope, fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .95))
    save(fig, 'duetserve_qps_four_metrics_confirmed_o200')
    csv_path = destination / 'duetserve_qps_four_metrics_confirmed_o200_data.csv'
    assert not csv_path.exists()
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(destination / 'duetserve_qps_four_metrics_confirmed_o200_metadata.json', dict(time=time.time(),
        outputs=outputs, data_csv=str(csv_path), seeds=seeds, independent_starts_per_point=3,
        output_SHA256={path: digest(path) for path in outputs}, data_csv_SHA256=digest(csv_path),
        SLO_SHA256=digest(ROOT / 'configs/frozen_SLO_regimes.json'),
        evidence='MEASURED', scope='Fresh prompts and frozen settings. This does not overwrite the single-start discovery plots.'))
    print('CONFIRMED_FOUR_PLOTS_SAVED', len(outputs), flush=True)


def main():
    if '--wait' in sys.argv:
        write(ROOT / 'state/confirmed_plots_controller.json', dict(pid=os.getpid(), time=time.time(),
            scope='CPU-only rendering waits for all54 frozen Primary confirmations.'))
        until = time.monotonic() + 14400
        while not (ROOT / 'state/primary_confirmation_complete_v2.json').exists():
            assert time.monotonic() < until
            time.sleep(15)
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if not resources().processes():
            resources().restore(list(range(8)))
        render()


if __name__ == '__main__':
    main()
