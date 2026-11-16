"""Draw the four requested discovery plots at the user's exact destination."""
import csv
import fcntl
import json
import time

from .common import QUIET_LOCK, REPO, ROOT, digest, resources, write


def main():
    # Plotting and font initialization must not share a timing lane with a
    # serving experiment. The next queued experiment resumes after this lock.
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        owner = resources()
        if not owner.processes():
            owner.restore(list(range(8)))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        destination = REPO / 'poc_flashvep'
        methods = (
            ('Agg-DP2 B8192 / S1024', 'concurrency_DP2_B8192_S1024_FULL_N128_O200_q{}_s2026111611_v1', '#d55e00', 'o'),
            ('Agg-DP2 B512 / S16', 'sweep_DP2_B512_S16_FULL_N128_O200_q{}_s2026111611_v1', '#0072b2', 's'),
            ('Native PD 1P1D', 'sweep_Native_N128_O200_q{}_s2026111611_v2', '#009e73', '^'),
        )
        rows = []
        for label, pattern, _, _ in methods:
            for qps in range(2, 11):
                name = pattern.format(qps)
                path = ROOT / 'raw' / name / 'metrics.json'
                value = json.loads(path.read_text())
                config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
                assert value['complete'] and value['output_gate']['passed']
                assert value['count'] == 128 and value['output_cap'] == 200
                rows.append(dict(method=label, offered_QPS=qps,
                    realized_arrival_QPS=value['realized_arrival_QPS'],
                    scheduled_trace_QPS=value['scheduled_trace_QPS'],
                    mean_TTFT_ms=value['mean_TTFT_ms'], mean_TBT_ms=value['mean_TBT_ms'],
                    joint_SLO_attainment_percent=100 * value['SLO_regimes']['Balanced']['Joint_attainment'],
                    generated_output_tokens_s=value['output_tokens_s'],
                    observation_seconds=value['observation_seconds'], request_count=value['count'],
                    distinct_prompt_ids=value['distinct_prompt_ids'], run_name=name,
                    metrics_SHA256=digest(path), configuration_SHA256=digest(ROOT / 'configs' / (name + '.json')),
                    arrival_seed=config['arrival_seed']))
        plots = (
            ('joint_SLO_attainment_percent', 'Joint SLO attainment (%)', 'SLO attainment', 'duetserve_qps_slo_discovery_o200'),
            ('mean_TTFT_ms', 'Mean TTFT (ms)', 'Mean TTFT', 'duetserve_qps_mean_ttft_discovery_o200'),
            ('mean_TBT_ms', 'Mean TBT (ms)', 'Mean TBT', 'duetserve_qps_mean_tbt_discovery_o200'),
            ('generated_output_tokens_s', 'Generated output tokens/s', 'Generation throughput', 'duetserve_qps_output_tokens_s_discovery_o200'),
        )
        plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                             'legend.fontsize': 9, 'pdf.fonttype': 42})
        outputs = []
        def draw(ax, metric, ylabel, title):
            for label, _, color, marker in methods:
                selected = [row for row in rows if row['method'] == label]
                ax.plot([row['offered_QPS'] for row in selected], [row[metric] for row in selected],
                        color=color, marker=marker, linewidth=1.8, markersize=5, label=label)
            ax.set(xlabel='Offered QPS (requests/s)', ylabel=ylabel, title=title, xticks=list(range(2, 11)))
            ax.set_xlim(1.8, 10.2)
            ax.set_ylim(bottom=0)
            if metric == 'joint_SLO_attainment_percent':
                ax.set_ylim(-2, 102)
            ax.grid(alpha=.25)
        subtitle = 'Qwen3-8B | 2 x H100 | Input8000 / Output200 | Poisson arrivals'
        footnote = 'Preliminary discovery: N128, one independent start/QPS, seed2026111611. No confidence interval.'
        scope = 'Balanced SLO: TTFT <=5000ms, every ITL <=34ms, TPOT <=33ms. Token/s excludes input and uses finite-cohort drain time.'
        for metric, ylabel, title, stem in plots:
            fig, ax = plt.subplots(figsize=(8.2, 5.3))
            draw(ax, metric, ylabel, title)
            ax.legend(loc='best')
            fig.suptitle(subtitle, fontsize=10, y=.99)
            fig.text(.03, .035, footnote, fontsize=7.8)
            fig.text(.03, .012, scope, fontsize=7.1)
            fig.tight_layout(rect=(0, .08, 1, .95))
            for suffix in ('.png', '.pdf'):
                path = destination / (stem + suffix)
                assert not path.exists(), 'Preserve an existing figure: ' + str(path)
                fig.savefig(path, dpi=180, bbox_inches='tight')
                outputs.append(str(path))
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
        for ax, (metric, ylabel, title, _) in zip(axes.flat, plots):
            draw(ax, metric, ylabel, title)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=3, fontsize=10)
        fig.suptitle(subtitle, fontsize=14)
        fig.text(.03, .041, footnote, fontsize=9)
        fig.text(.03, .015, scope, fontsize=8)
        fig.tight_layout(rect=(0, .14, 1, .95))
        for suffix in ('.png', '.pdf'):
            path = destination / ('duetserve_qps_four_metrics_discovery_o200' + suffix)
            assert not path.exists(), path
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs.append(str(path))
        plt.close(fig)
        csv_path = destination / 'duetserve_qps_four_metrics_discovery_o200_data.csv'
        assert not csv_path.exists()
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = destination / 'duetserve_qps_four_metrics_discovery_o200_metadata.json'
        assert not metadata.exists()
        write(metadata, dict(time=time.time(), evidence='MEASURED', outputs=outputs, data_csv=str(csv_path),
            request_count=128, independent_starts_per_point=1, arrival_seed=2026111611,
            primary_workload=dict(input_tokens=8000, output_tokens=200), confidence_interval=None,
            SLO_regime='FROZEN_BALANCED', SLO=dict(TTFT_ms=5000, ITL_ms=34, TPOT_ms=33),
            scope='Completed discovery measurements only. Not the ongoing fresh confirmation. No TP2 curve is eligible for this primary comparison. Token/s counts only output tokens over each fully drained finite cohort. It is not sustainable capacity.'))
        print(json.dumps(dict(saved=outputs, data=str(csv_path), metadata=str(metadata))), flush=True)


if __name__ == '__main__':
    main()
