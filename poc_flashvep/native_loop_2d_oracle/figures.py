"""Plot measured values and mark unavailable oracle results explicitly."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common import *

LABELS = {'native': 'Native', 'zero': 'Zero-offload', 'staged_native': 'Staged Native',
          'snapshot_full': 'Full snapshot', 'loop3': 'Loop c3', 'loop2': 'Loop c2',
          'loop31': 'Loop c31', 'loop28': 'Loop c28', 'token3072': 'Token k3072',
          'token3840': 'Token k3840', 'token1984': 'Token k1984', 'token2016': 'Token k2016',
          '2d_c2_k2048': '2D c2/k2048', '2d_c3_k3072': '2D c3/k3072',
          '2d_c28_k1536': '2D c28/k1536', '2d_c31_k1024': '2D c31/k1024'}


def main():
    initialize()
    rows = json.loads((ROOT / 'serving_results.json').read_text())
    ratios = json.loads((ROOT / 'paired_start_ratios.json').read_text())
    envelope = json.loads((ROOT / 'measured_fixed_action_envelope.json').read_text())
    gates = json.loads((ROOT / 'state/effective_qualification.json').read_text())
    manifest = []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def select(model, method, load='medium', start=1, markers=True):
        return next(row for row in rows if row['model'] == model and row['method'] == method
                    and row['load'] == load and row['start'] == start and row['markers'] == markers
                    and not row['calibration'])

    def save(fig, name, sources, scope):
        fig.tight_layout()
        fig.savefig(ROOT / 'figures' / (name + '.png'), dpi=190, bbox_inches='tight')
        fig.savefig(ROOT / 'figures' / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(figure='figures/' + name + '.png', vector='figures/' + name + '.svg',
                             sources=sources, scope=scope))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sources = []
    for ax, model, cut in zip(axes, ('ouro', 'huginn'), ('loop3', 'loop31')):
        methods = ['native', 'staged_native', 'snapshot_full', 'zero', cut]
        selected = [select(model, method) for method in methods]
        sources += [row['name'] for row in selected]
        bars = ax.bar(range(len(methods)), [row['service_TTFT_ms_p95'] for row in selected],
                      color=['#444444', '#bbbbbb', '#dddddd', '#4c78a8', '#59a14f'])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.0f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=25, ha='right')
        ax.set_ylabel('Service TTFT p95 (ms)')
        ax.set_title(model.capitalize() + ': medium load, N128, start 1')
        ax.set_ylim(0, max(bar.get_height() for bar in bars) * 1.17)
    save(fig, 'figure_1_native_staged_continuation', sources,
         'Measured event-marked host request latency. No component is exclusive kernel time. Runtime controls have no separate marker screen.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, model in zip(axes, ('ouro', 'huginn')):
        metrics = ['TTFT_p95_ms', 'service_TTFT_ms_p95', 'E2E_p95_ms', 'requests_per_second']
        for i, metric in enumerate(metrics):
            item = ratios[model]['zero'][metric]
            y = item['geometric_mean_ratio']
            low, high = item['ratio_CI95']
            ax.errorbar(i, y, yerr=[[y-low], [high-y]], fmt='o', color='#4c78a8', capsize=5)
            ax.scatter([i-.12, i, i+.12], item['ratios'], s=18, color='#999999', zorder=1)
        ax.axhspan(.95, 1.05, color='#59a14f', alpha=.15, label='±5% target')
        ax.axhline(1, color='black', linewidth=.8)
        ax.set_xticks(range(4), ['TTFT p95', 'Service p95', 'E2E p95', 'Completed\nthroughput'])
        ax.set_ylabel('Zero-offload / Native ratio')
        ax.set_title(model.capitalize() + ': 3 paired starts, 95% t intervals')
        ax.legend(fontsize=9)
    save(fig, 'figure_2_zero_offload_uncertainty', ['paired_start_ratios.json'],
         'Measured paired start ratios. Log-scale t intervals assume normal start-level log ratios. Latency equivalence is not established.')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    sources = []
    for column, model in enumerate(('ouro', 'huginn')):
        methods = [row['method'] for row in rows if row['model'] == model and not row['calibration']
                   and not row['runtime_control'] and row['load'] == 'medium' and row['start'] == 1 and row['markers']]
        selected = [select(model, method) for method in methods]
        sources += [row['name'] for row in selected]
        x = np.arange(len(methods))
        axes[0, column].bar(x, [row['service_TTFT_ms_p95'] for row in selected],
            color=['#59a14f' if row['Decode_SLO_screen'] else '#e15759' for row in selected])
        axes[0, column].set_ylabel('Service TTFT p95 (ms)')
        axes[0, column].set_title(model.capitalize() + ': exact N32-qualified actions')
        axes[1, column].plot(x, [row['ITL_p99_ms'] for row in selected], 'o-', color='#b279a2')
        slo = selected[0]['frozen_SLO']['ITL_ms']
        axes[1, column].axhline(slo, color='#e15759', linestyle='--', label=f'Frozen ITL p99: {slo:.2f} ms')
        axes[1, column].set_ylabel('ITL p99 (ms)')
        axes[1, column].legend(fontsize=8)
        for ax in axes[:, column]:
            ax.set_xticks(x, [LABELS[m] for m in methods], rotation=40, ha='right')
        if model == 'huginn':
            axes[0, column].text(.02, .96, 'Token and deeper cuts: scheduler-blocked, not slow measurements',
                transform=axes[0, column].transAxes, va='top', fontsize=8)
    save(fig, 'figure_3_measured_actions_and_decode_tail', sources,
         'Measured medium start 1. Green means the observed frozen Decode screen passes. It does not prove statistical SLO safety or maximum-ITL protection.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model in zip(axes, ('ouro', 'huginn')):
        selected = [row for row in envelope if row['model'] == model and row['load'] == 'medium']
        values = [row['TTFT_p95_ms'] for row in selected]
        ax.bar(range(4), values, color=['#444444', '#4c78a8', '#f28e2b', '#59a14f'])
        ax.set_xticks(range(5), ['Native', 'Token family', 'Loop family', '2D family', 'Dynamic /\ncompute-only oracle'])
        ax.text(4, max(values)*.5, 'NOT\nVALIDATED', ha='center', color='#777777')
        ax.set_xlim(-.6, 4.6)
        ax.set_ylabel('Best measured passing TTFT p95 (ms)')
        ax.set_title(model.capitalize() + ': fixed-action envelope, not an oracle')
        for i, row in enumerate(selected):
            label = LABELS[row['selected']]
            if model == 'huginn' and row['family'] == 'token':
                label += '\nToken partial blocked'
            ax.text(i, values[i], label, ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, max(values) * 1.3)
    save(fig, 'figure_4_measured_envelope_oracle_unavailable', ['measured_fixed_action_envelope.json'],
         'Measured finite candidate subset only. Native and one-axis endpoints belong to the 2D family. No feasible dynamic oracle or rigorous compute-only upper bound exists.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sources = ['structural_frontiers_not_gpu_costs.json', 'effective_qualification.json', 'serving_results.json']
    for ax, model, depth in zip(axes, ('ouro', 'huginn'), (4, 32)):
        x = np.linspace(0, 100, 101)
        y = np.arange(depth + 1)
        image = ax.imshow(y[:, None]/depth * x[None, :], extent=(0, 100, -.5, depth+.5),
                          origin='lower', aspect='auto', cmap='Blues', vmin=0, vmax=100, alpha=.8)
        for method, entry in gates['models'][model]['qualification'].items():
            if not entry.get('passed') or method in ('native', 'zero'):
                continue
            cfg = json.loads((ROOT / 'configs' / (entry['config'] + '.json')).read_text())
            c, k = cfg.get('native_frontier') or [cfg['native_partial_cut'], 0]
            a = select(model, method)
            color = '#2ca02c' if a['Decode_SLO_screen'] else '#d62728'
            ax.scatter(100*(cfg['length']-k)/cfg['length'], depth-c, s=65, edgecolor='white', color=color, zorder=3)
            ax.annotate(LABELS[method], (100*(cfg['length']-k)/cfg['length'], depth-c),
                        xytext=(-5, 9), textcoords='offset points', ha='right', fontsize=7)
        ax.set_xlabel('Suffix assigned to D (%)')
        ax.set_ylabel('Remaining recurrent loops on D')
        ax.set_title(model.capitalize() + ': one-handoff staircase geometry')
        fig.colorbar(image, ax=ax, label='Structural D recurrent-row fraction (%)')
    save(fig, 'figure_5_dependency_frontier_and_measured_points', sources,
         'Heatmap is structural accounting, not FLOPs, time, or a predicted SLO-safe region. Points are measured medium start-1 Decode-screen outcomes after N32 qualification.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = ['f_ouro_2d_c3_k3072_serial_primitive8'] + [f'c_ouro_2d_c3_k3072_medium_start{s}_128' for s in (1, 2, 3)]
    values, labels = [], []
    for name in names:
        folder = ROOT / 'raw' / name
        row = next(json.loads(line) for line in (folder/'native_prefill_graph_counts.jsonl').read_text().splitlines()
                   if json.loads(line)['role'] == 'prefill')
        values.append(row['host_costs_after_measurement_marker']['calls'])
        labels.append('Serial N8' if name.startswith('f_') else 'N128 start ' + name.split('start')[1][0])
    axes[0].bar(range(4), values, color='#4c78a8')
    axes[0].set_xticks(range(4), labels, rotation=20)
    axes[0].set_ylabel('Recorded successful P graph calls in measurement')
    axes[0].set_title('Ouro 2D: graph coverage changes between starts')
    axes[1].axis('off')
    axes[1].text(0, .95, 'Measured causal chain\n\n4-entry graph cache\n→ startup geometry occupies entries\n→ unseen primary geometry runs eager\n→ longer stream enclosures and variable TTFT\n\nThis does not prove structural 2D failure.\nA deterministic primary-geometry warm-up\nrequires a new GPU qualification window.', va='top', fontsize=11)
    save(fig, 'figure_6_graph_cache_root_cause', names,
         'Actual recorded successful graph calls. Different N8/N128 denominators are labeled. No kernel-time or hypothetical optimized speedup is inferred.')
    write(ROOT / 'figure_manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
