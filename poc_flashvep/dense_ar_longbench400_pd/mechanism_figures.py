"""Separate structural ownership,qualified GPU costs and offline predictions."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .common import ROOT
from .geometry import Partition
from .figures import save


def dag():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    colors = ['#4c78a8', '#f58518', '#e45756']
    for rect, color in zip([(0, 0, 7168, 32), (0, 32, 7168, 4), (7168, 0, 1024, 36)], colors):
        ax.add_patch(Rectangle(rect[:2], *rect[2:], facecolor=color, alpha=.8, edgecolor='white'))
    for x, y, label in [(3584, 16, 'P: prefix lower32'), (3584, 34, 'D: prefix upper4'), (7680, 18, 'D suffix\nfull36')]:
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    ax.set(xlim=(0, 8192), ylim=(0, 36), xlabel='Prompt position', ylabel='Transformer layer',
           title='Genuine interior partition: L8192, m4, s1024')
    ax.set_xticks([0, 7168, 8192])
    ax.set_yticks([0, 32, 36])
    ax = axes[1]
    positions = {'P lower prefix': (.12, .5), 'D upper prefix': (.47, .8),
                 'D lower suffix': (.47, .2), 'D upper suffix': (.85, .5)}
    for label, (x, y) in positions.items():
        ax.text(x, y, label.replace(' ', '\n', 1), ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=.6', fc='#edf2f7', ec='gray'))
    for a, b in [('P lower prefix', 'D upper prefix'), ('P lower prefix', 'D lower suffix'),
                 ('D upper prefix', 'D upper suffix'), ('D lower suffix', 'D upper suffix')]:
        ax.annotate('', xy=positions[b], xytext=positions[a], arrowprops=dict(arrowstyle='->', color='black', shrinkA=43, shrinkB=43))
    ax.set(xlim=(0, 1), ylim=(0, 1), title='Exact causal dependencies')
    ax.axis('off')
    p = Partition(8192, 4, 1024)
    save(fig, 'hybrid400_structural_DAG',
         'STRUCTURAL ACCOUNTING | Illustrative geometry, not a timing result or an independent GPU start.\n'
         'P rows229376 + D rows65536 = 36 ×8192. D rows equal Layer8 at this length. Attention and handoff are not equal.\n'
         'Aligned order: prefix upper → suffix full36. Qualified short fused order: suffix lower → all upper. Decode remains full36.',
         dict(evidence='STRUCTURAL_ACCOUNTING', partition=p.record(), no_timing_claim=True))


def attention():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    cuts = (2, 4, 8, 12)
    length = 8192
    x = np.arange(4)
    records = []
    for axis, color, offset in [('Layer', '#4c78a8', -.18), ('Token', '#f58518', .18)]:
        values = []
        for m in cuts:
            # Fractional suffix defines the exact structural iso-row expectation only.
            suffix = length * m / 36
            prefix = length - suffix
            total = 36 * length * (length + 1) / 2
            p_pairs = (36 - m) * length * (length + 1) / 2 if axis == 'Layer' else 36 * prefix * (prefix + 1) / 2
            values.append((total - p_pairs) / 1e9)
            records.append(dict(axis=axis, cut=m, D_pairs=total - p_pairs,
                P_pairs=p_pairs, fractional_token_suffix=suffix, D_GEMM_rows=length * m))
        axes[0].bar(x + offset, values, .36, label=axis, color=color)
    axes[0].set(xticks=x, xticklabels=['Last2', 'Last4', 'Last8', 'Last12'],
                ylabel='D causal attention pairs (billions)', title='Iso-row structural work distribution')
    ratios = [(2 - m / 36) - (1 - m / 36) / (length + 1) for m in cuts]
    axes[1].bar(x, ratios, color='#72b7b2')
    axes[1].set(xticks=x, xticklabels=['Last2', 'Last4', 'Last8', 'Last12'],
                ylabel='Token / Layer D attention pairs', title='Attention is not matched by GEMM rows', ylim=(0, 2.1))
    for i, value in enumerate(ratios):
        axes[1].text(i, value + .03, f'{value:.3f}×', ha='center')
    axes[0].legend()
    save(fig, 'hybrid400_structural_attention_distribution',
         'STRUCTURAL ACCOUNTING | L8192,36 layers. Fractional suffix is an analytical iso-row expectation,not an executed token count.\n'
         'Actual integer/aligned requests have separate aggregate-work manifests. Pair ratios are not measured GPU-time ratios.',
         dict(evidence='STRUCTURAL_ACCOUNTING', records=records, actual_GPU_time='NOT_MEASURED_BY_THIS_FIGURE'))


def gpu_cost():
    data = json.loads((ROOT / 'analysis/operator_repeat_observer_analysis_v3.json').read_text())
    selected = [c for c in data['cells'] if c['input_bucket'] == 3 and c['cut'] == 4]
    assert len(selected) == 2 and all(c['observer_gate_passed'] for c in selected)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    table = []
    for ax, role in zip(axes, ('prefill', 'decode')):
        for index, axis in enumerate(('Layer', 'Token')):
            cell = next(c for c in selected if c['axis'] == axis)
            values = []
            for t in cell['triplets']:
                r = t['roles'][role]
                ops = r['prompt_operator_union_ms_per_request']
                values.append([r['prompt_GPU_kernel_union_ms_per_request'], ops['ATTENTION'],
                    sum(ops.get(k, 0) for k in ('GEMM_QKV', 'GEMM_O_PROJ', 'GEMM_GATE_UP', 'GEMM_DOWN'))])
            means = np.mean(values, axis=0)
            ax.bar(np.arange(3) + (index - .5) * .36, means, .36, label=axis,
                   color=('#4c78a8', '#f58518')[index])
            table.append(dict(role=role, axis=axis, per_start_values=values, means_ms_per_request=means.tolist()))
        ax.set(xticks=range(3), xticklabels=['All kernels', 'Attention', 'Projection GEMM'],
               ylabel='Prompt kernel union (ms/request)', title='P prefix' if role == 'prefill' else 'D continuation')
        ax.legend()
    save(fig, 'hybrid400_qualified_long_operator_cost',
         'ISOLATED KERNEL MEASUREMENT | Original20,006–31,133-token prompts | Last4 / matched Token | C0 batch1/1/1 | Fixed output8.\n'
         'Three independent warm-shape clean/profile/clean triplets per axis. Both-role observer gates passed unchanged5% limits.\n'
         'Grouped metrics are not additive. Projection GEMM excludes LM head. Not natural online operator shares. SM/HBM NOT_MEASURED.',
         dict(evidence='ISOLATED_KERNEL_MEASUREMENT', cells=table,
              source='operator_repeat_observer_analysis_v3.json'))


def replay():
    source = json.loads((ROOT / 'analysis/c0_tail_replay_heldout_validation_v1.json').read_text())
    fields = ['TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms']
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for j, label in enumerate(('median', 'p90')):
        values = [source['checks'][k][label] for k in fields]
        ax.bar(np.arange(3) + (j - .5) * .36, values, .36, label=label)
    ax.set(xticks=range(3), xticklabels=['TTFT p95', 'ITL p99', 'TPOT p95'],
           ylabel='Absolute prediction error (%)', title='C0 state-updating replay: held-out validation')
    ax.legend()
    save(fig, 'hybrid400_offline_replay_validation',
         'OFFLINE REPLAY | 880 calibration templates | 432 unique held-out C0 server traces | Sixteen simulation draws/trace.\n'
         'Median and p90 are error statistics,not independent-start CIs. The extension is posthoc with frozen error limits.\n'
         'Joint-SLO error median/p90:0/0 percentage points. This does not qualify C3 or active-decode online insertion.',
         dict(evidence='OFFLINE_REPLAY', checks=source['checks'], SLO_gate=source['SLO_replay_gate_passed']))
    result = json.loads((ROOT / 'analysis/c0_bounded_axis_hybrid_oracle_v1.json').read_text())
    assert result['complete']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for output, marker in ((16, 'o'), (64, 's')):
        for ax, field, ylabel in [(axes[0], 'goodput_over_discovery_best_fixed', 'Predicted SLO-goodput ratio'),
                                 (axes[1], 'TTFT_over_discovery_best_fixed', 'Predicted TTFT p95 ratio')]:
            groups = [[r[field] for r in result['rows'] if r['input_bucket'] == b and r['output'] == output] for b in range(4)]
            means = [np.mean(g) for g in groups]
            ax.errorbar(range(4), means, yerr=[[m - min(g) for m, g in zip(means, groups)],
                [max(g) - m for m, g in zip(means, groups)]], marker=marker, label=f'Controlled output{output}', capsize=4)
            ax.axhline(1, ls='--', color='gray', lw=1)
            ax.set(xticks=range(4), xticklabels=['3K–6K', '6K–10K', '10K–20K', '20K–36K'],
                   xlabel='Original input bucket', ylabel=ylabel)
            ax.legend()
    save(fig, 'hybrid400_bounded_offline_oracle',
         'OFFLINE ORACLE | C0 whole-request D service | Ratios to model-selected discovery-fixed action,including interior Hybrid.\n'
         'Three future-arrival traces per cell. Bars show trace min/max,not a measured server-start CI. Calibration uncertainty is not propagated.\n'
         'Two coordinate sweeps are not a global optimum or upper bound. This is not a comparison with the strongest GPU-measured static.\n'
         'No measured online dynamic speedup or active-decode safety claim. Pure-Token action grids differ from the separate static GPU grid.',
         dict(evidence='OFFLINE_ORACLE', cells=result['rows'], no_online_speedup_claim=True))


if __name__ == '__main__':
    dag()
    attention()
    gpu_cost()
    replay()
    print('Created five mechanism and offline figures.', flush=True)
