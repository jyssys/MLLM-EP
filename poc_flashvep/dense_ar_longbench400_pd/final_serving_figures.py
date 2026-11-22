"""Plot actual final starts and keep safety failures visible."""
import collections
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, write
from .confirmed_figures import mean_ci
from .figures import save
from .final_continuous_selection import policy


COLORS = dict(Agg='#d95f02', NativeStrong='#009e73', NativeTuned='#56b4e9', TokenCap2Control='#6a3d9a', BestLayer='#0072b2',
    BestToken='#cc79a7', BestHybrid='#e69f00', BestWholeDeflection='#333333')


def common_plot_name(cfg):
    compute,read,quiescent,cap,completion=policy(cfg)
    parts=[]
    if compute:parts.append('compute')
    if read:parts.append('READ')
    if quiescent:parts.append('quiescent')
    if cap is not None:parts.append('cap'+str(cap))
    if completion:parts.append('completion')
    return 'NativeCommon '+('/'.join(parts) if parts else 'unchanged')


def main_serving():
    plan = json.loads((ROOT / 'configs/final_continuous_execution_plan_v1.json').read_text())
    judgement = json.loads((ROOT / 'analysis/final_continuous_strong_confirmation_v1.json').read_text())
    status_by_name={r['name']:r['status'] for cell in judgement['cells']
        for start in cell['starts'] for r in start['arms'].values()}
    measured = {}
    invalid_or_pending=[]
    for cell in plan['cells']:
        if cell['workload'] != 'Main400':
            continue
        for arm, name in cell['jobs'].items():
            if status_by_name.get(name)!='MEASURED_ONLINE':
                invalid_or_pending.append(dict(name=name,arm=arm,QPS=cell['QPS'],
                    status=status_by_name.get(name,'NOT_RUN')))
                continue
            path = ROOT / 'analysis' / (name + '_metrics.json')
            receipt = ROOT / 'raw' / name / 'dispatcher_result.json'
            if not path.exists() or not receipt.exists() or not json.loads(receipt.read_text())['complete']:
                continue
            row = json.loads(path.read_text())
            plot_arm=common_plot_name(json.loads((ROOT/'configs'/(name+'.json')).read_text())) if arm.startswith('NativeCommon') else arm
            measured.setdefault((plot_arm, cell['QPS']), []).append(dict(name=name, metrics=row,
                start=cell['start'], seed=cell['seed'], GPU_pair=cell['GPU_pair']))
    fields = [('TTFT_p95_ms', 'TTFT p95 (ms)', 'ttft_p95'),
        ('ITL_p99_ms', 'Client-visible ITL p99 (ms)', 'itl_p99'),
        ('TPOT_p95_ms', 'Request TPOT p95 (ms/token)', 'tpot_p95'),
        ('generated_output_tokens_per_second', 'Generated output tokens/s', 'output_tokens'),
        ('SLO_qualified_output_tokens_per_second', 'SLO-qualified output tokens/s', 'slo_goodput'),
        ('Joint_SLO_attainment', 'Joint SLO attainment', 'joint_slo')]
    manifest = []
    arms = sorted({key[0] for key in measured}, key=lambda a: (a.startswith('NativeCommon'), a))
    for field, ylabel, stem in fields:
        fig, ax = plt.subplots(figsize=(9, 5.3))
        points = []
        for arm in arms:
            color = COLORS.get(arm, '#777777')
            xs, ys = [], []
            for qps in sorted(q for a, q in measured if a == arm):
                rows = measured[arm, qps]
                value, ci = mean_ci([r['metrics'][field] for r in rows])
                xs.append(qps)
                ys.append(value)
                if ci is not None:
                    ax.errorbar(qps, value, yerr=[[value - ci[0]], [ci[1] - value]],
                        fmt='none', color=color, capsize=3)
                ax.annotate('n' + str(len(rows)), (qps, value), xytext=(3, 3),
                    textcoords='offset points', fontsize=6.5, color=color)
                points.append(dict(arm=arm, QPS=qps, mean=value, CI95=ci, independent_starts=len(rows),
                    starts=[dict(name=r['name'], seed=r['seed'], GPU_pair=r['GPU_pair'],
                        output_tokens=r['metrics']['output_tokens']) for r in rows]))
            ax.plot(xs, ys, 'o--' if arm.startswith('NativeCommon') else 'o-', color=color,
                linewidth=1 if arm.startswith('NativeCommon') else 1.5, label=arm)
        if field == 'TTFT_p95_ms':
            ax.axhline(5000, color='gray', linestyle=':', label='TTFT SLO 5000ms')
        ax.set(xlabel='Offered QPS (requests/s)', ylabel=ylabel,
            title='LongBench-Balanced400-v1 | BI0 | 2 × H100 | Actual continuous batches')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=2)
        caption = ('MEASURED ONLINE | N400/start | Natural EOS, cap2048 | B8192/S1024.\n'
            'Three planned starts at QPS1.8/2.8. Endpoints n1. Labels give valid starts. CIs require at least3 starts.\n'
            'Frozen choices can differ by QPS. Lines connect measured points. Safety failures remain plotted and need the separate decode gate.\n'
            'Finite-cohort throughput is not sustainable capacity. Common Native curves use their exact D policy,not ordinal control indices.')
        save(fig, 'final_c3_main400_' + stem, caption, dict(evidence='MEASURED_ONLINE',
            metric=field, points=points, safety_analysis='final_continuous_strong_confirmation_v1.json'))
        manifest.append(dict(stem='final_c3_main400_' + stem, metric=field, points=points))
    tests = []
    for row in judgement['cells']:
        if row['required_starts'] < 3 or not row['comparisons']:
            continue
        for arm, refs in row['comparisons'].items():
            for reference in refs:
                if reference!='NativeStrong' and not reference.startswith('NativeCommon'):
                    continue
                if reference not in refs:
                    continue
                tests.append(dict(workload=row['workload'], QPS=row['QPS'], arm=arm,
                    reference=reference, result=refs[reference]))
    for common in (False,True):
        group=[r for r in tests if r['reference'].startswith('NativeCommon')==common]
        if not group:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, max(6.5,.25*len(group)+1.5)), sharey=True)
        labels = [f"{r['workload']} q{r['QPS']} {r['arm']}" for r in group]
        reference_label='exact common-policy Native' if common else 'NativeStrong'
        for ax, field, title in zip(axes, ('ITL_p99_ms', 'TPOT_p95_ms'),
                ('ITL p99 / '+reference_label, 'TPOT p95 / '+reference_label)):
            for i, row in enumerate(group):
                value = row['result']['ratios'][field]
                color = '#009e73' if value['CI95'][1] <= 1.05 else '#d55e00'
                mean, ci = value['estimate'], value['CI95']
                ax.errorbar(mean, i, xerr=[[mean - ci[0]], [ci[1] - mean]], fmt='o', color=color, capsize=3)
            ax.axvline(1, color='gray', linewidth=1)
            ax.axvline(1.05, color='red', linestyle='--', label='5% upper limit')
            ax.set(xlabel=title)
            ax.grid(alpha=.2)
        axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
        axes[0].invert_yaxis()
        axes[1].legend(fontsize=8)
        fig.suptitle('Paired independent-start decode protection | n3 per arm/cell')
        stem='final_c3_decode_noninferiority'+('_common_policy' if common else '')
        save(fig, stem,
            'MEASURED ONLINE | Green: metric CI upper ≤1.05. Orange: metric does not pass.\n'
            'Pointwise unadjusted paired bootstrap95% CI,27 resamples. Qualification requires BOTH metrics against NativeStrong AND exact common-policy Native.\n'
            'Absolute decode attainment>=95% per start remains separate. Each passing request needs every ITL<=34ms and TPOT<=33ms.',
            dict(evidence='MEASURED_ONLINE', tests=group))
        manifest.append(dict(stem=stem, tests=group))
    write(ROOT / 'analysis/final_serving_figure_manifest_v1.json', dict(figures=manifest,
        invalid_or_pending_not_plotted=invalid_or_pending,
        scope='Quality-invalid starts do not become normal performance curves. Decode-NI failures remain explicit in the separate gate figures.'))


def credit_and_chunks():
    ledger = json.loads((ROOT / 'analysis/decode_tail_recovery_discovery_all_v1.json').read_text())
    rows = [r for r in ledger['rows'] if r['name'].startswith('CreditTune_') and r['status'] == 'MEASURED_ONLINE']
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for axis, group in __import__('itertools').groupby(sorted(rows, key=lambda r: (r['axis'], r['Source_credit'])),
                key=lambda r: r['axis']):
            group = list(group)
            for ax, field in zip(axes, ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms')):
                ax.plot([r['Source_credit'] for r in group], [r['metrics'][field] for r in group], 'o-', label=axis)
                ax.set(xlabel='Source outstanding credit', ylabel=field.replace('_', ' '), xticks=[1, 4, 8, 16])
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Fixed within-axis hidden pool16 | Discovery200 | QPS2.8 | n1 per point')
        save(fig, 'credit_fixed_pool_sensitivity',
            'MEASURED ONLINE, DISCOVERY | Same within-axis source/pair/trace. Actual batches are recorded,not fixed by credit.\n'
            'Credit1 is a serialization control,not the serving default. Native/Token do not allocate an unused hidden pool.\n'
            'No independent-start CI. Hybrid4/f1/8 and pure cut8 have nominal equal D rows,with alignment residuals recorded.',
            dict(evidence='MEASURED_ONLINE', rows=rows))
    path = ROOT / 'analysis/continuation_chunk_residency_summary_v1.json'
    if path.exists():
        source = json.loads(path.read_text())
        rows = source['cases']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = [r['name'].split('q')[1].replace('_v1', '') for r in rows]
        axes[0].bar(range(len(rows)), [r['means']['prefill_model_calls'] for r in rows], label='Actual calls')
        axes[0].plot(range(len(rows)), [r['means']['ideal_chunk_lower_bound'] for r in rows], 'ko', label='B8192 lower bound')
        axes[0].set(ylabel='D prefill model calls/request')
        axes[1].bar(range(len(rows)), [r['means']['inter_prefill_host_gap_ms'] for r in rows], label='Host gap')
        axes[1].bar(range(len(rows)), [r['means']['inter_prefill_gap_overlap_other_decode_host_ms'] for r in rows],
            alpha=.7, label='Overlap with other decode steps')
        axes[1].set(ylabel='Host enclosure (ms/request)')
        for ax in axes:
            ax.set_xticks(range(len(rows)), labels, rotation=55, ha='right', fontsize=7)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=.2)
        fig.suptitle('B8192 counts query tokens,not token-layer rows')
        save(fig, 'continuation_chunk_residency_host_audit',
            'TRACE ANALYSIS | One same-trace discovery start per method/QPS. No GPU-exclusive time claim.\n'
            'These cut4 pure axes and Hybrid4/f1/8 are not iso-work. A longer host enclosure does not prove a causal TTFT penalty.\n'
            'Use the separate common admission A/B to test that hypothesis.', dict(evidence='TRACE_ANALYSIS', source=source))


def main():
    main_serving()
    credit_and_chunks()


if __name__ == '__main__':
    main()
