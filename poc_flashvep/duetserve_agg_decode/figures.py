"""Plot each output cap and evidence family separately."""
import json
from collections import defaultdict

from .common import ROOT
from .statistics import collect, mean_ci, paired_ratio


def label(arm):
    if arm.startswith('PD_G1'):
        return 'Native PD (1P1D)'
    return arm.replace('DP2_', 'Agg-DP2 / ').replace('TP2_', 'Agg-TP2 / ').replace('_FULL', ' / FULL').replace('_S', ' / S')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aggregate, _ = collect()
    plots = (
        ('mean_TTFT_ms', 'Mean TTFT (ms)', '01_qps_mean_ttft'),
        ('mean_TBT_ms', 'Mean TBT (ms)', '02_qps_mean_tbt'),
        ('ITL_p99_ms', 'ITL p99 (ms)', '03_qps_itl_p99'),
        ('TPOT_p95_ms', 'Request TPOT p95 (ms)', '04_qps_tpot_p95'),
        ('output_tokens_s', 'Generated output tokens/s', '06_qps_raw_generation'),
    )
    grouped = defaultdict(list)
    for value in aggregate:
        if value['family'] in ('DISCOVERY', 'FRESH_CONFIRMATION', 'SECONDARY', 'TP2_DIAGNOSTIC'):
            grouped[(value['family'], value['output_cap'])].append(value)
    for (family, cap), rows in grouped.items():
        for metric, ylabel, stem in plots:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for arm in sorted({r['arm'] for r in rows}):
                selected = sorted((r for r in rows if r['arm'] == arm), key=lambda r: r['offered_QPS'])
                x = [r['offered_QPS'] for r in selected]
                y = [r['metrics'][metric]['mean'] for r in selected]
                ax.plot(x, y, marker='o', label=label(arm))
                if all(r['metrics'][metric]['CI95'] for r in selected):
                    lower = [r['metrics'][metric]['CI95'][0] for r in selected]
                    upper = [r['metrics'][metric]['CI95'][1] for r in selected]
                    ax.fill_between(x, lower, upper, alpha=.12)
            n = sorted({r['metrics'][metric]['n_starts'] for r in rows})
            ax.set(xlabel='Offered QPS (requests/s)', ylabel=ylabel,
                title=f'Qwen3-8B / Input8000 / Output{cap} / {family}')
            note = ('Strict TP1 output parity failed. Within-TP2 diagnostic only.' if family == 'TP2_DIAGNOSTIC' else
                    f'Independent starts per point: {n}. Finite-cohort drain rate is not sustainable capacity.')
            ax.text(.01, -.23, note,
                    transform=ax.transAxes, fontsize=8)
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            path = ROOT / 'figures' / f'{family.lower()}_O{cap}_{stem}'
            fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
            fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
            plt.close(fig)
    # SLO values come from request-level conjunctions, not percentile gates.
    for family, cap in grouped:
        if family == 'TP2_DIAGNOSTIC':
            continue  # No strict TP1-parity or cross-method SLO claim.
        for regime in ('TTFT_priority', 'Balanced', 'Decode_priority'):
            by_arm = defaultdict(list)
            for path in sorted((ROOT / 'raw').glob('*/metrics.json')):
                value = json.loads(path.read_text())
                cfg = json.loads((ROOT / 'configs' / (path.parent.name + '.json')).read_text())
                prefix = 'confirm_' if family == 'FRESH_CONFIRMATION' else 'secondary_' if family == 'SECONDARY' else ('sweep_', 'concurrency_')
                if not path.parent.name.startswith(prefix) or cfg.get('correctness_only') or cfg['cap'] != cap:
                    continue
                if family == 'FRESH_CONFIRMATION' and (ROOT / 'state/user_primary_scope_revision_v2.json').exists() \
                        and cfg['arm'] == 'DP2' and cfg['token_budget'] == 512:
                    continue
                if not value.get('output_gate', {}).get('passed'):
                    continue
                arm = cfg['arm'] + f"_B{cfg['token_budget']}_S{cfg['D_slots']}_{cfg['ar_cudagraph_mode']}"
                by_arm[arm].append(value)
            for metric, ylabel, stem in (
                ('Joint_attainment', 'Joint SLO attainment (%)', '05_qps_joint_SLO'),
                ('Output_token_SLO_goodput', 'SLO-qualified output tokens/s', '07_qps_qualified_generation'),
            ):
                fig, ax = plt.subplots(figsize=(8, 4.8))
                for arm, values in sorted(by_arm.items()):
                    points = defaultdict(list)
                    for value in values:
                        points[value['offered_QPS']].append(value['SLO_regimes'][regime][metric])
                    x = sorted(points)
                    scale = 100 if metric == 'Joint_attainment' else 1
                    y = [scale * sum(points[q]) / len(points[q]) for q in x]
                    ax.plot(x, y, marker='o', label=label(arm))
                    summaries = [mean_ci(points[q]) for q in x]
                    if all(v['CI95'] for v in summaries):
                        ax.fill_between(x, [scale * v['CI95'][0] for v in summaries],
                            [scale * v['CI95'][1] for v in summaries], alpha=.12)
                ax.set(xlabel='Offered QPS (requests/s)', ylabel=ylabel,
                       title=f'Input8000 / Output{cap} / {regime} / {family}')
                ax.grid(alpha=.25)
                ax.legend(fontsize=7)
                if family == 'TP2_DIAGNOSTIC':
                    ax.text(.01, -.23, 'Diagnostic only. Strict TP1 output parity failed. Do not use for cross-method superiority.',
                            transform=ax.transAxes, fontsize=8)
                fig.tight_layout()
                target = ROOT / 'figures' / f'{family.lower()}_O{cap}_{regime}_{stem}'
                fig.savefig(target.with_suffix('.png'), dpi=180, bbox_inches='tight')
                fig.savefig(target.with_suffix('.pdf'), bbox_inches='tight')
                plt.close(fig)
    for (family, cap), rows in grouped.items():
        for metric in ('mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms'):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for arm in sorted({r['arm'] for r in rows}):
                selected = sorted((r for r in rows if r['arm'] == arm), key=lambda r: r['offered_QPS'])
                if len(selected) < 2:
                    continue
                def by_seed(point):
                    return {seed: json.loads((ROOT / 'raw' / name / 'metrics.json').read_text())[metric]
                            for seed, name in zip(point['seeds'], point['starts'])}
                low = by_seed(selected[0])
                values = []
                for point in selected:
                    high = by_seed(point)
                    seeds = sorted(set(low) & set(high))
                    values.append(paired_ratio([high[s] for s in seeds], [low[s] for s in seeds]))
                x = [point['offered_QPS'] for point in selected]
                ax.plot(x, [value['ratio'] for value in values], 'o-', label=label(arm))
                if all(value['CI95'] for value in values):
                    ax.fill_between(x, [v['CI95'][0] for v in values], [v['CI95'][1] for v in values], alpha=.12)
            ax.axhline(2, color='black', linestyle='--', linewidth=1, label='2x load degradation hypothesis')
            ax.set(xlabel='Offered QPS (requests/s)', ylabel=f'{metric} / same-arm low-QPS value',
                   title=f'Input8000 / Output{cap} / {family} / paired starts')
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            target = ROOT / 'figures' / f'{family.lower()}_O{cap}_load_ratio_{metric}'
            fig.savefig(target.with_suffix('.png'), dpi=180, bbox_inches='tight')
            fig.savefig(target.with_suffix('.pdf'), bbox_inches='tight')
            plt.close(fig)
    budget_sensitivity(plt, aggregate)
    measured_timeline(plt)
    from .cupti28_figures import render as api28_figures
    api28_figures(plt)


def budget_sensitivity(plt, aggregate):
    rows = [r for r in aggregate if r['family'] == 'BUDGET_CONTROL']
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, ylabel in zip(axes.flat,
        ('mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'TTFT_p95_ms'),
        ('Mean TBT (ms)', 'ITL p99 (ms)', 'TPOT p95 (ms)', 'TTFT p95 (ms)')):
        for qps in sorted({r['offered_QPS'] for r in rows}):
            selected = sorted((r for r in rows if r['offered_QPS'] == qps),
                              key=lambda r: int(r['arm'].split('_B')[1].split('_')[0]))
            x = [int(r['arm'].split('_B')[1].split('_')[0]) for r in selected]
            ax.plot(x, [r['metrics'][metric]['mean'] for r in selected], 'o-', label=f'QPS{qps}')
        ax.set(xscale='log', xlabel='Per-replica token budget (tokens)', ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Controlled budget sweep / S128 / same graph capture set / single start per point')
    fig.tight_layout()
    for ext in ('.png', '.pdf'):
        fig.savefig(ROOT / 'figures' / ('08_controlled_budget_sensitivity' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def measured_timeline(plt):
    candidates = sorted((ROOT / 'analysis').glob('*NSYS_v*_kernel_union.json'))
    for path in candidates:
        result = json.loads(path.read_text())
        if result.get('role', 'replica0') != 'replica0':
            continue  # Do not draw GPU1 token emissions as GPU0 execution.
        name = result['name']
        raw = json.loads((ROOT / 'analysis' / (name + '_kernel_intervals.json')).read_text())
        trace = json.loads((ROOT / 'raw' / name / 'iteration_trace_replica0.json').read_text())
        offset = result['trace_to_host_offset_seconds']
        lower, upper = result['trace_window_ns']
        begin, end = lower / 1e9 + offset, upper / 1e9 + offset
        anchor = trace['GPU_event_clock_anchor']
        if not anchor:
            continue
        host_anchor = (anchor['host_lower'] + anchor['host_upper']) / 2
        forwards = [item for item in trace['execution'] if item['kind'] == 'MODEL_FORWARD'
                    and 'GPU_event_start_relative_anchor_ms' in item]
        def event_times(item):
            return [host_anchor + item[key] / 1000 for key in
                    ('GPU_event_start_relative_anchor_ms', 'GPU_event_end_relative_anchor_ms')]
        overlap = [item for item in forwards if event_times(item)[1] > begin and event_times(item)[0] < end]
        selected = next((item for item in overlap if any(item['prefilling'])), overlap[0])
        origin = max(begin, event_times(selected)[0] - .08)
        finish = min(end, origin + .7)
        fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
        for item in trace['scheduler']:
            if origin <= item['host_end'] <= finish:
                mixed = any(r['prefill_tokens'] for r in item['requests'])
                axes[0].plot((item['host_end'] - origin) * 1000, int(mixed), '|',
                             color='#d55e00' if mixed else '#0072b2')
        axes[0].set(yticks=[0, 1], yticklabels=['Decode', 'With Prefill'], ylabel='Host schedule')
        for item in overlap:
            a, b = event_times(item)
            if b <= origin or a >= finish:
                continue
            mixed = any(item['prefilling'])
            axes[1].broken_barh([((a - origin) * 1000, (b - a) * 1000)], (int(mixed) - .3, .6),
                               facecolors='#d55e00' if mixed else '#0072b2')
        axes[1].set(yticks=[0, 1], yticklabels=['Decode', 'With Prefill'], ylabel='CUDA event\nenclosure')
        streams = sorted({r['streamId'] for r in raw['kernels'] + raw['copies']})
        for row in raw['kernels']:
            a, b = row['start'] / 1e9 + offset, row['end'] / 1e9 + offset
            if b > origin and a < finish:
                axes[2].broken_barh([((a - origin) * 1000, (b - a) * 1000)],
                    (streams.index(row['streamId']) - .3, .6), facecolors='#009e73')
        for row in raw['copies']:
            a, b = row['start'] / 1e9 + offset, row['end'] / 1e9 + offset
            if b > origin and a < finish:
                axes[2].broken_barh([((a - origin) * 1000, (b - a) * 1000)],
                    (streams.index(row['streamId']) - .3, .6), facecolors='#cc79a7')
        axes[2].set(yticks=list(range(len(streams))), yticklabels=[str(s) for s in streams],
                    ylabel='Kernel / copy\nCUDA stream ID')
        from .analyze import records
        requests = records(ROOT / 'raw' / name / 'requests.jsonl')
        count = 0
        for row in requests:
            if row['sample_id'] % 2:
                continue
            times = [v['worker_time'] for v in row['token_events'] if origin <= v['worker_time'] <= finish]
            if times and count < 4:
                axes[3].plot([(t - origin) * 1000 for t in times], [count] * len(times), '.-', markersize=3,
                             label=f"D_req{row['sample_id']}")
                count += 1
        axes[3].set(xlabel='Time in the same GPU0 run (ms)', ylabel='Worker token\nemission')
        axes[3].legend(fontsize=7, loc='upper right')
        for ax in axes:
            ax.set_xlim(0, (finish - origin) * 1000)
            ax.grid(alpha=.2)
        qualified = (result['clean_physical_kernel_measurement'] == 'QUALIFIED_SHORT_WINDOW')
        status = 'Qualified short capture' if qualified else 'Diagnostic capture / clean-capacity gate NOT PASSED'
        fig.suptitle(f'GPU0 only / {status}\nGreen: actual kernels. Purple: copies. Event enclosures are not kernel time.', fontsize=10)
        fig.tight_layout()
        for ext in ('.png', '.pdf'):
            fig.savefig(ROOT / 'figures' / (name + '_09_actual_timeline' + ext), dpi=180, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
