"""Evidence-only reports and scoped publication. Never upgrades missing evidence to GO."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import ROOT, REPO, BASE, START, HARD_STOP, MODEL_PINS, PYTHON, write, setup
from .analyze import main as analyze, save, wilson
from .jobs import parity


def read(name):
    path = ROOT / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def table(headers, rows):
    def cell(value):
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows]]) + '\n'


def number(row, key, default=0.):
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def capacity_bounds(rows):
    result = []
    for model in ('ouro', 'huginn'):
        for count in (2, 4, 8):
            for arm in ('DP', 'DP_Mixed', 'PD', 'PD_G1', 'GenericLayer', 'B'):
                group = [row for row in rows if row['model'] == model and int(row['GPU_count']) == count and row['arm'] == arm]
                rates = sorted({number(row, 'offered_rate') for row in group})
                passed = []
                failed = []
                for rate in rates:
                    at_rate = [row for row in group if number(row, 'offered_rate') == rate]
                    ok = all(row['status'] == 'COMPLETE' and number(row, 'joint_attainment_percent') >= 90
                             and number(row, 'unregistered_GPU_process_samples') == 0
                             and row.get('weak_queue_check') == 'True'
                             and number(row, 'completed_throughput_with_drain') >= .9 * number(row, 'actual_trace_rate', float('inf'))
                             for row in at_rate)
                    (passed if ok else failed).append(rate)
                low = max(passed) if passed else None
                above = [rate for rate in failed if low is not None and rate > low]
                high = min(above) if above else None
                starts = len({row['seed'] for row in group if low is not None and number(row, 'offered_rate') == low})
                gate_failed = arm == 'B' and (ROOT / 'state' / (model + '_B_serving_parity_failure.json')).exists()
                result.append(dict(model=model, GPU_count=count, arm=arm, attempted_rates=rates,
                                   lower_tested_rate=low if low is not None else 'NOT_ESTABLISHED',
                                   accepted_lossless_capacity='REJECTED_CORRECTNESS_GATE' if gate_failed else 'NOT_ESTABLISHED_ORIGINAL_FIDELITY',
                                   upper_rejected_rate=high if high is not None else 'NOT_BRACKETED',
                                   starts_at_lower_rate=starts,
                                   evidence='BOUNDED_CAPACITY_ONLY' if group else 'NOT_MEASURED',
                                   uncertainty='Finite trace/rate grid. No selection-adjusted capacity interval. Queue heuristic is not a stationarity proof.'))
    save('capacity_bounds.csv', result)
    return result


def paired_rates(rows):
    grouped = {}
    for row in rows:
        if row['status'] != 'COMPLETE':
            continue
        key = (row['model'], row['GPU_count'], row['offered_rate'], row['seed'], row['planned_arrivals'],
               row.get('physical_GPUs'), row.get('dataset'))
        grouped.setdefault(key, {})[row['arm']] = row
    groups = {}
    fidelity = read('paired_output_fidelity.csv')
    for key, values in grouped.items():
        if 'B' not in values:
            continue
        for baseline in ('PD_G1', 'DP_Mixed', 'DP', 'PD'):
            if baseline not in values:
                continue
            left, right = values[baseline], values['B']
            if left.get('physical_GPUs') != right.get('physical_GPUs'):
                continue
            match = next((x for x in fidelity if x['left'] == left['run'] and x['right'] == right['run']), None)
            exact = match is not None and int(match['exact']) == int(right['planned_arrivals'])
            groups.setdefault((*key[:3], baseline), []).append((left, right, exact))
    result = []
    for key, pairs in sorted(groups.items()):
        before = np.array([number(a, 'qualified_requests') for a, b, exact in pairs])
        after = np.array([number(b, 'qualified_requests') for a, b, exact in pairs])
        totals = np.array([number(a, 'planned_arrivals') for a, b, exact in pairs])
        ratio = float(after.sum() / before.sum()) if before.sum() > 0 else 'UNDEFINED_ZERO_BASELINE'
        ci = 'NOT_MEASURED_SINGLE_START'
        delta_ci = 'NOT_MEASURED_SINGLE_START'
        if len(pairs) >= 2:
            indices = np.random.default_rng(20261029).integers(0, len(pairs), (3000, len(pairs)))
            left = before[indices].sum(axis=1)
            right = after[indices].sum(axis=1)
            values = np.divide(right, left, out=np.full_like(right, np.nan), where=left > 0)
            ci = np.percentile(values, [2.5, 97.5]).tolist() if np.isfinite(values).all() else 'UNBOUNDED_ZERO_BASELINE_DRAWS'
            delta_ci = np.percentile(100 * (right - left) / totals[indices].sum(axis=1), [2.5, 97.5]).tolist()
        result.append(dict(model=key[0], GPU_count=key[1], offered_rate=key[2], baseline=key[3], candidate='B',
                           independent_starts=len(pairs), baseline_qualified=int(before.sum()), candidate_qualified=int(after.sum()),
                           total_arrivals=int(totals.sum()), qualified_goodput_ratio=ratio,
                           qualified_goodput_ratio_ci95=ci, attainment_difference_ci95_pp=delta_ci,
                           all_outputs_exact=all(exact for a, b, exact in pairs), accepted_lossless_speedup=False,
                           raw_pairs=[[a['run'], b['run']] for a, b, exact in pairs],
                           evidence='MEASURED_PAIRED_FINITE_TRACE_START_CLUSTER_BOOTSTRAP',
                           limitation='Resample whole independent starts. Very few starts give exploratory intervals. This is not a sustainable-capacity ratio CI.'))
    save('paired_rate_comparisons.csv', result)
    return result


def latency_uncertainty(rows):
    grouped = {}
    for row in rows:
        if row['status'] != 'COMPLETE' or row['workload'] != 'W1' or row.get('diagnostic') == 'True':
            continue
        key = (row['model'], row['GPU_count'], row['arm'], row['offered_rate'], row.get('dataset'),
               row.get('execution_epoch', 'original'))
        grouped.setdefault(key, []).append(row)
    results = []
    for key, starts in sorted(grouped.items()):
        for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'observed_ITL_ms'):
            for q in (50, 95, 99):
                column = metric + f'_p{q}'
                values = [number(row, column, float('nan')) for row in starts]
                values = np.asarray([value for value in values if np.isfinite(value)])
                ci = 'NOT_MEASURED_SINGLE_START'
                if len(values) >= 2:
                    indices = np.random.default_rng(20261030).integers(0, len(values), (3000, len(values)))
                    ci = np.percentile(values[indices].mean(axis=1), [2.5, 97.5]).tolist()
                results.append(dict(model=key[0], GPU_count=key[1], arm=key[2], offered_rate=key[3],
                                    dataset=key[4], execution_epoch=key[5], metric=column, independent_starts=len(values),
                                    mean_of_start_percentiles_ms=float(values.mean()) if len(values) else 'NOT_MEASURED',
                                    ci95_ms=ci, evidence='MEASURED_START_CLUSTER_BOOTSTRAP',
                                    statistic='Mean of per-start percentiles. Not a pooled-request percentile interval.',
                                    limitation='Exploratory with few starts. Not selection-adjusted.'))
    save('latency_run_cluster_ci.csv', results)


def runtime_warnings():
    records = []
    for path in sorted((ROOT / 'logs').glob('*.log')):
        lines = path.read_text(errors='replace').splitlines()
        completed = [i for i, line in enumerate(lines) if line.startswith('DONE ')]
        for i, line in enumerate(lines):
            if any(term in line for term in (' ERROR ', 'Traceback (most recent call last)', 'OutOfMemoryError', 'CUDA out of memory')):
                records.append(dict(log=str(path.relative_to(ROOT)), line=i + 1, message=line,
                                    after_last_DONE=bool(completed) and i > completed[-1],
                                    classification='LOG_WARNING_NOT_AUTOMATICALLY_A_FAILED_RUN'))
    save('runtime_warning_inventory.csv', records)
    return records


def figures(rows, latency):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    arms = ('DP', 'DP_Mixed', 'PD', 'PD_G1', 'B')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, model in zip(axes, ('ouro', 'huginn')):
        for arm in arms:
            group = [r for r in rows if r['model'] == model and r['GPU_count'] == '2' and r['arm'] == arm and r['status'] == 'COMPLETE']
            rates = sorted({number(r, 'offered_rate') for r in group})
            if rates:
                invalid_B = arm == 'B' and (ROOT / 'state' / (model + '_B_serving_parity_failure.json')).exists()
                ax.plot(rates, [np.mean([number(r, 'joint_attainment_percent') for r in group if number(r, 'offered_rate') == rate]) for rate in rates],
                        'x:' if invalid_B else 'o-', label=arm + ' [OUTPUT-CONFOUNDED]' if invalid_B else arm)
        ax.axhline(90, color='black', linestyle='--')
        ax.set_title(model + ': W1, two GPUs')
        ax.set_xlabel('Nominal offered rate (requests/s)')
        ax.set_ylabel('Mean per-run TTFT+TPOT attainment (%)')
        ax.set_ylim(0, 102)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle('Actual finite open-loop runs; not proven sustainable capacity. Raw N and starts remain in CSV.')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/capacity_vs_arrival_rate.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, model in enumerate(('ouro', 'huginn')):
        for line, metric in enumerate(('ttft_ms', 'tpot_ms')):
            ax = axes[line, column]
            for arm in arms:
                group = [r for r in latency if r['model'] == model and r['workload'] == 'W1' and r['GPU_count'] == '2' and r['arm'] == arm and r['status'] == 'COMPLETE']
                rates = sorted({number(r, 'offered_rate') for r in group})
                for percentile, style in ((50, '-'), (99, '--')):
                    if rates:
                        invalid_B = arm == 'B' and (ROOT / 'state' / (model + '_B_serving_parity_failure.json')).exists()
                        label = arm + (' [OUTPUT-CONFOUNDED]' if invalid_B else '') + ' p' + str(percentile)
                        ax.plot(rates, [np.mean([number(r, metric + f'_p{percentile}') for r in group if number(r, 'offered_rate') == rate]) for rate in rates], style, label=label)
            ax.set_title(model + ' ' + metric)
            ax.set_xlabel('Offered requests/s')
            ax.set_ylabel('Mean per-run percentile (ms)')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=6, loc='upper center', ncol=5)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(ROOT / 'figures/TTFT_TPOT_p50_p99_vs_rate.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = 0
    for path in sorted((ROOT / 'raw').glob('*diagnostic*/transfer_events_*.jsonl')):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        completed = [r for r in events if r['kind'] == 'completion_observed'][:8]
        if not completed:
            continue
        zero = min(r['time'] for r in events)
        for event in completed:
            ax.plot([1000 * (event['actual_submit'] - zero), 1000 * (event['time'] - zero)], [plotted, plotted], linewidth=2)
            plotted += 1
        if plotted >= 32:
            break
    if not plotted:
        ax.text(.5, .5, 'NOT_MEASURED: no diagnostic transfer timeline', ha='center', transform=ax.transAxes)
    ax.set_xlabel('Milliseconds since first recorded host event in each process')
    ax.set_ylabel('Diagnostic transfer segment')
    ax.set_title('Host submission-to-observed-completion intervals; not DMA time or additive E2E cost')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/frontier_timeline.svg')
    plt.close(fig)


def report(final=False):
    setup()
    if final:
        write(ROOT / 'state/stop_resource_observer.json', dict(time=time.time(), reason='GPU measurements ended before final reporting.'))
    analyze()
    from .validate import main as validate
    evidence_issues = validate()
    from ..looped_pd_abc import profile_analysis
    profile_analysis.ROOT = ROOT
    profile_analysis.memory_and_transport()
    profile_analysis.nonblocking_evidence()
    profile_rows = profile_analysis.analyze_profiles()
    for row in profile_rows:
        try:
            name = Path(row['archive']).parts[1]
            contract = json.loads((ROOT / 'raw' / name / 'serving_contract.json').read_text())
            visible = int(row['device'])
            row.update(run=name, model=contract['model'], arm=contract['arm'],
                       visible_device=visible, physical_GPU=contract['gpus'][visible])
        except (KeyError, ValueError, IndexError, FileNotFoundError):
            row['physical_GPU'] = 'NOT_MEASURED'
    save('profile_device_summary.csv', profile_rows or [dict(evidence='NOT_MEASURED', reason='No parsed native profile artifact.')])
    from .diagnostic_analysis import main as diagnostic_analysis
    diagnostic_rows = diagnostic_analysis()
    rows = read('03_primary_w1_capacity.csv')
    latency = read('04_latency_percentiles.csv')
    bound = capacity_bounds(rows)
    pair_rows = paired_rates(rows)
    latency_uncertainty(latency)
    warning_rows = runtime_warnings()
    decisions = []
    for model in ('ouro', 'huginn'):
        for count in (2, 4, 8):
            measured = [row for row in rows if row['model'] == model and row['GPU_count'] == str(count) and row['arm'] == 'B']
            parity_rows = [row for row in read('paired_output_fidelity.csv') if row['model'] == model and row['GPU_count'] == str(count) and row['baseline'] == 'PD_G1']
            bad = [row for row in parity_rows if int(row['N']) != int(row['planned_arrivals']) or int(row['exact']) != int(row['N'])]
            global_failure = (ROOT / 'state' / (model + '_B_serving_parity_failure.json')).exists()
            if global_failure and not measured:
                status = 'NOT_MEASURED_TIME_BUDGET' if final else 'PENDING'
                reason = 'The model-level correctness gate prevented B measurement at this GPU budget. This cell was not tested.'
            elif global_failure:
                status = 'NO_GO_CORRECTNESS'
                reason = ('The model-level B performance gate is closed after a matched serving mismatch. '
                          'No unmeasured GPU-budget cell is a measured correctness failure.')
            elif not measured:
                status = 'NOT_MEASURED_TIME_BUDGET' if final else 'PENDING'
                reason = 'No B primary serving run at this GPU budget.'
            elif bad:
                status = 'NO_GO_CORRECTNESS'
                reason = 'B and matched PD_G1 differ on natural outputs. Recorded latency is not accepted as equivalent-work speedup.'
            else:
                status = 'CONDITIONAL_FIDELITY'
                reason = 'Internal paired outputs require review. Original-reference fidelity, generic-streaming superiority, and an unbounded sustainable maximum remain unproved.'
            decisions.append(dict(model=model, GPU_count=count, method_verdict=status,
                                  system_verdict=status, novelty='LOOP_SPECIFIC_NOVELTY_NOT_ESTABLISHED',
                                  capacity_status='REJECTED_CORRECTNESS_GATE' if global_failure and measured else 'BOUNDED_CAPACITY_ONLY' if measured else 'NOT_MEASURED',
                                  measurement_status='MEASURED_OUTPUT_CONFOUNDED' if global_failure and measured else 'NOT_MEASURED_CORRECTNESS_GATE' if global_failure else 'MEASURED' if measured else 'NOT_MEASURED',
                                  reason=reason))
    save('model_GPU_decisions.csv', decisions)
    state = json.loads((ROOT / 'state/campaign.json').read_text()) if (ROOT / 'state/campaign.json').exists() else {}
    if final and state.get('stage') not in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED', 'TIME_BUDGET_STOPPED'):
        raise RuntimeError('A live GPU campaign cannot receive a final report')
    title = '# Final B SLO-serving decision\n\n' if final else '# B SLO-serving checkpoint — experiment in progress\n\n'
    text = title + ('Status: FINALIZED_WITH_LIMITATIONS.\n' if final else 'Status: IN_PROGRESS. No final decision is accepted.\n')
    text += f'Wall time since setup: {(time.time()-START)/3600:.3f} hours. Hard allowance: 15 hours.\n'
    text += f'Parent evidence: `{BASE}`. Current controller stage: `{state.get("stage", "UNKNOWN")}`.\n\n'
    text += 'Only Ouro R4 and Huginn R32 are active. Both use original BF16 weights and full fixed recurrence.\n'
    text += 'No A/C policy, dynamic model exit, training, approximation, or new distributed engine is used.\n\n'
    text += '## Original execution epoch: independent method and system decisions\n\n'
    text += table(['Model', 'GPUs', 'Method verdict', 'System verdict', 'Capacity scope'], [[r[k] for k in ('model', 'GPU_count', 'method_verdict', 'system_verdict', 'capacity_status')] for r in decisions])
    text += '\nNo original-model-lossless or loop-exclusive novelty GO is accepted.\n'
    text += 'A separately qualified repair epoch does not erase these original failures.\n'
    text += 'Read `b_repair_diagnosis.md` and `repair_epoch_report.md` for later diagnosis and qualified reruns, if available.\n'
    text += 'Missing or failed measurements do not become negative speedups or positive capacity evidence.\n\n'
    text += '## Baseline-only SLO lock\n\n'
    lock = ROOT / 'configs/slo_lock.json'
    slo = json.loads(lock.read_text()) if lock.exists() else {}
    text += table(['Model', 'TTFT limit ms', 'TPOT limit ms', 'Secondary ITL ms'], [[m, s['TTFT_ms'], s['TPOT_ms'], s['secondary_ITL_ms']] for m, s in slo.items()]) if slo else 'PENDING baseline-only calibration.\n'
    text += '\nThe lock uses disjoint calibration text and no held-out B result. It never changes after the first write.\n'
    text += 'The primary joint indicator uses request-average TPOT, not maximum ITL.\n'
    text += 'One-token successful responses have TPOT0. Unknown intra-callback timestamps remain unknown.\n\n'
    text += '## W1 rate bounds, not a proven infinite-horizon maximum\n\n'
    text += table(['Model', 'GPUs', 'Arm', 'Raw SLO-passing rate', 'First higher rejected rate', 'Starts', 'Lossless capacity acceptance'], [[r[k] for k in ('model', 'GPU_count', 'arm', 'lower_tested_rate', 'upper_rejected_rate', 'starts_at_lower_rate', 'accepted_lossless_capacity')] for r in bound])
    text += '\nPassing requires point attainment at least90%, a completed trace, and the preregistered weak backlog check.\n'
    text += 'The complete attempted-rate table is `03_primary_w1_capacity.csv`. It retains failures and drain duration.\n'
    text += 'Rate ceilings without a failing upper point remain NOT_BRACKETED. Finite traces cannot establish an unrestricted maximum.\n'
    text += 'A raw SLO-passing point is not an accepted capacity when output correctness fails.\n'
    text += 'Wilson attainment intervals are not simultaneous, selection-adjusted, or independent-run capacity intervals.\n\n'
    text += '## Actual W1 serving measurements\n\n'
    text += table(['Model', 'GPUs', 'Arm', 'Offered req/s', 'Seed', 'Completed/planned', 'Joint %', 'Window goodput req/s', 'Drain s', 'Queue scope'], [
        [r['model'], r['GPU_count'], r['arm'], r['offered_rate'], r['seed'], f"{r['completed']}/{r['planned_arrivals']}",
         r.get('joint_attainment_percent', 'NOT_MEASURED'), r.get('qualified_requests_s_window', 'NOT_MEASURED'),
         r.get('drain_seconds', 'NOT_MEASURED'), r.get('queue_stability', 'NOT_MEASURED')] for r in rows])
    text += '\nEvery arrival belongs to the declared trace denominator. Drain-inclusive goodput remains a separate column.\n'
    text += 'A failed window cannot certify capacity even if some partial outputs look fast.\n\n'
    text += '## Direct paired finite-trace comparisons\n\n'
    text += table(['Model', 'GPUs', 'Offered rate', 'Baseline', 'Starts', 'Qualified baseline/B', 'Goodput ratio', '95% CI', 'Exact outputs'], [
        [row['model'], row['GPU_count'], row['offered_rate'], row['baseline'], row['independent_starts'],
         f"{row['baseline_qualified']}/{row['candidate_qualified']} of {row['total_arrivals']}",
         f"{row['qualified_goodput_ratio']:.3f}×" if isinstance(row['qualified_goodput_ratio'], float) else row['qualified_goodput_ratio'],
         row['qualified_goodput_ratio_ci95'], row['all_outputs_exact']] for row in pair_rows])
    text += '\nThe intervals resample whole paired starts, not independent tokens or repeated copies of one prompt.\n'
    text += 'They condition on observed trajectories. They are not capacity, simultaneous, or selection-adjusted intervals.\n'
    text += 'Output-confounded rows remain visible but cannot support an equivalent-work speedup.\n\n'
    text += '`latency_run_cluster_ci.csv` reports whole-start uncertainty for the mean of each start\'s latency percentiles.\n'
    text += 'It does not claim a pooled-request tail interval. Single-start cells have no between-start confidence interval.\n\n'
    text += '## Latency, fidelity, and causal evidence\n\n'
    text += '`04_latency_percentiles.csv` contains per-request TTFT/TPOT/E2E p50/p95/p99 and observed-emission ITL p50/p95/p99/max.\n'
    text += '`06_transfer_and_critical_path.csv` contains cohort transport counters, exposed handoff stall, and sampled per-device HBM.\n'
    text += 'Transport bytes include final hidden rows. The KV-only accounting subtracts source-derived BF16 hidden-row sizes and labels that derivation.\n'
    text += 'Out-of-band control-message bytes remain NOT_MEASURED.\n'
    text += 'Cumulative transfer observations can overlap. They are not additive E2E or pure GPU DMA time.\n'
    text += f'`diagnostic_transfer_timing.csv` contains {len(diagnostic_rows)} separately instrumented transfer records.\n'
    text += 'It records readiness observation, submission, completion, descriptor cost, and the associated prefill callback where available.\n'
    text += 'Pure attention/FFN/dispatch GPU-critical-path percentage attribution is NOT_MEASURED.\n\n'
    text += f'Separate native GPU profiles produce {len(profile_rows)} device-summary rows in `profile_device_summary.csv`.\n'
    text += 'Those rows use unions of recorded device events. They do not measure clean serving E2E or its causal component shares.\n'
    text += 'The event window spans the first and last recorded GPU event. It omits leading and trailing idle time.\n\n'
    text += '`e2e_component_percent.csv` separates observed request phases into percentage shares when all boundaries exist.\n'
    text += 'Its denominator is summed request E2E wall time. It is not cohort makespan or GPU utilization.\n'
    text += 'Prefill and decode phases include scheduling, backpressure, and synchronization. They are not pure compute time.\n\n'
    text += 'The initial sequential PD_G1/B qualification passes 40/40 outputs and depth traces per model.\n'
    text += 'The separate reversed-order byte checks pass valid KV and hidden identity.\n'
    text += 'Fresh official-HF triage matches Ouro 7/8 and Huginn 6/8 outputs. The sample is diagnostic, not an accuracy estimate.\n'
    text += 'The prior frozen counts remain separate at 88/100 and 89/100. Full benchmark answer accuracy is NOT_MEASURED.\n'
    text += 'A BF16/backend explanation for every mismatch is not proved. The global reference status remains FIDELITY_UNRESOLVED.\n\n'
    text += '`paired_output_fidelity.csv` records each common-trace B/control comparison, including output and stop differences.\n'
    text += '`baseline_output_fidelity.csv` also compares colocated and ordinary PD outputs against matched PD_G1, without requiring a B run.\n'
    text += 'It retains generated-token counts and recorded decode-loop work. Natural-workload differences do not become equal-work speedups.\n'
    text += 'B performance acceptance stops after a matched PD_G1 sequence mismatch. Existing timings remain diagnostic.\n'
    text += 'Different outputs or lengths cannot be described as equivalent-work lossless speedup.\n\n'
    text += f'Saved-evidence integrity checks report {len(evidence_issues)} invariant violations. Read `evidence_validation.csv`.\n'
    text += 'The audit checks source pins, recorded depth, GPU mapping, completion counts, and background evidence persistence.\n'
    text += 'An integrity pass does not resolve model fidelity or prove benchmark accuracy.\n\n'
    text += '### Model-level serving output gates\n\n'
    gate_rows = []
    for model in ('ouro', 'huginn'):
        failure_path = ROOT / 'state' / (model + '_B_serving_parity_failure.json')
        if failure_path.exists():
            failure = json.loads(failure_path.read_text())
            gate_rows.append([model, 'CLOSED', f"{failure['exact']}/{failure['N']}", failure['left'], failure['right']])
        else:
            gate_rows.append([model, 'No recorded gate failure', 'See paired_output_fidelity.csv', '', ''])
    text += table(['Model', 'Serving B gate', 'Exact sequences', 'Control run', 'B run'], gate_rows)
    triage_path = ROOT / 'configs/mismatch_triage.json'
    if triage_path.exists():
        triage_rows = []
        for model, value in json.loads(triage_path.read_text()).items():
            for name in ('control_repeat_vs_original_control', 'control_repeat_vs_original_B', 'sequential_matched_output', 'concurrent_KV_and_hidden_identity'):
                test = value.get(name)
                triage_rows.append([model, name, f"{test['exact']}/{test['N']}" if test else 'NOT_MEASURED'])
        text += '\n' + table(['Model', 'Post-serving diagnostic', 'Matched sequences or explicit KV+hidden byte pairs'], triage_rows)
        text += '\nA self-repeat difference can show scheduling sensitivity. It does not prove that every B mismatch is harmless.\n'
        text += 'Selected mismatch prompts do not estimate downstream accuracy. The serving gate remains closed.\n\n'
    text += '## Workload coverage and novelty\n\n'
    text += 'W1 uses 200 distinct ShareGPT conversations. Supplementary W2/W3/W4 are fixed before B serving.\n'
    text += 'Long-window extensions cycle the same source pool with unique occurrence IDs and new request seeds.\n'
    text += 'Extended W1 uses complete cycles of all 200 source prompts. It does not substitute an easier low-rate prefix.\n'
    text += 'The extension creates no additional distinct source texts. Prefix reuse is not enabled.\n'
    text += 'Actual EOS and cap stops remain separate. Real-text bounded generation is not guaranteed complete-answer serving.\n\n'
    text += '`05_workload_w2_w3_w4.csv` retains secondary positive and negative observations.\n'
    text += '`07_scale_2_4_8gpu.csv` retains equal-GPU deployments and their actual P:D partitions.\n'
    text += 'A one-start scale result cannot supply an accepted between-start speedup interval.\n'
    text += 'Generic immutable logical-layer streaming can have the same exposed readiness DAG as B.\n'
    text += 'A separately qualified GenericLayer engine is NOT_MEASURED. Its label is explicitly rejected by the worker.\n'
    text += 'Read `08_novelty_generic_streaming.md`. Phase separation and ordinary NIXL overlap are not new contributions.\n\n'
    text += '## Failures, omissions, and reproduction\n\n'
    failures = [r for r in latency if r['status'] not in ('COMPLETE',)]
    text += table(['Run', 'Status', 'Error'], [[r['run'], r['status'], r.get('error', '')] for r in failures]) if failures else 'No terminal run failure appears in the current saved tables.\n'
    text += '\nUnexecuted model/GPU/method cells remain explicit in the decision and capacity matrices.\n'
    text += f'The log scan retains {len(warning_rows)} error or traceback lines in `runtime_warning_inventory.csv`.\n'
    text += 'Some log warnings can occur after the final completion during process cleanup. They do not automatically change the saved terminal status.\n'
    text += 'Their exact position and text remain available. No warning is silently deleted.\n'
    text += 'The report does not infer safe deployment, official accuracy, or missing GPU speedups from structural accounting.\n'
    text += '`reproduce.md`, `commands.jsonl`, per-run source ZIPs, and the dataset manifest define reproduction.\n'
    text += 'Clean serving is isolated from other owned GPU workers and burners. The shared host is not claimed exclusive.\n'
    text += 'Owned burners resume after timing and at cleanup. The prior A/B/C evidence and unrelated staged work remain preserved.\n'
    from .repair_reporting import report as repair_report
    repair_report(final=final)
    repair_decisions = read('repair_epoch_model_decisions.csv')
    text += '\n## Separate repair epoch: do not replace or pool the original evidence\n\n'
    text += table(['Model', 'GPUs', 'Epoch', 'Verdict', 'Internal qualification', 'Serving runs'],
                  [[row[key] for key in ('model', 'GPU_count', 'execution_epoch', 'verdict',
                                         'internal_qualification', 'measured_serving_runs')]
                   for row in repair_decisions])
    text += '\nRead `repair_epoch_report.md` for full paired output counts, actual serving data, and reference limitations.\n'
    text += 'Canonicalization can change outputs relative to the original run and can add GPU work.\n'
    text += 'A matched-runtime repair is not an original-model-lossless or optimized-engine performance GO.\n'
    text += '\n## Detailed closeout and the earlier SLO result\n\n'
    text += 'Read `10_why_previous_gain_did_not_transfer.md` for the frozen prior comparison and CPU-only SLO rescoring.\n'
    text += 'Read `11_final_summary_ko.md` for the Korean review, unresolved causes, and measured component shares.\n'
    text += 'The dataset storage change is lossless. Run the restore command in `reproduce.md` after cloning.\n'
    (ROOT / '09_final_decision.md').write_text(text)
    figures(rows, latency)
    if final:
        if not (ROOT / '05_workload_w2_w3_w4.csv').exists():
            save('05_workload_w2_w3_w4.csv', [dict(model=model, workload=workload, evidence='NOT_MEASURED',
                                                 reason='No completed supplemental serving job before closeout.')
                                            for model in ('ouro', 'huginn') for workload in ('W2', 'W3', 'W4')])
        research_path = ROOT / 'research_state.json'
        research = json.loads(research_path.read_text())
        research.update(status='FINALIZED_WITH_LIMITATIONS', final_report='09_final_decision.md',
                        finalized_epoch=time.time(), pending=['Scoped commit and remote SHA verification'],
                        decision_file='model_GPU_decisions.csv', integrity_issues=evidence_issues)
        write(research_path, research)
    return decisions


def publish():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    test = subprocess.run([PYTHON, '-m', 'pytest', '-q',
                           'poc_flashvep/looped_pd_b_slo_serving/test_contracts.py',
                           '--junitxml=' + str(ROOT / 'configs/final_cpu_tests.xml')],
                          cwd=REPO, env=environment, capture_output=True, timeout=120)
    (ROOT / 'logs/final_cpu_tests.log').write_bytes(test.stdout + test.stderr)
    if test.returncode:
        raise RuntimeError('Final CPU contract tests failed. No publication action was taken.')
    from ..looped_pd_abc.telemetry import flush
    flush()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    if branch != 'flashvep/looped-pd-b-slo-serving-poc':
        raise RuntimeError('Unexpected branch. No publication action was taken.')
    own = ['poc_flashvep/looped_pd_b_slo_serving', 'poc_flashvep/reports/looped_pd_b_slo_serving']
    command = ['git', 'ls-files', '--stage', '-z', '--', '.', *[':!' + p for p in own], ':!poc_flashvep/reports/looped_pd_b_slo_serving_poc_spec.md']
    before = hashlib.sha256(subprocess.check_output(command, cwd=REPO)).hexdigest()
    expected = json.loads((ROOT / 'research_state.json').read_text())['outside_index_sha256']
    if before != expected:
        raise RuntimeError('The outside staged index changed. Preserve it and require publication review.')
    subprocess.run(['git', 'add', '--', *own], cwd=REPO, check=True)
    diagnostic_pth = ROOT / 'configs/01_looped_pd_b_timing.pth'
    if diagnostic_pth.exists():
        # The repository ignores model .pth payloads globally. This exact file is plain diagnostic startup code.
        subprocess.run(['git', 'add', '-f', '--', str(diagnostic_pth.relative_to(REPO))], cwd=REPO, check=True)
    # These files contain captured activation rows and logits, not model weights.
    # The repository's generic checkpoint ignore rule also matches these small artifacts.
    for artifact in sorted((ROOT / 'raw').glob('repair_capture*/actual_operands_*.pt')):
        if artifact.stat().st_size > 50_000_000:
            raise RuntimeError('Review an unexpectedly large diagnostic operand artifact before publication')
        subprocess.run(['git', 'add', '-f', '--', str(artifact.relative_to(REPO))], cwd=REPO, check=True)
    paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z', '--', *own], cwd=REPO)
    if paths:
        result = subprocess.run(['git', 'commit', '--only', '--pathspec-from-file=-', '--pathspec-file-nul',
                                 '-m', 'Record fixed-depth B open-loop SLO serving evidence and explicit fidelity limits'],
                                cwd=REPO, input=paths, capture_output=True)
        (ROOT / 'logs/publication_commit.log').write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError('Scoped commit failed')
    after = hashlib.sha256(subprocess.check_output(command, cwd=REPO)).hexdigest()
    if after != before:
        raise RuntimeError('The outside index did not remain unchanged')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    for name in ('00_scope_baselines_and_slo_lock.md', '01_dataset_trace_manifest.md',
                 '02_correctness_and_reference_fidelity.md', '03_primary_w1_capacity.csv',
                 '04_latency_percentiles.csv', '05_workload_w2_w3_w4.csv',
                 '06_transfer_and_critical_path.csv', '07_scale_2_4_8gpu.csv',
                 '08_novelty_generic_streaming.md', '09_final_decision.md', 'reproduce.md',
                 'b_repair_diagnosis.md', 'repair_epoch_report.md', 'repair_epoch_model_decisions.csv'):
        relative = str((ROOT / name).relative_to(REPO))
        subprocess.run(['git', 'cat-file', '-e', sha + ':' + relative], cwd=REPO, check=True)
    push = subprocess.run(['git', 'push', '-u', 'origin', 'HEAD:refs/heads/' + branch], cwd=REPO, capture_output=True, timeout=600)
    (ROOT / 'logs/publication_push.log').write_bytes(push.stdout + push.stderr)
    if push.returncode:
        raise RuntimeError('Push failed. The local commit remains preserved.')
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + branch], cwd=REPO, text=True).split()[0]
    write(ROOT / 'state/publication_receipt.json', dict(commit=sha, remote=remote, verified=sha == remote,
                                                      branch=branch, time=time.time(), outside_index_sha256=after))
    if sha != remote:
        raise RuntimeError('Remote SHA differs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    report(args.final)
    if args.publish:
        if not args.final:
            raise ValueError('Do not publish an unreviewed final result as complete')
        publish()


if __name__ == '__main__':
    main()
