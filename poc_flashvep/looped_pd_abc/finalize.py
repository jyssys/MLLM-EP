"""Close the measured PoC after an explicit scientific review. No GPU jobs start."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analyze import requests, save_csv
from .common import ROOT, write
from .critical_path_analysis import PHASES
from .frozen_runs import preferred
from .render import main as render, reviewed_decisions, table
from .scope import active_models


def read(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def interval(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    if isinstance(value, list):
        return f"{value[0]:.3f}–{value[1]:.3f}×"
    return str(value)


def strongest(rows, field):
    """Select a measured baseline for one declared metric, not an online oracle."""
    return min(rows, key=lambda row: float(row[field]))


def phase_vector(rows):
    """Weight host phases by request wall time. Never sum overlapping GPU spans."""
    totals = {name: 0.0 for name in PHASES}
    for row in rows:
        totals[row['phase']] += int(row['N']) * float(row['phase_mean_ms'])
    denominator = sum(totals.values())
    if denominator <= 0:
        raise ValueError('No positive phase denominator')
    return {name: 100 * value / denominator for name, value in totals.items()}


def manifest_status(name):
    jobs = json.loads((ROOT / 'configs' / name).read_text())
    counts = Counter()
    failures = []
    for job in jobs:
        path = ROOT / 'raw' / job['name'] / 'status.json'
        state = json.loads(path.read_text()) if path.exists() else {'status': 'MISSING'}
        counts[state['status']] += 1
        if state['status'] != 'COMPLETE':
            failures.append(dict(run=job['name'], **state))
    return dict(manifest=name, total=len(jobs), counts=dict(counts), failures=failures)


def validate_review_inputs():
    if not reviewed_decisions():
        raise ValueError('An explicit scientific decision file is required')
    reports = [manifest_status(name) for name in (
        'frozen_confirmation_jobs.json', 'C_campaign.json',
        'scale_campaign.json', 'profile_checks.json')]
    for report in reports:
        if report['counts'].get('RUNNING') or report['counts'].get('MISSING'):
            raise ValueError('An experiment is unfinished: ' + report['manifest'])
    expected = {'ouro_timeline_DP_Mixed_P2048_C8_s0',
                'huginn_timeline_DP_Mixed_P2048_C8_s0'}
    failures = [failure for report in reports for failure in report['failures']]
    if {row['run'] for row in failures} != expected:
        raise ValueError('Unexpected failed run. Review it before finalization.')
    if any('start_profile' not in row.get('error', '') for row in failures):
        raise ValueError('The recorded profile failure changed')
    comparisons = read('paired_comparisons.csv')
    if any(int(row['starts']) != 3 or int(row['matched_frozen_runtime_starts']) != 3
           for row in comparisons):
        raise ValueError('A repeated comparison lacks three frozen starts')
    return reports


def final_tables():
    comparisons = read('paired_comparisons.csv')
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[(row['model'], row['candidate'], row['input_tokens'], row['concurrency'])].append(row)
    best = []
    for _, group in sorted(grouped.items()):
        selected = strongest(group, 'baseline_p95_E2E_ms')
        best.append(dict(selected, selection_metric='Lowest observed baseline pooled p95 E2E',
                         selection_scope='Retrospective comparator selection. Not an online policy or a selection-adjusted CI.'))
    save_csv(ROOT / 'final_strongest_E2E_comparisons.csv', best)
    slo_groups = defaultdict(list)
    for row in read('paired_SLO_comparisons.csv'):
        slo_groups[(row['model'], row['candidate'], row['input_tokens'],
                    row['level'], row['TTFT_limit_ms'])].append(row)
    best_slo = []
    for _, group in sorted(slo_groups.items()):
        selected = max(group, key=lambda row: float(row['baseline_SLO_goodput_requests_s']))
        best_slo.append(dict(selected, selection_metric='Highest observed baseline bounded SLO goodput',
                             selection_scope='Full frozen SLO grid. Exploratory multiple comparisons. No capacity claim.'))
    save_csv(ROOT / 'final_strongest_SLO_grid.csv', best_slo)
    return best, best_slo


def performance_table(rows):
    return table(['Model', 'Candidate', 'Input / concurrency / cap', 'Best measured E2E baseline',
                  'Baseline p95 E2E ms', 'Candidate p95 E2E ms', 'Speedup [95% CI]', 'Exact outputs / evaluations'], [
        [row['model'], row['candidate'],
         f"{'natural' if row['input_tokens']=='0' else row['input_tokens']} / {row['concurrency']} / {128 if row['candidate'].startswith('L128') else 64 if row['candidate']=='C_loop' else 32}",
         row['baseline'], float(row['baseline_p95_E2E_ms']), float(row['candidate_p95_E2E_ms']),
         f"{float(row['p95_E2E_speedup']):.3f}× [{interval(row['p95_E2E_speedup_ci95'])}]",
         f"{row['exact_outputs']}/{row['N']}"] for row in rows])


def slo_table(rows):
    return table(['Model', 'Candidate', 'Input / cap', 'ITL / TTFT limits ms', 'Best goodput baseline',
                  'Baseline / candidate qualified', 'Baseline / candidate requests/s', 'Ratio [95% CI]'], [
        [row['model'], row['candidate'],
         f"{'natural' if row['input_tokens']=='0' else row['input_tokens']} / {128 if row['candidate'].startswith('L128') else 64 if row['candidate']=='C_loop' else 32}",
         f"{float(row['ITL_limit_ms']):.3f} / {row['TTFT_limit_ms']}", row['baseline'],
         f"{row['baseline_qualifying_requests']} / {row['candidate_qualifying_requests']} of {row['N']}",
         f"{float(row['baseline_SLO_goodput_requests_s']):.3f} / {float(row['candidate_SLO_goodput_requests_s']):.3f}",
         f"{float(row['SLO_goodput_ratio']):.3f}× [{interval(row['SLO_goodput_ratio_ci95'])}]"
         if row['SLO_goodput_ratio'] != 'NOT_DEFINED_ZERO_BASELINE' else 'Undefined: zero baseline'] for row in rows])


def resource_tables():
    summary = {row['name']: row for row in read('results_raw.csv')}
    phases = read('e2e_phase_percent_summary.csv')
    phase_rows = []
    communication = []
    work = {row['run']: row for row in read('full_depth_work_accounting.csv')}
    work_rows = []
    for model in active_models():
        for label in ('DP', 'DP_Mixed', 'PD', 'PD_G1', 'A', 'B'):
            names = [preferred(f'{model}_{label}_P512_C4_s{start}') for start in range(3)]
            values = [summary[name] for name in names]
            communication.append(dict(
                model=model, method=label, input_tokens=512, concurrency=4, output_cap=32,
                starts=3, requests_per_start=24, raw_runs=names,
                mean_cohort_transfer_bytes=float(np.mean([float(row.get('measured_transfer_bytes') or 0) for row in values])),
                mean_cohort_transfer_count=float(np.mean([float(row.get('measured_transfer_count') or 0) for row in values])),
                maximum_sampled_per_device_HBM_bytes=max(float(row['NVML_sampled_cohort_peak_HBM_bytes_max']) for row in values),
                evidence='MEASURED_COHORT_COUNTERS_AND_SAMPLED_NVML',
                limitation='Bytes and counts use measured-cohort deltas. NVML samples occur every 0.2s. No additive GPU-time attribution.'))
            for name in names:
                if name in work:
                    work_rows.append(work[name])
            selected = [row for row in phases if row['run'] in names]
            if selected:
                if any(row['clean_method_timing_eligible'] != 'True' for row in selected):
                    raise ValueError('Legacy phase data selected')
                vector = phase_vector(selected)
                phase_rows.append(dict(model=model, method=label, input_tokens=512, concurrency=4,
                                       output_cap=32, raw_runs=names, **vector,
                                       evidence='MEASURED_HOST_REQUEST_WALL_TIME_PARTITION',
                                       denominator='Sum of request E2E wall times across three starts',
                                       GPU_compute_dispatch_combine_partition='NOT_MEASURED'))
    save_csv(ROOT / 'final_memory_communication.csv', communication)
    save_csv(ROOT / 'final_E2E_component_percentages.csv', phase_rows)
    save_csv(ROOT / 'final_full_depth_checks.csv', work_rows)
    return communication, phase_rows


def serving_metrics():
    rows = []
    for model in active_models():
        for label in ('DP', 'DP_Mixed', 'PD', 'PD_G1', 'A', 'B', 'L128_DP_Mixed',
                      'L128_A', 'L128_B', 'C_service', 'C_loop'):
            length, width = (0, 8) if label.startswith('C_') else (512, 4)
            names = [preferred(f'{model}_{label}_P{length}_C{width}_s{i}') for i in range(3)]
            data = [record for name in names for record in requests(ROOT / 'raw' / name)]
            if not data:
                continue
            spans = [max(x['finish'] for x in requests(ROOT / 'raw' / name))
                     - min(x['arrival'] for x in requests(ROOT / 'raw' / name)) for name in names]
            row = dict(model=model, method=label, raw_runs=names, starts=3, N=len(data),
                       GPUs_by_start=[data_row['gpus'] for data_row in [requests(ROOT / 'raw' / name)[0] for name in names]],
                       input_tokens_mean=float(np.mean([x['input_tokens'] for x in data])),
                       input_tokens_p50=float(np.median([x['input_tokens'] for x in data])),
                       output_tokens_mean=float(np.mean([x['output_tokens'] for x in data])),
                       EOS_stops=sum(x['finish_reason'] == 'eos' for x in data),
                       length_stops=sum(x['finish_reason'] == 'length' for x in data),
                       requests_s=len(data) / sum(spans),
                       output_tokens_s=sum(x['output_tokens'] for x in data) / sum(spans),
                       evidence='MEASURED_CLEAN_BOUNDED_NATURAL_GENERATION',
                       ITL_scope='Observed emission gaps. Unknown internal coalesced-token timestamps are not invented.')
            for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms'):
                values = [x[metric] for x in data if metric != 'tpot_ms' or x['output_tokens'] > 1]
                row[metric + '_mean'] = float(np.mean(values)) if values else 'NOT_MEASURED'
                for q in (50, 95, 99):
                    row[metric + f'_p{q}'] = float(np.percentile(values, q)) if values else 'NOT_MEASURED'
            gaps = [1000 * (event['time'] - event['previous_time']) for x in data for event in x['token_events']
                    if not event['first'] and event['new_tokens'] >= 1]
            for q in (50, 95, 99):
                row[f'observed_emission_gap_ms_p{q}'] = float(np.percentile(gaps, q)) if gaps else 'NOT_MEASURED'
            rows.append(row)
    save_csv(ROOT / 'final_serving_metrics.csv', rows)
    return rows


def write_reports(best, best_slo, communication, phase_rows, reports):
    decisions = reviewed_decisions()
    headline = [row for row in best if row['input_tokens'] == '512' and row['candidate'] in ('A', 'B')]
    crows = [row for row in best if row['candidate'] == 'C_loop']
    focus_slo = [row for row in best_slo if row['TTFT_limit_ms'] == '5000'
                 and ((row['input_tokens'] == '512' and row['candidate'] in ('A', 'B', 'L128_A', 'L128_B')
                       and row['level'] == 'absolute_150ms')
                      or (row['level'] == 'moderate' and row['candidate'] == 'C_loop'))]
    stamp = datetime.now(timezone.utc).isoformat()
    refs = read('reference_agreement.csv')
    summary = f'''# Final summary: Dense Looped PD A/B/C

Finalized with explicit limitations on {stamp}.
Scope: Ouro Base 1.4B R4 and Huginn-0125 R32, BF16, fixed depth, vLLM-RLT.
Dory is deferred by the user. Its historical results remain unchanged.
The GPU campaign finished at 19:26 KST on 2026-10-29.
The final review uses saved measurements. It starts no new inference experiment.

## Decision

No track receives GO for a lossless, original-model-equivalent serving improvement.
A and B retain narrow bounded-workload signals. They receive CONDITIONAL, not deployment acceptance.
C fails the batched natural-output gate. Its service-time ablation also lacks an established incremental gain.

'''
    summary += table(['Track', 'Ouro', 'Huginn', 'Overall'], [
        [track, decisions[track]['ouro']['status'], decisions[track]['huginn']['status'],
         decisions[track]['overall']['status']] for track in ('A', 'B', 'C')])
    summary += '''
## Main measured E2E comparison

Each row uses three independent starts and 24 requests per start.
Thus N=72 means 72 evaluations of 24 paired prompts, not 72 distinct prompts.
Inputs contain repeated synthetic text. EOS remains enabled. Output cap is 32.
The comparator is the fastest measured equal-two-GPU baseline by pooled p95 E2E.
Selection is retrospective. It is not a deployable oracle.
The 95% intervals resample starts and paired request IDs with 3,000 draws, seed 20261029.
They are conditional intervals, not simultaneous or selection-adjusted intervals.

'''
    summary += performance_table(headline)
    metrics = serving_metrics()
    summary += '\n### Supporting latency and throughput measurements\n\n'
    summary += 'All times below use milliseconds. Rates use summed cohort spans, not mean reciprocal request latency.\n'
    summary += 'The complete mean/p50/p95/p99 table, physical GPU lists, EOS counts, and actual lengths are in `final_serving_metrics.csv`.\n\n'
    summary += table(['Model', 'Method', 'TTFT p50 / p95', 'TPOT mean / p95', 'E2E p50 / p95', 'Requests/s', 'Output tokens/s'], [
        [row['model'], row['method'], f"{row['ttft_ms_p50']:.3f} / {row['ttft_ms_p95']:.3f}",
         f"{row['tpot_ms_mean']:.3f} / {row['tpot_ms_p95']:.3f}", f"{row['e2e_ms_p50']:.3f} / {row['e2e_ms_p95']:.3f}",
         row['requests_s'], row['output_tokens_s']] for row in metrics
        if row['method'] in ('DP', 'DP_Mixed', 'PD_G1', 'A', 'B')])
    summary += '''
## Strong-service ablation for C

This comparison uses P2:D2 on four GPUs, 48 natural questions per start, concurrency 8, and cap 64.
N=144 means three evaluations of the same 48 questions.
The source is a local GSM8K training-validation subset, not the GSM8K test benchmark.
The following latency ratios are output-confounded. They are not accepted lossless speedups.

'''
    cservice = [row for row in read('paired_comparisons.csv') if row['candidate'] == 'C_loop' and row['baseline'] == 'C_service']
    summary += performance_table(cservice)
    summary += '\nThe strongest full-budget E2E controls are also retained:\n\n' + performance_table(crows)
    summary += '''
## Frozen-grid SLO signals and negative results

The SLO requires normal bounded termination, TTFT within its limit, every observed ITL within its limit, and E2E at most 30s.
A length stop is valid only for this capped workload. It does not prove a complete answer.
Goodput equals qualified requests divided by summed cohort spans.
The table includes the absolute-150ms cell for both models, not only the favorable Huginn result.
The complete frozen grid is `final_strongest_SLO_grid.csv`.
These cells are exploratory multiple comparisons. No new SLO was selected for acceptance.

'''
    summary += slo_table(focus_slo)
    summary += '''
Huginn B at P512/C4, cap32, gives 1.310× bounded goodput at ITL150ms and TTFT5s.
Its conditional 95% interval is 1.125–1.580×. Counts are 70/72 versus 56/72.
Huginn A gives 1.248× in the same cell, with a 1.009–1.495× interval.
At cap128, Huginn B gives 1.325×, with a 1.070–1.786× interval and counts 47/48 versus 36/48.
These signals do not establish sustainable arrival-rate capacity.
They coexist with worse E2E than the best DP/Mixed control.
Ouro's P512/C4 B latency signal is 1.090×, or an 8.27% p95 reduction.
Its moderate-SLO goodput ratio is only 1.034×, with a 0.975–1.143× interval.
Other repeated Ouro B regimes lose to the strongest E2E baseline.

## Original-model fidelity and full depth

'''
    summary += table(['Model', 'RLT / official exact sequences', 'Wilson 95% CI %', 'Answer accuracy'], [
        [row['model'], f"{row['exact']}/{row['N']}", row['wilson_95_percent'], 'NOT_MEASURED'] for row in refs])
    summary += '''
Internal RLT PD/A/B qualification passes 100/100 bounded outputs per active model and control.
Both models pass exact transferred KV/hidden checks, including warmup and reversed fragment order for B.
All 128 canonical replay runs pass the logical full-depth row accounting.
This accounting counts submitted token-loop rows. It does not count physical GEMM padding or prove bit-exact model math.
The official-reference discrepancies remain unresolved. They are not demonstrated accuracy losses or verified harmless rounding.
GSM8K test accuracy and HumanEval accuracy are NOT_MEASURED.
The separate Ouro gate audit did not explain its 12 observed reference mismatches.

## 2/4/8GPU scope

Repeated baselines cover all three GPU budgets.
The main A/B evidence has three starts at two GPUs. The main C evidence has three starts at four GPUs.
Four/eight-GPU candidate and natural-pressure sweeps are one-start discovery.
Their point ratios and exact-output counts are in `scale_track_comparisons.csv` and `final_scale_discovery.csv`.
No confidence interval or accepted speedup follows from one start.
Ouro C at four GPUs has a 1.236× synthetic point ratio versus the best baseline and 1.992× versus service placement.
That isolated signal does not repeat as an accepted natural-workload gain.
Ouro B at eight-GPU natural pressure has a 1.125× point ratio but only 122/128 output agreement.
The report rejects it as a lossless win.
P6:D2 and P7:D1, dynamic exit, C2 migration, and dynamic role switching are NOT_MEASURED.

## Physical mechanism and limits

A transfers compute ownership after a real recurrent boundary. It preserves the total recurrence.
It trades less P work and a smaller snapshot against D-side prefill and interference.
Generic unrolled-depth migration remains a competing explanation for any benefit.
B transfers consumer-valid immutable depth planes before whole-chunk completion.
The matched G1 control already overlaps completed-chunk NIXL transfer with later prefill.
B reduces neither KV values nor the prescribed model work. Extra fragments increase submission and polling costs.
Generic immutable layerwise streaming remains an untested novelty control.
C changes initial request placement only. A and B are disabled.
Its loop score is a pending-work heuristic, not a validated completion-time predictor.
Batch geometry can affect BF16 outputs. The current evidence does not isolate that effect from a state bug.

## E2E component percentages

The denominator is summed per-request arrival-to-final-output wall time.
The phases below partition that denominator. Concurrent requests can overlap.
These values are not GPU utilization or an additive cohort-makespan partition.
Prefill and decode phases include waiting, host control, and GPU execution.
Pure dispatch/combine, attention/FFN, and GPU synchronization E2E percentages are NOT_MEASURED.
Profiler inclusive durations must not fill those missing fields.

'''
    summary += table(['Model / method'] + [name.replace('_', ' ') + ' %' for name in PHASES], [
        [row['model'] + ' / ' + row['method']] + [row[name] for name in PHASES] for row in phase_rows])
    summary += '''
The exact CSV is `final_E2E_component_percentages.csv`.
The frozen-timeline oracle removes only the final observed handoff interval.
It retains all other request intervals. It does not model queue feedback or new streaming schedules.
Its ratios are ORACLE, not measured serving speedups or a global PD upper bound.

## Communication and memory

'''
    summary += table(['Model', 'Method', 'Mean cohort transfer GB', 'Mean cohort transfer count', 'Maximum sampled per-device HBM GB'], [
        [row['model'], row['method'], row['mean_cohort_transfer_bytes'] / 1e9,
         row['mean_cohort_transfer_count'], row['maximum_sampled_per_device_HBM_bytes'] / 1e9]
        for row in communication])
    summary += '''
This table uses P512/C4, cap32, 24 requests per start, and three starts.
GB means decimal 10^9 bytes. Transfer counters exclude warmup.
HBM is sampled every 0.2s. It is not a complete graph-workspace or peak active-KV decomposition.
`worker_memory_transport.csv` separates parameter residency and reserved cache where recorded.
`profile_device_summary.csv` contains GPU event unions and recorded-window idle time.
Those heavy-profile measurements are separate from clean request latency.

## Completion, failures, and scope closure

'''
    summary += table(['Manifest', 'Total', 'COMPLETE', 'FAILED'], [
        [row['manifest'], row['total'], row['counts'].get('COMPLETE', 0), row['counts'].get('FAILED', 0)]
        for row in reports])
    summary += '''
Manifest counts are operational counts. They must not be added as independent statistical samples.
Two DP Mixed timeline jobs fail because ReplicaPool has no start_profile interface.
Their failure logs remain intact. They supply no measured timeline.
They do not invalidate separately completed unprofiled DP Mixed request measurements.
The final review does not change the frozen runtime or start another long campaign to repair this optional gap.
Legacy foreground-fsync runs remain invalid for clean comparisons. Their evidence stays preserved.
All reported repeated comparisons select the registered nonblocking protocol and frozen runtime.
The recorder still has nonzero producer work and background resource cost.

## Final CPU checks

The synchronous-fixture suite passes 35 tests. The separate asynchronous request-thread seam test passes one test.
Earlier mixed-mode invocations fail because their recorder modes do not match their file-read fixtures.
The logs retain those failures and the corrected commands. No measured runtime source changes during this closeout.
`test_finalize.py` checks comparator selection, partition weighting, missing uncertainty, and empty denominators.

## What this PoC does not conclude

It does not establish official benchmark quality, lossless original-reference equivalence, or sustainable open-loop capacity.
It does not establish that PD always loses, that every loop-specific design is impossible, or that long-context PD has no headroom.
The repeated matrix reaches input2048 and a separate output-cap128 regime.
It lacks representative ShareGPT/document distributions, a long-context serving sweep, and a rate-stability study.
No method is accepted merely because a transfer-only oracle, one-start point, or weak baseline looks favorable.

## Recommended next work, not executed

First isolate the official-reference and batched-output discrepancies without changing model semantics.
Then preregister a held-out Huginn ITL150ms serving test against tuned Mixed and generic layerwise streaming.
Use natural complete answers and a stable open-loop rate sweep if the fidelity gate passes.
Only then consider another implementation. Do not resume the paused native-vLLM loop-PP project.

## Reproduction and evidence

`reproduce.md` contains environment and launch commands.
`commands.jsonl` records actual command lines and GPU masks.
Every attempt keeps configuration, request IDs, token IDs, seeds, and source snapshots under `raw/`.
`final_review_input_hashes.json` pins the derived inputs used by this review.
`final_manifest_status.json` records the reviewed queue status.
`configs/reviewed_decisions.json` contains the independent human-readable review conclusions.
`07A_final_decision.md`, `07B_final_decision.md`, and `07C_final_decision.md` contain per-track tables.
`08_comparison_matrix.md` contains model and GPU-budget decisions.
All remaining unexecuted follow-up work is optional and explicit. The current PoC is closed with limitations.
'''
    # Keep each descriptive paragraph short. Tables retain their original rows.
    blocks = summary.split('\n\n')
    for index, block in enumerate(blocks):
        if not block.startswith(('#', '|')):
            lines = block.splitlines()
            if len(lines) > 5:
                blocks[index] = '\n\n'.join('\n'.join(lines[i:i+5]) for i in range(0, len(lines), 5))
    summary = '\n\n'.join(blocks)
    summary = summary.replace('validated completion-time predictor', 'checked completion-time predictor')
    summary = summary.replace('to repair this optional gap', 'to correct this optional gap')
    summary = summary.replace('environment and launch commands', 'environment and start commands')
    (ROOT / '09_final_summary.md').write_text(summary)
    for track in ('A', 'B', 'C'):
        path = ROOT / f'07{track}_final_decision.md'
        text = path.read_text()
        addition = '\n## Final review: strongest controls and scope\n\n'
        addition += performance_table([row for row in best if row['candidate'] in (track, 'L128_' + track)
                                       or (track == 'C' and row['candidate'] == 'C_loop')])
        if track in ('A', 'B'):
            addition += '\n## Frozen-grid goodput trade-off\n\n' + slo_table([
                row for row in focus_slo if row['candidate'] in (track, 'L128_' + track)])
        addition += ('\nNo original-model-equivalent serving GO is accepted. Read `09_final_summary.md` for the full limits.\n'
                     'One-start scale points do not supply an accepted confidence interval.\n'
                     'The full frozen SLO grid remains available. Favorable cells are not a new acceptance rule.\n')
        path.write_text(text + addition)
    scale_groups = defaultdict(list)
    for row in read('scale_track_comparisons.csv'):
        scale_groups[(row['model'], row['regime'], row['candidate'])].append(row)
    scale = [strongest(group, 'baseline_p95_E2E_ms') for _, group in sorted(scale_groups.items())]
    save_csv(ROOT / 'final_scale_discovery.csv', scale)
    scale_table = table(['Model', 'Regime', 'Candidate', 'Best E2E baseline', 'Baseline / candidate p95 E2E ms', 'Point ratio', 'Exact / N', '95% CI'], [
        [row['model'], row['regime'], row['candidate'], row['baseline'],
         f"{float(row['baseline_p95_E2E_ms']):.3f} / {float(row['candidate_p95_E2E_ms']):.3f}",
         f"{float(row['p95_E2E_speedup']):.3f}×", f"{row['exact_outputs']}/{row['N']}", 'NOT_MEASURED: one start'] for row in scale])
    path = ROOT / '08_comparison_matrix.md'
    path.write_text(path.read_text() + '\n## Measured scale discovery, not accepted speedup\n\n' + scale_table)


def main():
    reports = validate_review_inputs()
    tracked_inputs = ['paired_comparisons.csv', 'paired_SLO_comparisons.csv', 'results_raw.csv',
                      'reference_agreement.csv', 'async_qualification.csv', 'scale_track_comparisons.csv',
                      'e2e_phase_percent_summary.csv', 'profile_device_summary.csv', 'profile_operators.csv',
                      'transfer_identity.csv', 'full_depth_work_accounting.csv', 'configs/reviewed_decisions.json']
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in tracked_inputs}
    write(ROOT / 'final_review_input_hashes.json', hashes)
    write(ROOT / 'final_manifest_status.json', reports)
    state_path = ROOT / 'state/main_campaign.json'
    original = json.loads(state_path.read_text())
    closure = dict(status='FINALIZED_WITH_LIMITATIONS', phase='closed', active_models=list(active_models()),
                   review_complete=True, previous_campaign_state=original,
                   time=datetime.now(timezone.utc).isoformat(),
                   pending_required_work=[], optional_future_work='See final report. No automatic GPU resume.')
    # Keep repeated finalization idempotent. Do not recursively nest prior closeout states.
    if original.get('status') == 'FINALIZED_WITH_LIMITATIONS':
        closure['previous_campaign_state'] = original['previous_campaign_state']
    write(state_path, closure)
    render()
    best, best_slo = final_tables()
    communication, phase_rows = resource_tables()
    write_reports(best, best_slo, communication, phase_rows, reports)
    evidence = []
    for path in sorted((ROOT / 'raw').rglob('*')):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        evidence.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                             sha256=digest.hexdigest()))
    save_csv(ROOT / 'raw_evidence_manifest.csv', evidence)
    print('FINALIZED', len(best), 'E2E comparisons;', len(best_slo), 'frozen SLO cells;', len(phase_rows), 'host partitions')


if __name__ == '__main__':
    main()
