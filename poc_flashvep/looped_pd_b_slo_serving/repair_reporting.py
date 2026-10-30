"""Report repair qualifications separately from the rejected original epoch."""
import csv
import json

from .common import ROOT


def report(final=False):
    from .finish import table
    from .repair_review import main as review
    review()
    def load(name, default):
        path = ROOT / name
        return json.loads(path.read_text()) if path.exists() else default
    qualifications = load('configs/repair_epoch_qualification.json', {})
    expansion = load('configs/b_repair_expansion.json', [])
    # Recover the failed model's terminal state from its preserved full regression.
    for model in ('ouro', 'huginn'):
        last = next((item for item in expansion if item['stage']['model'] == model
                     and item['stage']['count'] == 200), None)
        if last and not last['result']['passed']:
            qualifications.setdefault('models', {}).setdefault(model, dict(
                passed=False, reason='Full200-request output/repeat qualification failed. See the expansion table.'))
    serving = load('configs/repair_epoch_serving_results.json', [])
    summary_path = ROOT / 'repair_epoch_w1_capacity.csv'
    measured = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
    text = '# B repair and separate execution epoch\n\n'
    text += 'Status: FINALIZED_WITH_LIMITATIONS.\n' if final else 'Status: IN_PROGRESS.\n'
    text += 'The original output failures remain unchanged. They are not accepted serving speedups.\n'
    text += 'This report separates later diagnostic changes and their qualification.\n'
    text += 'The original15-hour deadline and frozen SLOs still apply.\n'
    text += 'Checkpoint weights, BF16 tensor precision, and fixed recurrence remain unchanged.\n\n'
    text += '## Completed expansion stages\n\n'
    rows = []
    for item in expansion:
        stage, result = item['stage'], item['result']
        comparisons = result.get('comparisons', [])
        counts = [f"{value['exact']}/{value['N']}" for value in comparisons]
        rows.append([stage['model'], stage['count'], stage['cap'],
                     counts[0] if counts else 'NOT_MEASURED', counts[1] if len(counts) > 1 else 'NOT_MEASURED',
                     result['passed']])
    text += table(['Model', 'Requests', 'Output cap', 'Control/B exact', 'Control repeat exact', 'Stage pass'], rows)
    text += '\nA small passing stage is not a full repair acceptance. Counts measure generated sequences, not answer accuracy.\n\n'
    text += '## Model-level qualification\n\n'
    rows = []
    for model in ('ouro', 'huginn'):
        value = qualifications.get('models', {}).get(model, {})
        rows.append([model, qualifications.get('epoch', 'NOT_AVAILABLE'), value.get('passed', 'PENDING'),
                     f"{value.get('KV_exact', 'NOT_MEASURED')}/{value.get('KV_N', 'NOT_MEASURED')}",
                     value.get('full_depth_work_checks', 'NOT_MEASURED'), value.get('reason', '')])
    text += table(['Model', 'Epoch', 'Qualified', 'KV exact/N', 'Full-depth counter checks', 'Failure reason'], rows)
    text += '\nQualification requires200 original W1 requests, an identical-control repeat, stop parity, and fixed-depth work counters.\n'
    text += 'It also requires reversed-fragment prompt-KV and final-hidden byte checks with the same repair.\n'
    text += 'The source hash must match all full qualification runs.\n'
    text += 'A failed model does not disable the other model.\n\n'
    text += '## Reference and baseline-shift limitations\n\n'
    reference_rows = []
    for model in ('ouro', 'huginn'):
        value = qualifications.get('models', {}).get(model, {})
        for key, label in (('pre_repair_vs_post_repair_control', 'Original control vs canonical control'),
                           ('fresh_native_vs_saved_official_HF', 'Canonical Native vs saved official HF')):
            check = value.get(key)
            reference_rows.append([model, label, f"{check['exact']}/{check['N']}" if check else 'NOT_MEASURED'])
    text += table(['Model', 'Comparison', 'Exact generated sequences'], reference_rows)
    text += '\nCanonical geometry can change BF16 rounding and generated outputs relative to the original execution.\n'
    text += 'Internal B/control parity does not establish output parity with the pre-repair baseline or official HF.\n'
    text += 'The eight-prompt official-reference check is diagnostic only. It is not downstream accuracy.\n'
    text += 'No original-model-lossless performance claim follows from the repair qualification.\n\n'
    decisions = []
    for model in ('ouro', 'huginn'):
        value = qualifications.get('models', {}).get(model, {})
        for count in (2, 4, 8):
            actual = [row for row in measured if row['model'] == model and int(row['GPU_count']) == count]
            if count == 2 and (ROOT / 'state' / (model + '_repair_epoch_parity_failure.json')).exists():
                verdict = 'NO_GO_CORRECTNESS'
            elif count == 2 and (ROOT / 'state' / (model + '_repair_epoch_runtime_blocked.json')).exists():
                verdict = 'BLOCKED'
            elif count == 2 and value.get('passed') is False:
                verdict = 'NO_GO_CORRECTNESS'
            elif count != 2 or not actual:
                verdict = 'NOT_MEASURED_TIME_BUDGET' if final else 'PENDING'
            elif value.get('passed'):
                verdict = 'CONDITIONAL_FIDELITY'
            else:
                verdict = 'NO_GO_CORRECTNESS'
            decisions.append(dict(model=model, GPU_count=count, execution_epoch=qualifications.get('epoch', 'NOT_AVAILABLE'),
                                  verdict=verdict, internal_qualification=value.get('passed', 'PENDING'),
                                  measured_serving_runs=len(actual), original_model_fidelity='UNRESOLVED',
                                  sustainable_capacity='FINITE_TRACE_GRID_ONLY', loop_specific_novelty='NOT_ESTABLISHED'))
    with (ROOT / 'repair_epoch_model_decisions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    text += '## Separate-epoch decision scope\n\n'
    text += table(['Model', 'GPUs', 'Verdict', 'Internal qualification', 'Serving runs'],
                  [[row['model'], row['GPU_count'], row['verdict'], row['internal_qualification'], row['measured_serving_runs']]
                   for row in decisions])
    text += '\nA conditional verdict is not proof of a performance gain or original-model losslessness.\n'
    text += 'No new4/8GPU result is inferred from a two-GPU repair.\n\n'
    text += '## Actual separate-epoch serving\n\n'
    text += table(['Model', 'Arm', 'Offered req/s', 'Seed', 'Completed/planned', 'Joint SLO %', 'Window goodput req/s', 'Queue scope'],
                  [[row['model'], row['arm'], row['offered_rate'], row['seed'],
                    f"{row['completed']}/{row['planned_arrivals']}", row.get('joint_attainment_percent', 'NOT_MEASURED'),
                    row.get('qualified_requests_s_window', 'NOT_MEASURED'), row.get('queue_stability', 'NOT_MEASURED')]
                   for row in measured])
    if not measured:
        text += '\nNo separate-epoch serving measurement is available yet.\n'
    text += '\nThese rows never enter the original-epoch capacity table.\n'
    text += 'Latency and transfer CSVs identify the execution epoch for each run.\n'
    text += 'All three serving arms use the same canonical execution changes and physical GPU pair at each point.\n'
    text += 'The second independent seed uses a different GPU pair and a rotated arm order.\n'
    text += 'Rates and budgets are declared before the new epoch sees serving outcomes.\n'
    text += 'No SLO threshold changes. No pre-repair and post-repair samples are pooled.\n\n'
    text += 'Canonicalization can add GPU work. Its overhead belongs to every new-epoch comparator.\n'
    text += 'A gain over canonical controls is not automatically a gain over the unmodified optimized engine.\n'
    text += 'Pre/post E2E ratios are not equal-work speedups when generated outputs differ.\n\n'
    text += '## Serving output gate\n\n'
    text += table(['Model', 'Offered req/s', 'Seed', 'Control/B exact', 'All outputs exact'],
                  [[item['model'], item['rate'], item['seed'],
                    f"{item['B_control_output']['exact']}/{item['B_control_output']['N']}", item['outputs_exact']]
                   for item in serving if 'B_control_output' in item])
    text += '\nA new mismatch closes that model\'s new performance gate. It does not erase its recorded failure.\n'
    if measured:
        import numpy as np
        from .jobs import parity
        groups = {}
        by_name = {row['run']: row for row in measured}
        for item in serving:
            if not item.get('names') or not item.get('outputs_exact'):
                continue
            for baseline in ('PD_G1', 'DP_Mixed'):
                before = by_name.get(item['names'][baseline])
                after = by_name.get(item['names']['B'])
                if before is None or after is None or before['status'] != 'COMPLETE' or after['status'] != 'COMPLETE':
                    continue
                check = parity(before['run'], after['run'])
                exact = check['N'] == item['count'] and check['exact'] == item['count'] and check['complete_ids']
                groups.setdefault((item['model'], item['rate'], baseline), []).append((before, after, exact))
        comparisons = []
        for (model, rate, baseline), pairs in sorted(groups.items()):
            a = np.array([float(x['qualified_requests']) for x, y, exact in pairs])
            b = np.array([float(y['qualified_requests']) for x, y, exact in pairs])
            ratio = float(b.sum() / a.sum()) if a.sum() else 'UNDEFINED_ZERO_BASELINE'
            interval = 'NOT_MEASURED_SINGLE_START'
            if len(pairs) >= 2:
                indices = np.random.default_rng(20261030).integers(0, len(pairs), (3000, len(pairs)))
                denominator = a[indices].sum(axis=1)
                if np.all(denominator > 0):
                    interval = np.percentile(b[indices].sum(axis=1) / denominator, [2.5, 97.5]).tolist()
                else:
                    interval = 'UNBOUNDED_ZERO_BASELINE_RESAMPLES'
            comparisons.append(dict(model=model, offered_rate=rate, baseline=baseline, candidate='B',
                                    independent_starts=len(pairs), baseline_qualified=int(a.sum()),
                                    candidate_qualified=int(b.sum()), observed_qualified_goodput_ratio=ratio,
                                    ci95=interval, all_outputs_exact=all(exact for x, y, exact in pairs),
                                    accepted_original_model_lossless_speedup=False,
                                    evidence='MEASURED_FINITE_TRACE_PAIRED_START_CLUSTER_BOOTSTRAP',
                                    limitation='Very few starts. Not a capacity-ratio or selection-adjusted interval.'))
        if comparisons:
            with (ROOT / 'repair_epoch_paired_goodput.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
        text += '\n## Observed paired goodput, not an accepted lossless speedup\n\n'
        text += table(['Model', 'Offered req/s', 'Baseline', 'Independent starts', 'Goodput ratio', '95% CI', 'Exact outputs'],
                      [[row['model'], row['offered_rate'], row['baseline'], row['independent_starts'],
                        f"{row['observed_qualified_goodput_ratio']:.3f}×" if isinstance(row['observed_qualified_goodput_ratio'], float)
                        else row['observed_qualified_goodput_ratio'], row['ci95'], row['all_outputs_exact']]
                       for row in comparisons])
        text += '\nThe bootstrap resamples whole paired starts. It does not treat tokens as independent requests.\n'
    text += 'Finite rate grids do not establish an unrestricted sustainable maximum.\n'
    text += 'Between-start uncertainty needs multiple completed starts. A single point has no such confidence interval.\n'
    text += 'No original-model-lossless GO is established. Official-HF fidelity remains unresolved.\n'
    text += 'Generic immutable-layer streaming is not independently measured. Loop-exclusive novelty remains unresolved.\n'
    text += 'Read `b_repair_diagnosis.md`, `b_repair_ablation_summary.csv`, and the per-run source snapshots.\n'
    (ROOT / 'repair_epoch_report.md').write_text(text)
    from .repair_figures import main as figures
    figures()


if __name__ == '__main__':
    report()
