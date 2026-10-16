"""Write a transparent scorecard from completed frozen cohorts."""
import csv
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def pct(value):
    return f'{100 * value:.2f}%'


def main():
    assert read(ROOT/'results/quality_analysis_wait_status.json')['success']
    summary = read(ROOT/'results/quality_score/native_quality_r4.json')
    analysis = read(ROOT/'results/quality_cap_analysis.json')
    with (ROOT/'results/quality_score/quality_generations.csv').open() as stream:
        records = list(csv.DictReader(stream))
    lines = ['# Public quality scorecard', '',
        '`REFERENCE_PROTOCOL_NOT_IDENTICAL` applies to every score below.',
        'The public grader differs from the authors\' in-house harness and LLM judge.',
        'The checkpoint, P1 prompt, R4, BF16, and seeds remain frozen.',
        'The primary sampled policy uses temperature 1.0, top-p 0.7, and no top-k limit.',
        'Greedy results use temperature 0 and are a separate baseline.', '',
        '## Complete cohorts', '',
        '| Dataset / policy | Cap | Completed correct / all | Problem-bootstrap 95% CI | EOS / all | Capped | Parsed / all |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(summary['all_caps'], key=lambda r: (r['dataset'], r['lane'], r['cap'])):
        assert row['collection_complete']
        low, high = row['problem_bootstrap_95CI']
        lines.append(f"| {row['dataset']} / {row['lane']} | {row['cap']:,} | "
            f"{row['correct']}/{row['total']} = {pct(row['score'])} | {pct(low)}–{pct(high)} | "
            f"{row['natural_complete']}/{row['total']} | {row['capped']} | {row['parsed']}/{row['total']} |")
    lines += ['', 'All outputs remain in the denominator, including length stops.',
        'A matching extracted answer inside a length-stopped output does not count as completed correct.',
        'The bootstrap uses 5,000 problem-cluster resamples.',
        'All seeds for one sampled AIME problem remain together.',
        'There are 30 independent AIME problems, not 90 independent problems.',
        'MATH100 is the frozen 100-problem subset, not the full MATH-500 benchmark.', '',
        '## Final declared caps and lengths', '',
        '| Dataset / policy | Final cap | Generated tokens P50 / P90 / P95 / max | Mean repeated 16-gram fraction |',
        '|---|---:|---:|---:|']
    for name, row in sorted(summary['final_lanes'].items()):
        detail = next(r for r in analysis['summaries'] if
            (r['dataset'], r['lane'], r['cap']) == (row['dataset'], row['lane'], row['cap']))
        lengths = ' / '.join(f'{n:,.1f}' for n in detail['length_p50_p90_p95_max'])
        lines.append(f"| {row['dataset']} / {row['lane']} | {row['cap']:,} | {lengths} | "
                     f"{row['mean_repeated_16gram_fraction']:.5f} |")
    lines += ['', '## Seed-wise sampled AIME result', '',
        '| Seed | Completed correct / all |', '|---|---:|']
    primary = summary['final_lanes']['aime24_sampling']
    for seed, value in primary['by_seed'].items():
        lines.append(f"| {seed} | {value['correct']}/{value['total']} |")
    lines += ['', 'Pass@10: `NOT_RUN`.',
        'Three seeds cannot estimate the authors\' ten-sample pass@10 protocol.', '',
        '## Cap and request-history sensitivity', '',
        '| Dataset / policy | Cap transition | Changed common prefix / all | Paired completed-correct difference, pp | Problem-bootstrap 95% CI, pp |',
        '|---|---:|---:|---:|---:|']
    for row in analysis['paired_cap_changes']:
        delta = row['completed_correct_paired_delta']
        low, high = delta['CI95']
        lines.append(f"| {row['dataset']} / {row['lane']} | {row['lower_cap']:,} → {row['higher_cap']:,} | "
            f"{row['common_prefix_changed']}/{row['total']} | {100*delta['mean']:+.2f} | "
            f"{100*low:+.2f}–{100*high:+.2f} |")
    lines += ['', 'The cap comparison also changes the engine\'s preceding request history.',
        'Changed common prefixes prevent a censoring-only explanation of score differences.',
        'Fixed per-request seeds do not remove this limitation.',
        'The separate fixed-history diagnostic does not replace any public benchmark output.', '',
        '## Author comparison', '',
        '| Property | Author reference | This public evaluation |', '|---|---|---|',
        '| AIME2024 R4 pass@1 | 64.7% | See the sampled final-cap row above |',
        '| AIME2024 R4 pass@10 | 90.0% | NOT_RUN: three seeds only |',
        '| Sampling | Temperature 1.0, top-p 0.7 | Temperature 1.0, top-p 0.7, explicit unlimited top-k |',
        '| Harness and grading | Private harness and LLM judge | Frozen public integer exact match |',
        '| Exact AIME cap, prompt, seed stream | Not fully disclosed | Frozen public protocol and raw token IDs |', '',
        'The source is [Ouro Table 9 and Appendix C](https://arxiv.org/html/2510.25741v2).',
        'The paper\'s separate 8,192-token safety experiment does not establish its AIME cap.',
        'A confidence interval that includes 64.7% does not establish protocol equivalence.',
        'The campaign does not select a protocol by proximity to the published score.', '',
        '## Grading and stop checks', '']
    ledger = []
    for row in records:
        if row['complete'] == 'True' and row['parsed'] != 'True':
            ledger.append({k: row[k] for k in ['dataset', 'lane', 'cap', 'problem_id', 'seed',
                'extracted_answer', 'parse_rule', 'reference', 'answer_matches', 'correct_completed']})
    for row in analysis['summaries']:
        assert all(row[k] == 0 for k in ['stop_without_terminal_EOS', 'EOS_before_last_position',
            'length_stop_with_EOS', 'over_effective_cap', 'over_total_context_limit'])
    lines += ['Every observed natural stop ends with EOS ID2.',
        'No length-stopped output contains EOS ID2.',
        'No recorded generation exceeds its effective cap or the 32,768-position limit.', '',
        'The unchanged grader records completed but unparsed outputs separately.',
        'The ledger includes invalid answer formats and out-of-range AIME integers.',
        'It does not automatically classify every unparsed answer as a grader defect.',
        'No manual grade, EOS rule, or benchmark answer changes.', '',
        'Operational quality, reference fidelity, and systems authorization remain separate decisions.',
        'See `results/quality_decision.json` for the operational decision.', '',
        'Evidence: `results/quality_score/quality_generations.csv`, `results/quality_cap_analysis.json`, and the raw-output archive.', '']
    target = ROOT/'quality_scorecard.md'
    assert not target.exists(), 'Preserve an earlier scorecard.'
    target.write_text('\n'.join(lines))
    write(ROOT/'results/completed_unparsed_answer_ledger.json', dict(records=ledger,
        grades_changed=False, scope='Frozen-parser diagnostic. No post-hoc answer normalization.'))


if __name__ == '__main__':
    main()
