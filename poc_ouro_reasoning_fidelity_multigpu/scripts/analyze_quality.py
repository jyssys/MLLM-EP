"""Derive cap sensitivity and plots only after complete frozen quality collection."""
import csv
import os
import subprocess
from collections import Counter, defaultdict
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def key(row):
    return row['dataset'], row['lane'], row['problem_id'], row['seed']


def problem_ci(values, seed=41015):
    grouped = defaultdict(list)
    for problem, value in values:
        grouped[problem].append(value)
    means = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), (5000, len(means)))].mean(axis=1)
    return dict(mean=float(means.mean()), CI95=list(map(float, np.quantile(draws, [.025, .975]))),
                independent_problems=len(means), resamples=5000,
                unit='Problem cluster. Preserve all observed seeds within each problem.')


def main():
    assert read(ROOT/'results/quality_collection_and_scoring_complete.json')['success']
    source = ROOT/'results/quality_v1'
    score = ROOT/'results/quality_score'
    with (score/'quality_generations.csv').open() as stream:
        graded = list(csv.DictReader(stream))
    correctness = {(r['dataset'], r['lane'], r['problem_id'], int(r['seed']), int(r['cap'])):
                   r['correct_completed'] == 'True' for r in graded}
    raw = []
    for path in sorted(source.glob('*_cap*_wave*.json')):
        lane = path.name.split('_')[1]
        for row in read(path):
            row['lane'] = lane
            raw.append(row)
    grouped = defaultdict(list)
    for row in raw:
        grouped[row['dataset'], row['lane'], row['cap']].append(row)
    summaries = []
    for (dataset, lane, cap), rows in sorted(grouped.items()):
        expected = (90 if lane == 'sampling' else 30) if dataset == 'aime24' else 100
        assert len(rows) == expected, (dataset, lane, cap, len(rows))
        lengths = np.array([r['generated_tokens'] for r in rows])
        summaries.append(dict(dataset=dataset, lane=lane, cap=cap, total=len(rows),
            finish_reasons=dict(Counter(r['finish_reason'] for r in rows)),
            stop_reasons=dict(Counter(str(r['stop_reason']) for r in rows)),
            stop_with_terminal_EOS=sum(r['finish_reason']=='stop' and r['generated_ids'][-1:]==[2] for r in rows),
            stop_without_terminal_EOS=sum(r['finish_reason']=='stop' and r['generated_ids'][-1:]!=[2] for r in rows),
            EOS_before_last_position=sum(2 in r['generated_ids'][:-1] for r in rows),
            length_stop_with_EOS=sum(r['finish_reason']=='length' and 2 in r['generated_ids'] for r in rows),
            over_effective_cap=sum(len(r['generated_ids'])>r['effective_cap'] for r in rows),
            over_total_context_limit=sum(r['total_positions']>32768 for r in rows),
            closing_think=sum('</think>' in r['text'] for r in rows),
            length_p50_p90_p95_max=list(map(float, np.quantile(lengths, [.5, .9, .95, 1]))),
            exact_frozen_prompt_ids=True,
            correct_completed=sum(correctness[(*key(r), cap)] for r in rows)))
    pairs = []
    cap_deltas = []
    for dataset, lane in sorted({(r['dataset'], r['lane']) for r in raw}):
        caps = sorted(c for d, l, c in grouped if (d, l) == (dataset, lane))
        for low, high in zip(caps, caps[1:]):
            left = {key(r): r for r in grouped[dataset, lane, low]}
            right = {key(r): r for r in grouped[dataset, lane, high]}
            assert left.keys() == right.keys()
            deltas = []
            comparison = []
            for sample in left:
                a, b = left[sample], right[sample]
                x, y = a['generated_ids'], b['generated_ids']
                first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
                delta = int(correctness[(*sample, high)]) - int(correctness[(*sample, low)])
                deltas.append((a['problem_id'], delta))
                item = dict(dataset=dataset, lane=lane, problem_id=a['problem_id'], seed=a['seed'],
                    lower_cap=low, higher_cap=high, lower_length=len(x), higher_length=len(y),
                    common_prefix_equal=first is None, first_divergent_token_index=first,
                    exact_outputs_equal=x == y, lower_finish=a['finish_reason'], higher_finish=b['finish_reason'],
                    completed_correct_delta=delta)
                pairs.append(item)
                comparison.append(item)
            cap_deltas.append(dict(dataset=dataset, lane=lane, lower_cap=low, higher_cap=high,
                common_prefix_changed=sum(not r['common_prefix_equal'] for r in comparison),
                total=len(comparison), completed_correct_paired_delta=problem_ci(deltas),
                interpretation='Cap and preceding engine history differ. Changed trajectories prevent a pure censoring-only interpretation.'))
    target = ROOT/'results/quality_cap_analysis.json'
    assert not target.exists()
    write(target, dict(summaries=summaries, paired_cap_changes=cap_deltas,
        source_sha256=sha(score/'native_quality_r4.json'), pass_at_10='NOT_RUN: only three sampled seeds.',
        independent_unit='Problem, not token, seed, or engine wave.'))
    write(ROOT/'results/quality_cap_prefix_pairs.json', pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'sampling':'tab:blue', 'greedy':'tab:orange'}
    figures=ROOT/'figures'
    fig, axes=plt.subplots(1, 3, figsize=(13, 4))
    for dataset, lane in sorted({(r['dataset'], r['lane']) for r in summaries}):
        selected=[r for r in summaries if (r['dataset'],r['lane'])==(dataset,lane)]
        label=f'{dataset} {lane}'
        caps=[r['cap']/1024 for r in selected]
        axes[0].plot(caps,[100*r['correct_completed']/r['total'] for r in selected],marker='o',label=label)
        axes[1].plot(caps,[100*r['finish_reasons'].get('stop',0)/r['total'] for r in selected],marker='o',label=label)
        axes[2].plot(caps,[r['length_p50_p90_p95_max'][0]/1024 for r in selected],marker='o',label=label)
    for axis, ylabel in zip(axes,['Completed-correct / all, %','Natural termination / all, %','Generated tokens P50, Ki tokens']):
        axis.set_xlabel('Requested cap, Ki tokens')
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle('Public exact-match evaluation. REFERENCE_PROTOCOL_NOT_IDENTICAL')
    fig.tight_layout()
    fig.savefig(figures/'cap_completion_quality.png',dpi=150)
    plt.close(fig)
    latest=read(score/'native_quality_r4.json')['final_lanes']
    fig, axes=plt.subplots(1,3,figsize=(13,4))
    for axis,(name,final) in zip(axes,sorted(latest.items())):
        selected=grouped[final['dataset'],final['lane'],final['cap']]
        complete=[r['generated_tokens'] for r in selected if r['finish_reason']=='stop']
        capped=[r['generated_tokens'] for r in selected if r['finish_reason']!='stop']
        axis.hist([complete,capped],bins=16,stacked=True,label=['EOS','Capped'],color=['tab:green','tab:red'])
        axis.set_title(name)
        axis.set_xlabel('Generated tokens')
        axis.set_ylabel('Outputs')
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures/'natural_output_lengths.png',dpi=150)
    plt.close(fig)
    for version in ['4.54.1','4.57.1']:
        subprocess.run([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.audit_output_decoding',
                        '--version',version,'--full-output'],cwd=PROJECT,
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


if __name__ == '__main__':
    main()
