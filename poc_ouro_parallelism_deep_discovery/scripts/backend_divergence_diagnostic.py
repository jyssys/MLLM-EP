"""Describe raw native/reference output divergence without attributing a cause."""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path
import statistics

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha, write
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval


def token_relation(left, right):
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    relation = 'EXACT' if left == right else 'PREFIX_ONLY' if common == min(len(left), len(right)) else 'INTERNAL_DIVERGENCE'
    return dict(relation=relation, common_prefix_tokens=common,
                first_divergent_token=None if relation != 'INTERNAL_DIVERGENCE' else common,
                first_output_agrees=bool(left and right and left[0] == right[0]))


def load_verified(relative, manifest):
    if sha(ROOT/relative) != manifest[relative]:
        raise ValueError('Changed admitted raw output: '+relative)
    with gzip.open(ROOT/relative, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def terminal_flags(left, right, eos):
    relation = token_relation(left, right)
    index = relation['first_divergent_token']
    return dict(first_difference_touches_EOS=bool(index is not None and eos in (left[index], right[index])),
                first_difference_at_shorter_terminal=bool(index is not None and index == min(len(left), len(right))-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    native_path = ROOT/'analysis/native_quality_extension_complete_v1/summary.json'
    staged_path = ROOT/'analysis/baseline_extension_complete_v1/summary.json'
    integrity_path = ROOT/'artifacts/native_quality_integrity_complete_v1.json'
    native, staged, integrity = read(native_path), read(staged_path), read(integrity_path)
    for report in [native, staged, integrity]:
        for relative, digest in report['input_sha256'].items():
            if sha(ROOT/relative) != digest:
                raise ValueError('Changed admitted input: '+relative)
    old = {(row['size'], row['id']): row for row in staged['records'] if row['mode'] == 'fixed'}
    records, sources = [], {}
    for row in native['records']:
        prior = old[row['size'], row['id']]
        current = load_verified(row['source'], integrity['input_sha256'])
        previous = load_verified(prior['source'], staged['input_sha256'])
        if current['prompt_ids'] != previous['prompt_ids']:
            raise ValueError('Raw prompt token IDs differ.')
        relation = token_relation(previous['output_ids'], current['output_ids'])
        if relation['common_prefix_tokens'] != row['common_prefix_tokens']:
            raise ValueError('Existing raw comparison differs.')
        records.append(dict(size=row['size'], id=row['id'], dataset=row['dataset'], **relation,
            **terminal_flags(previous['output_ids'], current['output_ids'], current['EOS']),
            native_tokens=len(current['output_ids']), staged_tokens=len(previous['output_ids']),
            native_finish=current['finish_reason'], staged_finish=previous['finish_reason'],
            native_correct=row['correct_completed'], staged_correct=prior['correct_completed'],
            native_api_text_equals_canonical=current['native_api_text'] == current['text']))
        for relative in [row['source'], prior['source']]:
            sources[relative] = sha(ROOT/relative)
    summaries = []
    for size, domain in sorted({(r['size'], r['dataset']) for r in records}):
        selected = [r for r in records if (r['size'], r['dataset']) == (size, domain)]
        if len(selected) != 32 or len({r['id'] for r in selected}) != 32:
            raise ValueError('Require32 unique requests per cell.')
        categories = collections.Counter(r['relation'] for r in selected)
        prefix = [r['common_prefix_tokens'] for r in selected if r['relation'] == 'INTERNAL_DIVERGENCE']
        internal = [int(r['relation'] == 'INTERNAL_DIVERGENCE') for r in selected]
        summaries.append(dict(size=size, dataset=domain, N=32, categories=dict(categories),
            internal_divergence_percent=100*statistics.mean(internal),
            internal_divergence_CI_percent=[100*x for x in interval(internal)],
            internal_divergence_common_prefix_median=None if not prefix else statistics.median(prefix),
            first_output_agrees=sum(r['first_output_agrees'] for r in selected),
            internal_difference_before_token32=sum(r['relation'] == 'INTERNAL_DIVERGENCE' and
                r['common_prefix_tokens'] < 32 for r in selected),
            first_difference_touches_EOS=sum(r['first_difference_touches_EOS'] for r in selected),
            first_difference_at_shorter_terminal=sum(r['first_difference_at_shorter_terminal'] for r in selected),
            native_api_text_equals_canonical=sum(r['native_api_text_equals_canonical'] for r in selected),
            equal_correctness_bits=sum(r['native_correct'] == r['staged_correct'] for r in selected)))
    args.output.mkdir(parents=True)
    with (args.output/'requests.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    sources.update({str(path.relative_to(ROOT)): sha(path) for path in [native_path, staged_path, integrity_path]})
    write(args.output/'summary.json', dict(summaries=summaries, records=records, input_sha256=sources,
        evidence_class='DESCRIPTIVE_RAW_OUTPUT_DIAGNOSTIC', model_or_GPU_execution='NOT_RUN',
        candidate_result='NOT_RUN', index_rule='Zero-based first unequal token. Prefix-only length differences are separate.',
        limitation='Internal means unequal tokens within both stored arrays, which can include an EOS at termination. Terminal flags remain explicit. No first unequal operator, numerical cause, independent quality equivalence, or candidate speedup follows. Different stop implementations remain a confound.'))
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
