"""Score unchanged Natural-EOS output with the pinned campaign task scorers."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from statistics import mean

from ..mllm_epd_prefill_heavy_token_split import quality_audit
from .online_summary import digest, read, rows


def score(manifest, runs, scorer_root, label):
    quality_audit.ROOT = scorer_root
    scoring = quality_audit.scorers()
    task_rows = {row['request_id']: row for row in read(manifest)['requests']}
    runs_scored = {}
    for run in runs:
        evidence = run / 'raw' / f'{label}_requests.jsonl'
        contract = read(run / 'configs' / f'{label}.json')
        if contract['output_count'] is not None:
            raise ValueError('Task quality requires unmodified Natural EOS outputs')
        section = {}
        for observation in rows(evidence):
            key = observation['request_id']
            original = task_rows[key]
            if observation['prompt_SHA256'] != original['prompt_SHA256'] or key in section:
                raise ValueError(f'Changed prompt or duplicate {key}')
            result = {'score': None, 'complete_task_scorer': False, 'official_full_pipeline': 'NOT_RUN'}
            if observation['status'] == 'MEASURED_ONLINE':
                with redirect_stdout(io.StringIO()):
                    result = quality_audit.score_request(original, observation['text'], scoring)
            section[key] = {'request_id': key, 'task': original['source'], 'status': observation['status'], **result}
        if section.keys() != task_rows.keys():
            raise ValueError('Comparison must cover the complete frozen natural roster')
        runs_scored[run.name] = {'requests': section, 'raw_SHA256': digest(evidence)}
    baseline = runs_scored[runs[0].name]['requests']
    for block in runs_scored.values():
        groups = defaultdict(list)
        for row in block['requests'].values():
            groups[row['task']].append(row)
        block['tasks'] = {}
        for task, group in sorted(groups.items()):
            valid = [row['score'] for row in group if row['score'] is not None]
            paired = [(baseline[row['request_id']]['score'], row['score']) for row in group
                      if baseline[row['request_id']]['score'] is not None and row['score'] is not None]
            block['tasks'][task] = {
                'scheduled': len(group), 'scored': len(valid), 'mean': mean(valid) if valid else None,
                'new_zero_after_baseline_nonzero': sum(left > 0 and right == 0 for left, right in paired),
                'new_positive_after_baseline_zero': sum(left == 0 and right > 0 for left, right in paired),
                'paired_score_change_percentage_points': 100 * mean(right - left for left, right in paired) if paired else None,
                'complete_task_scorer': all(row['complete_task_scorer'] for row in group),
            }
    return {'source_receipts': scoring[3], 'reference': runs[0].name, 'runs': runs_scored,
            'scope': 'Task scores on frozen raw Natural-EOS output. MMLongBench-Doc full pipeline requires external extraction.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--label', default='measured_1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scorer-root', type=Path, default=Path('poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124'))
    args = parser.parse_args()
    result = score(args.manifest, args.runs, args.scorer_root, args.label)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
