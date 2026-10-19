"""Compare frozen natural calibration baselines. This is not candidate quality."""
import argparse
import statistics

from poc_ouro_parallelism_deep_discovery.contract import ROOT, MODELS, read, sha, write
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scored', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--comparison-tp', type=int, choices=[2, 4], default=4)
    args = parser.parse_args()
    source, output = ROOT / args.scored, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    records = read(source)['records']
    comparisons = []
    for size in MODELS:
        for domain in ['gsm8k', 'math500']:
            arms = []
            target_tp = args.comparison_tp
            for tp, runtime in [(1, 'native_vllm_async'),
                                (target_tp, f'native_vllm_async_tp{target_tp}baseline')]:
                selected = [r for r in records if (r['size'], r['dataset'], r['mode'], r['TP'], r['runtime'])
                            == (size, domain, 'fixed', tp, runtime)]
                assert len(selected) == 8 and len({r['id'] for r in selected}) == 8
                arms.append({r['id']: r for r in selected})
            left, right = arms
            assert set(left) == set(right)
            differences, details = [], []
            for key in sorted(left):
                a, b = read(ROOT / left[key]['source']), read(ROOT / right[key]['source'])
                for field in ['prompt_ids', 'revision', 'cap', 'temperature', 'source_protocol_sha256']:
                    assert a[field] == b[field], (key, field)
                x, y = a['output_ids'], b['output_ids']
                first = next((i for i, (v, w) in enumerate(zip(x, y)) if v != w), None)
                if first is None and len(x) != len(y):
                    first = min(len(x), len(y))
                differences.append(int(right[key]['correct_completed']) - int(left[key]['correct_completed']))
                details.append(dict(id=key, TP1_source=left[key]['source'], **{f'TP{target_tp}_source': right[key]['source']},
                    raw_token_ids_equal=x == y, first_divergence=first,
                    TP1_correct=left[key]['correct_completed'], **{f'TP{target_tp}_correct': right[key]['correct_completed']},
                    TP1_tokens=len(x), **{f'TP{target_tp}_tokens': len(y)}, TP1_finish=left[key]['finish_reason'],
                    **{f'TP{target_tp}_finish': right[key]['finish_reason']}, TP1_wall_ms=a['wall_ms'],
                    **{f'TP{target_tp}_wall_ms': b['wall_ms']}))
            comparisons.append(dict(size=size, dataset=domain, N=8,
                TP1_correct=sum(left[k]['correct_completed'] for k in left),
                **{f'TP{target_tp}_correct': sum(right[k]['correct_completed'] for k in right)},
                TP1_truncations=sum(left[k]['truncated'] for k in left),
                **{f'TP{target_tp}_truncations': sum(right[k]['truncated'] for k in right)},
                **{f'TP{target_tp}_minus_TP1_accuracy_pp': 100 * statistics.mean(differences)},
                paired_request_bootstrap_95_CI_pp=[100 * v for v in interval(differences)],
                bootstrap_repeats=4000, raw_token_parity=sum(d['raw_token_ids_equal'] for d in details),
                details=details))
    write(output, dict(comparisons=comparisons, evidence_class='QUALITY_ONLY',
        scored_source=str(source.relative_to(ROOT)), scored_sha256=sha(source),
        candidate='NONE', held_out='UNTOUCHED',
        scope='Native fixed-R4 calibration baselines. Eight requests/domain. No official reference-score reproduction.',
        caution='Small calibration N cannot establish population quality equivalence. Different output lengths invalidate unqualified natural speedup ratios.'))
    print([(r['size'], r['dataset'], r['TP1_correct'], r[f'TP{args.comparison_tp}_correct'], r['raw_token_parity'])
           for r in comparisons])


if __name__ == '__main__':
    main()
