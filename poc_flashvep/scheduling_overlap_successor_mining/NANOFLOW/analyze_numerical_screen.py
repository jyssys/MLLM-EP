"""CPU comparison of native same-prefix logits, not performance evidence."""
import argparse
import csv
import json
import os
from pathlib import Path

import torch


def compare(left, right, name):
    a, b = sorted(left.glob('step*.pt')), sorted(right.glob('step*.pt'))
    assert len(a) == len(b) and a, (left, right, len(a), len(b))
    rows = []
    for pa, pb in zip(a, b):
        x = torch.load(pa, map_location='cpu', weights_only=False)
        y = torch.load(pb, map_location='cpu', weights_only=False)
        assert x['step'] == y['step'] and x['request_indices'] == y['request_indices']
        u, v = x['logits'].float(), y['logits'].float()
        assert u.shape == v.shape and torch.isfinite(u).all() and torch.isfinite(v).all()
        gap = u.topk(2, dim=-1).values.diff(dim=-1).abs().flatten()
        error = (u-v).abs()
        kl = (u.softmax(-1) * (u.log_softmax(-1)-v.log_softmax(-1))).sum(-1)
        for i, rid in enumerate(x['request_indices']):
            rows.append(dict(comparison=name, step=x['step'], request_index=rid,
                             left_argmax=int(u[i].argmax()), right_argmax=int(v[i].argmax()),
                             exact_logits=bool(torch.equal(u[i], v[i])),
                             argmax_match=bool(u[i].argmax() == v[i].argmax()),
                             baseline_top2_gap=float(gap[i]),
                             max_logit_abs_error=float(error[i].max()),
                             rms_logit_error=float(error[i].square().mean().sqrt()),
                             kl=float(kl[i])))
    return rows


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(4)
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    args = p.parse_args()
    manifest = json.loads((args.root/'manifest.json').read_text())
    baseline, treatment = manifest.get('variants', ['graph', 'plan2_graph'])
    pairs = [('graph_restart' if baseline == 'graph' else 'baseline_restart', f'b0_{baseline}', f'b1_{baseline}'),
             ('split_restart', f'b0_{treatment}', f'b1_{treatment}'),
             ('cross_plan_b0', f'b0_{baseline}', f'b0_{treatment}'),
             ('cross_plan_b1', f'b1_{baseline}', f'b1_{treatment}')]
    rows, summary = [], {}
    for name, a, b in pairs:
        measured = compare(args.root/(a+'_logits'), args.root/(b+'_logits'), name)
        rows.extend(measured)
        n = len(measured)
        summary[name] = {'token_observations': n,
                         'argmax_agreement': sum(r['argmax_match'] for r in measured)/n,
                         'exact_logit_fraction': sum(r['exact_logits'] for r in measured)/n,
                         'max_abs_error': max(r['max_logit_abs_error'] for r in measured),
                         'mean_kl': sum(r['kl'] for r in measured)/n,
                         'mismatches': [r for r in measured if not r['argmax_match']]}
    with (args.root/'same_prefix_logits.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'scope': 'NATIVE_SAME_PREFIX_CORRECTNESS_DIAGNOSTIC_NOT_PERFORMANCE',
              'comparisons': summary, 'performance_eligible': False}
    reference = {r['request_id']: r for r in json.loads(Path(manifest['reference']).read_text())['requests']}
    historical = {}
    for stem in (f'b{b}_{v}' for b in range(2) for v in (baseline, treatment)):
        sampled = json.loads((args.root/(stem+'.json')).read_text())['requests']
        n = good = exact = 0
        for row in sampled:
            target = reference[row['request_id']]
            assert row['input_ids'] == target['input_ids']
            assert len(row['output_ids']) == len(target['output_ids'])
            matches = [a == b for a, b in zip(row['output_ids'], target['output_ids'])]
            good += sum(matches)
            n += len(matches)
            exact += int(all(matches))
        historical[stem] = {'token_agreement': good/n, 'exact_requests': exact,
                            'requests': len(sampled), 'note': 'teacher_forced_history_not_free_generation'}
    result['historical_reference_agreement'] = historical
    (args.root/'same_prefix_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    main()
