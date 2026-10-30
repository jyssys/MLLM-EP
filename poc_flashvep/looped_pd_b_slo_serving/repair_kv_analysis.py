"""Compare actual causal KV prefixes captured before the first divergent token."""
import argparse
import csv

from .common import ROOT, write
from .jobs import output_rows


def load(name):
    import torch
    values = []
    for path in (ROOT / 'raw' / name).glob('actual_operands_*.pt'):
        values.extend(torch.load(path, map_location='cpu', weights_only=True).get('KV_prefixes', []))
    if len(values) != 1:
        raise ValueError(f'Expected one causal-prefix capture for {name}, received {len(values)}')
    return values[0]


def compare(left, right, result_name):
    import torch
    a, b = load(left), load(right)
    if (a['depth'], a['position'], a['prompt_length']) != (b['depth'], b['position'], b['prompt_length']):
        raise ValueError('Captured KV addresses differ')
    x = next(row for row in output_rows(left) if row['sample_id'] == 3)
    y = next(row for row in output_rows(right) if row['sample_id'] == 3)
    rows = []
    for kind in ('key', 'value'):
        for layer in range(len(a[kind])):
            av, bv = a[kind][layer], b[kind][layer]
            for region, start, end in (('prompt', 0, a['prompt_length']),
                                       ('decode_prefix', a['prompt_length'], a['position'])):
                aa, bb = av[start:end], bv[start:end]
                delta = aa.float() - bb.float()
                rows.append(dict(kind=kind, physical_layer=layer, region=region, positions=end-start,
                                 exact=bool(torch.equal(aa, bb)), different_elements=int(torch.count_nonzero(delta)),
                                 max_absolute=float(delta.abs().max()) if delta.numel() else 0.,
                                 relative_L2=float(delta.norm() / aa.float().norm().clamp_min(1e-30))))
    summary = dict(left=left, right=right, depth=a['depth'], current_position_excluded=a['position'],
                   generated_history_through_index124_equal=x['output_ids'][:125] == y['output_ids'][:125],
                   prompt_tokens_equal=x['input_ids'] == y['input_ids'],
                   entire_prefix_exact=all(row['exact'] for row in rows),
                   prompt_prefix_exact=all(row['exact'] for row in rows if row['region'] == 'prompt'),
                   prior_decode_prefix_exact=all(row['exact'] for row in rows if row['region'] == 'decode_prefix'),
                   total_different_elements=sum(row['different_elements'] for row in rows),
                   evidence='MEASURED_ACTUAL_CAUSAL_KV_TENSORS_NOT_A_LATENCY_MEASUREMENT')
    write(ROOT / 'configs' / (result_name + '.json'), summary)
    with (ROOT / (result_name + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--result-name', required=True)
    args = parser.parse_args()
    compare(args.left, args.right, args.result_name)


if __name__ == '__main__':
    main()
