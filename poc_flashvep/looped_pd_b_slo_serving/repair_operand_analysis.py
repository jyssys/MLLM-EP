"""Compare captured real request operands. No latency or accuracy inference."""
import argparse
import csv
import json

from .common import ROOT, write


def load(name):
    import torch
    values = {}
    for path in (ROOT / 'raw' / name).glob('actual_operands_*.pt'):
        data = torch.load(path, map_location='cpu', weights_only=True)
        role = 'decode' if any(row['logits'] is not None for row in data['records']) else 'prefill'
        for row in data['records']:
            key = (role, row['step'], row['operation'], row['depth'], row['loop_count'])
            if key in values:
                raise ValueError(f'Ambiguous repeated operand key: {name} {key}')
            values[key] = row
    return values


def compare(left, right, result_name):
    import torch
    a, b = load(left), load(right)
    rows = []
    for key in sorted(a.keys() & b.keys()):
        for tensor_name in ('input_hidden', 'hidden', 'logits'):
            x, y = a[key][tensor_name], b[key][tensor_name]
            if x is None or y is None:
                continue
            delta = x.float() - y.float()
            row = dict(role=key[0], step=key[1], operation=key[2], depth=key[3], loops=key[4],
                       operand=tensor_name, exact=bool(torch.equal(x, y)),
                       maximum_absolute=float(delta.abs().max()),
                       relative_L2=float(delta.norm() / x.float().norm().clamp_min(1e-30)),
                       left_batch_rows=a[key]['rows'], right_batch_rows=b[key]['rows'],
                       left_row=a[key]['row'], right_row=b[key]['row'])
            if tensor_name == 'logits':
                tx, ix = x.float().topk(2)
                ty, iy = y.float().topk(2)
                row.update(left_top1=int(x.argmax()), right_top1=int(y.argmax()),
                           left_margin=float(tx[0] - tx[1]), right_margin=float(ty[0] - ty[1]))
            rows.append(row)
    if not rows:
        raise ValueError('No paired operands were captured')
    destination = ROOT / (result_name + '.csv')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(left=left, right=right, paired_operand_count=len(rows),
                   missing_left=len(b.keys() - a.keys()), missing_right=len(a.keys() - b.keys()),
                   first_nonexact_by_role={role: next((r for r in rows if r['role'] == role and not r['exact']), None)
                                          for role in ('prefill', 'decode')},
                   changed_greedy_logits=[r for r in rows if r['operand'] == 'logits' and r['left_top1'] != r['right_top1']],
                   evidence='MEASURED_ACTUAL_OPERAND_DIFFERENCES_IN_INSTRUMENTED_RUNS',
                   limitation='Captures can change batch timing. They do not reproduce clean serving latency.')
    write(ROOT / 'configs' / (result_name + '.json'), summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--result-name', default='b_repair_actual_operands')
    args = parser.parse_args()
    compare(args.left, args.right, args.result_name)


if __name__ == '__main__':
    main()
