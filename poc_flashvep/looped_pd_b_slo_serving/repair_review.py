"""Summarize completed diagnostic ablations without promoting partial outputs."""
import csv
import json

from .common import ROOT, write
from .jobs import parity


def main():
    rows = []
    results = {}
    variants = [('repair_v2_ouro_' + name, name) for name in
                ('head_fixed', 'fixed_decode_core', 'decode_core_and_head',
                 'bf16_reduction_off', 'graph_off', 'async_off', 'streams_off')]
    variants.append(('repair_v3_ouro_all_linear_fixed', 'all_linear_fixed'))
    variants.append(('repair_v4_ouro_canonical_all', 'canonical_all'))
    for prefix, variant in variants:
        names = [f'{prefix}_N7_cap126_{arm}_repeat{index}'
                 for index, arm in enumerate(('PD_G1', 'B', 'PD_G1'))]
        statuses = []
        for name in names:
            path = ROOT / 'raw' / name / 'status.json'
            statuses.append(json.loads(path.read_text())['status'] if path.exists() else 'PENDING')
        result = dict(variant=variant, runs=names, statuses=statuses)
        row = dict(variant=variant, requests_per_run=7, cap=126,
                   statuses='/'.join(statuses), control_vs_B_exact='PENDING',
                   control_repeat_exact='PENDING', all_three_completed_and_exact=False,
                   evidence='MEASURED_DIAGNOSTIC_GENERATION_NOT_SERVING_SPEEDUP')
        if statuses == ['COMPLETE'] * 3:
            paired, repeated = parity(names[0], names[1]), parity(names[0], names[2])
            result.update(control_vs_B=paired, control_self_repeat=repeated)
            row.update(control_vs_B_exact=f"{paired['exact']}/{paired['N']}",
                       control_repeat_exact=f"{repeated['exact']}/{repeated['N']}",
                       all_three_completed_and_exact=all(d['N'] == 7 and d['exact'] == 7 and d['complete_ids']
                                                         for d in (paired, repeated)))
        rows.append(row)
        results[variant] = result
    with (ROOT / 'b_repair_ablation_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(ROOT / 'configs/b_repair_ablation_results.json', results)
    for row in rows:
        print(row['variant'], row['statuses'], row['control_vs_B_exact'], row['control_repeat_exact'])
    return results


if __name__ == '__main__':
    main()
