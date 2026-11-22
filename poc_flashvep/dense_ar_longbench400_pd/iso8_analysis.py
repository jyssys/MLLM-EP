"""Audit every exact-work interior/pure confirmation cell and its decode gates."""
import itertools
import json
import time
from .common import ROOT, write
from .confirmation_analysis import load, ratio


def main():
    contract = json.loads((ROOT / 'configs/iso8_hybrid_confirmation_frozen_v1.json').read_text())
    rows = []
    for bucket, output in itertools.product(range(4), (16, 64)):
        cells = [c for c in contract['cells'] if c['input_bucket'] == bucket and c['output'] == output]
        assert len(cells) == 3
        row = dict(input_bucket=bucket, controlled_output=output, QPS=cells[0]['QPS'],
            status='NOT_RUN', paired_complete_starts=0, required_starts=3, starts=[])
        values = {a: [] for a in ('Native', 'Layer', 'Token', 'Hybrid')}
        try:
            for cell in cells:
                result = {}
                configs = {}
                for a in values:
                    result[a], configs[a] = load(cell[a], contract['source_capsule_SHA256'])
                if any(v is None for v in result.values()):
                    continue
                base = configs['Native']
                for cfg in configs.values():
                    assert all(cfg[k] == base[k] for k in ('dataset_SHA256', 'arrival_offsets', 'seed', 'gpus', 'output_cap'))
                work = {}
                for a in ('Layer', 'Token', 'Hybrid'):
                    folder = ROOT / 'raw' / cell[a]
                    records = {}
                    for role in ('prefill', 'decode'):
                        logged = json.loads((folder / f'axis_events_{role}.json').read_text())['rows']
                        records[role] = {r['request_id']: r for r in logged if r['event'] == 'PARTITION'
                            and r['request_id'].startswith('q') and r['request_id'][1:].isdigit()}
                    assert len(records['prefill']) == len(records['decode']) == 16
                    for rid in records['prefill']:
                        assert all(records['prefill'][rid][k] == records['decode'][rid][k]
                                   for k in ('length', 'upper', 'suffix', 'P_rows', 'D_rows'))
                    p_rows = sum(r['P_rows'] for r in records['prefill'].values())
                    d_rows = sum(r['D_rows'] for r in records['decode'].values())
                    assert p_rows == cell['P_rows_each'] and d_rows == cell['D_rows_each']
                    work[a] = dict(P_rows=p_rows, D_rows=d_rows, actual_coverage_gate_passed=True)
                for a in values:
                    values[a].append(result[a])
                row['starts'].append(dict(jobs={a: cell[a] for a in values}, actual_work=work,
                    Native_scope='Full36 P work plus the explicit D last-prompt-query recomputation. Native is not iso-offload-work matched.'))
            row['paired_complete_starts'] = len(values['Native'])
            if len(values['Native']) == 3:
                def paired(a, b, field):
                    return ratio([r[field] for r in values[a]], [r[field] for r in values[b]])
                fields = ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'SLO_qualified_output_tokens_per_second')
                row['comparisons'] = {a + '_over_' + b: {f: paired(a, b, f) for f in fields}
                    for a, b in (('Hybrid', 'Layer'), ('Hybrid', 'Token'), ('Hybrid', 'Native'),
                                 ('Layer', 'Token'), ('Layer', 'Native'), ('Token', 'Native'))}
                row['safety'] = {a: dict(
                    native_relative_decode_5percent=all(paired(a, 'Native', f)['CI95'][1] <= 1.05
                        for f in ('ITL_p99_ms', 'TPOT_p95_ms')),
                    absolute_decode_attainment_95percent=all(r['decode_SLO_attainment'] >= .95 for r in values[a]))
                    for a in ('Layer', 'Token', 'Hybrid')}
                hybrid_safe = all(row['safety']['Hybrid'].values())
                faster = all(row['comparisons']['Hybrid_over_' + a]['TTFT_p95_ms']['CI95'][1] < 1
                             for a in ('Layer', 'Token'))
                row['status'] = 'HYBRID_ISOWORK_AXIS_WIN' if faster and hybrid_safe else 'TIE_OR_NO_HYBRID_WIN'
                row['hybrid_native_TTFT_win'] = row['comparisons']['Hybrid_over_Native']['TTFT_p95_ms']['CI95'][1] < 1
                row['family_means'] = {a: {f: sum(r[f] for r in v) / len(v) for f in fields} for a, v in values.items()}
        except Exception as error:
            row.update(status='INVALID', reason=repr(error))
        rows.append(row)
    write(ROOT / 'analysis/iso8_hybrid_heldout_confirmation_v1.json', dict(time=time.time(), cells=rows,
        completed_cells=sum(r['paired_complete_starts'] == 3 for r in rows),
        evidence='MEASURED_ONLINE',
        scope='C0 actual1/1/1 with exactly equal aggregate Transformer rows for Layer8,Token8 and Hybrid4/suffix≈L/8. '
            'Attention and payload bytes are not equal. Native retains its explicit last-prompt-query recomputation. '
            'Pointwise unadjusted paired start-level empirical CIs. No familywise error guarantee. '
            'This is not best-tuned static selection,C3 serving,or sustainable capacity. Missing cells stay NOT_RUN.'))
    print([(r['input_bucket'], r['controlled_output'], r['paired_complete_starts'], r['status']) for r in rows], flush=True)


if __name__ == '__main__':
    main()
