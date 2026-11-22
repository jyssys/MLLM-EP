"""Bounded future-trace search in the validated whole-request C0 replay."""
import json
import time
import numpy as np
from .common import ROOT, write
from .geometry import Partition
from .c0_queue_model import aligned_suffix
from .c0_tail_replay import simulate, templates


ACTIONS = [dict(kind='Native', cut=0)]
ACTIONS += [dict(kind=kind, cut=m) for kind in ('Layer', 'Token') for m in (2, 4, 8, 12)]
ACTIONS += [dict(kind='Hybrid', cut=m, suffix_fraction=f)
            for m in (2, 4, 8, 12) for f in (.0625, .125, .25, 1 / 3)]


def partition(length, action):
    kind, m = action['kind'], action['cut']
    s = 0
    if kind == 'Token':
        s, m = aligned_suffix(length, m), 0
    elif kind == 'Hybrid':
        s = min((length - 1) // 16 * 16, max(16, round(length * action['suffix_fraction'] / 16) * 16))
    return Partition(length, m, s)


def main():
    proof = json.loads((ROOT / 'analysis/c0_tail_replay_heldout_validation_v1.json').read_text())
    assert proof['SLO_replay_gate_passed'], 'Do not compute SLO headroom from a failed replay'
    source = json.loads((ROOT / 'analysis/c0_host_service_model_v1.json').read_text())
    model, pool = source['model'], templates(source['training'])
    sets = json.loads((ROOT / 'data/iso8_hybrid_sets_frozen_v1.json').read_text())['rows']
    contract = dict(actions=ACTIONS, independent_arrival_seeds=[2026112351, 2026112352, 2026112353],
        search='Two complete coordinate sweeps from the discovery-selected best fixed action. '
            'Four common template draws score search candidates. Sixteen separate common draws score final comparisons.',
        objective='Maximize SLO-qualified output tokens/s. Use lower TTFTp95 to break exact goodput ties.',
        safety='Predicted ITLp99 and TPOTp95 must not exceed1.05 times the Native prediction. '
            'Absolute TTFT5000/everyITL34/TPOT33 thresholds are unchanged.',
        evidence='FROZEN_BOUNDED_OFFLINE_ORACLE_DESIGN',
        scope='Fixed actions are selected on discovery prompts before held-out search. '
            'The oracle sees the complete held-out arrival trace and controlled output length. '
            'Every candidate resimulates P,D,credit acquisition and credit return. '
            'Finite coordinate search is not a global optimum or a certified upper bound. '
            'C0 whole-request serialization is not a C3 safe-window model. '
            'Predicted decode gates are not measured server-start confidence intervals.')
    path = ROOT / 'configs/c0_bounded_axis_hybrid_oracle_v1.json'
    if path.exists():
        assert json.loads(path.read_text()) == contract
    else:
        write(path, contract)
    rows = []
    for bucket in range(4):
        protocol = json.loads((ROOT / 'configs' / f'matrix_input{bucket}_protocol_frozen_v1.json').read_text())
        for output in (16, 64):
            qps = next(c['qps'] for c in protocol['cells'] if c['output'] == output and c['relative_load'] == 1.15)
            def dataset(split):
                ds = next(r['dataset'] for r in sets if r['input_bucket'] == bucket and r['split'] == split)
                return ds, [r['input_tokens'] for r in json.loads((ROOT / 'data' / ds).read_text())]
            discovery_name, discovery = dataset('discovery')
            heldout_name, heldout = dataset('heldout')
            def evaluate(lengths, order, seed, draws):
                arrivals = np.r_[0., np.cumsum(np.random.default_rng(seed).exponential(1 / qps, len(lengths) - 1))].tolist()
                partitions = [partition(length, ACTIONS[a]) for length, a in zip(lengths, order)]
                values = [simulate(arrivals, lengths, partitions, output, model, pool, 371001 + i) for i in range(draws)]
                return {k: float(np.mean([v[k] for v in values])) for k in values[0]}
            def safe(value, native):
                return all(value[k] <= 1.05 * native[k] for k in ('ITL_p99_ms', 'TPOT_p95_ms'))
            def score(value):
                return value['SLO_qualified_output_tokens_per_second'], -value['TTFT_p95_ms']
            discovery_values = [evaluate(discovery, [a] * len(discovery), 2026112391, 16) for a in range(len(ACTIONS))]
            fixed = {}
            for family in ('Layer', 'Token', 'Hybrid', 'Overall'):
                indices = [a for a in range(len(ACTIONS)) if family == 'Overall' or ACTIONS[a]['kind'] == family]
                eligible = [a for a in indices if safe(discovery_values[a], discovery_values[0])]
                fixed[family] = max(eligible, key=lambda a: score(discovery_values[a])) if eligible else 0
            write(ROOT / 'configs' / f'c0_oracle_fixed_I{bucket}_O{output}_v1.json', dict(
                frozen_before_heldout_search=True, discovery_dataset=discovery_name,
                fixed_action_indices=fixed, actions=ACTIONS, discovery_predictions=discovery_values))
            for seed in contract['independent_arrival_seeds']:
                native = evaluate(heldout, [0] * len(heldout), seed, 16)
                baseline = {family: evaluate(heldout, [a] * len(heldout), seed, 16) for family, a in fixed.items()}
                order = [fixed['Overall']] * len(heldout)
                search_native = evaluate(heldout, [0] * len(heldout), seed, 4)
                evaluations = 0
                for sweep in range(2):
                    for i in (range(len(order)) if sweep == 0 else reversed(range(len(order)))):
                        current = evaluate(heldout, order, seed, 4)
                        best = order[i]
                        best_score = score(current) if safe(current, search_native) else (-1., -float('inf'))
                        for a in range(len(ACTIONS)):
                            candidate = list(order)
                            candidate[i] = a
                            value = evaluate(heldout, candidate, seed, 4)
                            evaluations += 1
                            if safe(value, search_native) and score(value) > best_score:
                                best, best_score = a, score(value)
                        order[i] = best
                final = evaluate(heldout, order, seed, 16)
                row = dict(input_bucket=bucket, output=output, QPS=qps, seed=seed,
                    discovery_dataset=discovery_name, heldout_dataset=heldout_name,
                    fixed_actions={family: ACTIONS[a] for family, a in fixed.items()},
                    Native=native, fixed_predictions=baseline, oracle_prediction=final,
                    oracle_actions_by_request={'q' + str(i): ACTIONS[a] for i, a in enumerate(order)},
                    search_evaluations=evaluations, predicted_decode_gate=safe(final, native),
                    goodput_over_discovery_best_fixed=final['SLO_qualified_output_tokens_per_second']
                        / baseline['Overall']['SLO_qualified_output_tokens_per_second'],
                    TTFT_over_discovery_best_fixed=final['TTFT_p95_ms'] / baseline['Overall']['TTFT_p95_ms'])
                rows.append(row)
                write(ROOT / 'analysis/c0_bounded_axis_hybrid_oracle_v1.json', dict(time=time.time(),
                    rows=rows, expected_cells=24, complete=len(rows) == 24,
                    evidence='OFFLINE_ORACLE', online_admission_allowed=False, scope=contract['scope']))
                print('BOUNDED_ORACLE', bucket, output, seed, row['goodput_over_discovery_best_fixed'],
                      row['TTFT_over_discovery_best_fixed'], row['predicted_decode_gate'], flush=True)


if __name__ == '__main__':
    main()
