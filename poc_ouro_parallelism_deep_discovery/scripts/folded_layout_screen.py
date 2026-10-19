"""Arithmetic screen for one explicit folded layout. No runtime prediction."""
from fractions import Fraction
from poc_ouro_parallelism_deep_discovery.contract import ROOT, write


def main():
    h, intermediate, element = 2048, 5632, 2
    attention_weights = 4 * h * h * element
    mlp_weights = 3 * h * intermediate * element
    records = []
    for P in [2, 4, 8]:
        q = Fraction(P-1, P)
        ring_weight_bytes = mlp_weights * q
        attention_broadcast_bytes = attention_weights
        weight_bytes = ring_weight_bytes + attention_broadcast_bytes
        tp_per_row = 4 * h * element * q
        folded_per_row = 2 * h * element * q
        row_threshold = weight_bytes / (tp_per_row - folded_per_row)
        record = dict(P=P, measured=False, P8_status='UNMEASURED_PROJECTION_ONLY' if P == 8 else 'CPU_ACCOUNTING_ONLY',
                      attention_weight_bytes=attention_weights, mlp_weight_bytes=mlp_weights,
                      weight_protocol_bytes_per_layer_per_rank=float(weight_bytes),
                      TP_protocol_bytes_per_fresh_row=float(tp_per_row),
                      folded_activation_protocol_bytes_per_fresh_row=float(folded_per_row),
                      prefill_volume_crossover_fresh_rows=float(row_threshold),
                      MLP_only_volume_crossover_fresh_rows=float(ring_weight_bytes/(2*h*element*q)))
        samples = []
        for fresh_rows in [1, 2, 4, 8, 1024, 4096, 8192, 16384]:
            tp = tp_per_row * fresh_rows
            folded = folded_per_row * fresh_rows + weight_bytes
            samples.append(dict(fresh_rows=fresh_rows, TP_protocol_bytes=float(tp),
                                folded_protocol_bytes=float(folded), folded_over_TP=float(folded/tp)))
        record['samples'] = samples
        records.append(record)
    target = ROOT/'artifacts/folded_layout_volume_screen_v1.json'
    assert not target.exists()
    write(target, dict(evidence_class='ANALYTIC_BOUND', guarantee='Arithmetic comparison under the explicit protocol-volume assumptions. Not a latency bound.',
          source='https://arxiv.org/html/2604.26294v1#S4.T2', records=records,
          recurrence='Repeating the same uncached schedule R times multiplies both volumes by R. The ratio does not improve merely from weight identity.',
          decode='Fresh query rows are current active tokens. Historical context length is not the fresh MLP row count.',
          exclusions=['No measured wire or HBM traffic', 'No timing or overlap model',
                      'No gradient or optimizer memory', 'No claim about generic folded layouts with different protocols',
                      'No context-parallel decode extrapolation from a full-prefill formula']))
    for record in records:
        print(record['P'], record['prefill_volume_crossover_fresh_rows'])


if __name__ == '__main__':
    main()
