"""Stress per-request Native, pure-axis and interior actions without a selector claim."""
import json
import time
import torch
from .common import ROOT, write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed


def main():
    torch.set_num_threads(1)
    prior = json.loads((ROOT / 'state/C0_family_with_native_q24_gate_v1.json').read_text())
    assert prior['passed'] and prior['native_zero_boundary']
    actions = [dict(kind='Native', cut=0)]
    actions += [dict(kind=axis, cut=m) for axis in ('Layer', 'Token') for m in (2, 4, 8, 12)]
    actions += [dict(kind='Hybrid', cut=m, suffix_fraction=f)
                for m, f in ((2, .125), (4, .125), (8, .0625), (12, .0625))]
    capsule, sha = freeze('heterogeneous_native_axis_400_qualification_v1_20261121')
    results = []
    for n, dataset in ((32, 'continuous_qualification32_natural.json'),
                       (400, 'longbench_balanced400_v1.json')):
        mapping = {'q' + str(i): actions[i % len(actions)] for i in range(n)}
        name = create(f'Q27_Heterogeneous_NativeLayerTokenHybrid_Natural{n}_v1', 'Hybrid', dataset,
            cut=12, pair=(0, 1), correctness=True, qps=2.8, seed=2026112201,
            axis_actions_by_request=mapping, hybrid_suffix_fraction=.125,
            partition_aligned_chunks=True, d_compute_throttle=True,
            axis_capture_hidden_last_only=True, gpu_state_digest=True,
            axis_import_hash_after_model=True, axis_hash_workers=8,
            continuous_source_credit_limit=4, source_capsule=capsule, source_capsule_SHA256=sha,
            wall_timeout=4500, serving_timeout=4000,
            scope='Predetermined round-robin action stress. Every request remains original natural traffic. '
                'Native fallback uses the normal connector and last-query recomputation. '
                'This is a state/lifetime correctness test,not an online prediction rule or a performance result.')
        assert execute(name), name
        coverage = audit(name)
        assert coverage['passed'] and coverage['decode_full36_observation_complete']
        assert all(v['status'] == 'PASS' for v in coverage['imported_prefix_KV'])
        score = quality(name)
        assert score['passed'], score
        mix = no_mixed(name)
        assert coverage['mixed_batches'] > 0, 'Heterogeneous row handling must execute during this test'
        results.append(dict(name=name, n=n, coverage=coverage, numerical=score, compute_deferral=mix))
        write(ROOT / 'state/C3_heterogeneous_native_axis400_gate_v1.json', dict(time=time.time(),
            passed=len(results) == 2, full400_passed=len(results) == 2, mixed_rows_passed=True,
            qualified_axes=['Native', 'Layer', 'Token', 'Hybrid'],
            qualified_actions=[dict(axis=a['kind'], cut=a['cut'], suffix_fraction=a.get('suffix_fraction')) for a in actions],
            results=results, evidence='CORRECTNESS_ONLY',
            scope='Inherited independent per-action N8 proofs plus heterogeneous Natural32/400 stress. '
                'Compute deferral prevents meaningful prefill/decode coexecution. The scheduler still batches compatible requests. '
                'This is not a measurement of a dynamic controller,ITL noninferiority or serving gain.'))
    print('HETEROGENEOUS_NATIVE_AND_AXES_FULL400_PASS', flush=True)


if __name__ == '__main__':
    main()
