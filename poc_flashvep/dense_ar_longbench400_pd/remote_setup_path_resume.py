"""Retain both original BI0 paths and resume the unmodified transport gates."""
import argparse
import json
import time
import torch
from .common import ROOT, write, digest
from .experiment import create
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .qualification_gate import compare
from .quiescent_admission_qualification import strict_first
from .compute_throttle_qualification import no_mixed
from .remote_setup_qualification import native_state, assemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('Native', 'StockToken'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    variant = args.variant
    axis = 'Native' if variant == 'Native' else 'Token'
    old = json.loads((ROOT / 'configs' / f'Q28_RemoteSetup_{variant}_N8_v1.json').read_text())
    path_proof = json.loads((ROOT / 'analysis/alternate_original_compiled_path_teacher_results_v1.json').read_text())
    actual8 = f'Q28_RemoteSetup_{variant}_N8_v1'
    proof8 = next(r for r in path_proof['results'] if r['actual'] == actual8)
    assert proof8['comparison']['passed'], 'Never accept the mismatch without an original-model path proof'
    actual1 = f'Q28_RemoteSetup_{variant}_N1_v1'
    if variant == 'Native':
        proof1 = native_state(actual1, 'Q22_NativeValidPrefix_N8_FullState_v2')
    else:
        first = strict_first('Q13_StockFull36_Teacher4_N8_v1', actual1)
        proof1 = dict(passed=all(first.values()), rows=[first])
    assert proof1['passed']
    original_paths = dict(
        path_A_teacher='Q22_NativeValidPrefix_N8_FullState_v2' if variant == 'Native' else 'Q13_StockFull36_Teacher4_N8_v1',
        path_B_teacher=proof8['teacher'], N1_strict=proof1, N8_strict=proof8['comparison'],
        historical_strict_comparison_failed=True, cross_path_bitwise_invariance=False,
        scope='The original compiled model produces two distinct BF16 tensor paths. '
            'The N8 changed-policy tensors match an independent original-model teacher for the second path exactly. '
            'The teacher uses the original full36 layers and weights. It performs no tensor or token correction. '
            'Its kernel capture is an accuracy diagnostic,not a latency reference. '
            'The exact operator that selects these paths is NOT_IDENTIFIED. '
            'FP32 diagnostic errors do not replace the separate task-quality gate. '
            'Do not claim strict equality between paths or all independent BI0 starts.')
    results = [dict(name=actual1, numerical=proof1, numerical_context='Original path A'),
               dict(name=actual8, numerical=proof8['comparison'], numerical_context='Original path B')]
    common = dict(cut=0 if variant == 'Native' else 4, pair=tuple(old['gpus']),
        correctness=True, execution_family='INSTALLED_DEFAULT_COMPILE',
        d_compute_throttle=True, defer_D_remote_setup=True,
        continuous_source_credit_limit=0, source_capsule=old['source_capsule'],
        source_capsule_SHA256=old['source_capsule_SHA256'], wall_timeout=4500, serving_timeout=4000)
    for n, dataset in ((32, 'continuous_qualification32_natural.json'),
                       (400, 'longbench_balanced400_v1.json')):
        extra = dict(state_capture=True, axis_state_capture=False) if variant == 'Native' else dict(
            token_stock_model=True, axis_state_capture=True, axis_hash_workers=8,
            axis_capture_hidden_last_only=True, axis_import_hash_after_model=True)
        name = create(f'Q28_RemoteSetup_{variant}_N{n}_v1', axis, dataset,
            qps=2.8, seed=2026112201, gpu_state_digest=True, **common, **extra)
        assert execute(name), name
        folder = ROOT / 'raw' / name
        receipt = json.loads((folder / 'D_remote_setup_deferral.json').read_text())
        assert receipt['passed'] and not receipt['pending']
        assert any(r.get('decode_active') and r.get('received') for r in receipt['rows'])
        if variant == 'Native':
            state = native_state(name)
        else:
            state = audit(name)
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(v['status'] == 'PASS' for v in state['imported_prefix_KV'])
        score = quality(name)
        assert score['passed'], (name, score)
        results.append(dict(name=name, state=state, numerical=score, compute_deferral=no_mixed(name)))
        write(ROOT / 'state' / f'D_remote_setup_{variant}_gate_v1.json', dict(time=time.time(),
            passed=len(results) == 4, full400_passed=len(results) == 4, results=results,
            original_numerical_paths=original_paths, evidence='CORRECTNESS_ONLY',
            scope='State transport,execution coverage and task quality pass under the declared BI0 path contract. '
                'N1 and N8 have separate path-matched original teachers. Historical strict failures remain failures across paths. '
                'No cross-path bitwise or performance guarantee. Normal completion only.'))
    assemble()
    print('REMOTE_SETUP_PATH_QUALIFIED', variant, flush=True)


if __name__ == '__main__':
    main()
