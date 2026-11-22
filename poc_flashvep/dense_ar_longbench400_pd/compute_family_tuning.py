"""Validate each compute-deferral family before static discovery timing."""
import argparse
import json
import time
import torch
from .common import ROOT, write
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .metrics import summarize


def candidates():
    rows = []
    for axis in ('Layer', 'Token'):
        for cut in (2, 8, 12):
            rows.append(dict(label=('StockToken' if axis == 'Token' else axis) + str(cut),
                             axis=axis, cut=cut, stock=axis == 'Token'))
    rows.extend(dict(label=f'Hybrid{m}_f{f}', axis='Hybrid', cut=m, fraction=f)
                for m, f in ((2, .125), (8, .0625), (12, .0625)))
    rows.append(dict(label='FusedHybrid4', axis='Hybrid', cut=4, fraction=.125, fused=True))
    rows.append(dict(label='CompiledLayer4', axis='Layer', cut=4, compiled=True))
    return rows


def action_extra(action, dataset):
    extra = dict(qualification_gate='C3_continuous_actions_gate_v1.json')
    if action['axis'] == 'Token':
        data = json.loads((ROOT / 'data' / dataset).read_text())
        suffix, work = matched_suffixes([r['input_tokens'] for r in data], action['cut'])
        extra.update(suffix_by_request={'q' + str(i): s for i, s in enumerate(suffix)},
                     aggregate_work_audit=work, token_stock_model=True,
                     qualification_gate='C3_stock_token_gate_v1.json')
    if action['axis'] == 'Hybrid':
        extra.update(hybrid_suffix_fraction=action['fraction'], partition_aligned_chunks=True)
    if action.get('fused'):
        extra.update(fuse_short_hybrid=True, qualification_gate='C3_fused_short_hybrid_gate_v1.json')
    if action.get('compiled'):
        extra.update(preserve_compiled_full_model=True,
                     qualification_gate='C3_preserved_compile_Layer_stage_gate_v2.json')
    return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lane', type=int, choices=(0, 1))
    args = parser.parse_args()
    torch.set_num_threads(1)
    first = (0, 4)[args.lane]
    capsule, sha = freeze('compute_family_static_tuning_v1_20261121')
    selected = candidates()[args.lane::2]
    design = ROOT / 'configs' / f'compute_family_tuning_lane{args.lane}_v1.json'
    contract = dict(candidates=selected, pair=[first, first + 1], source_capsule_SHA256=sha,
        correctness_order=[32, 400], QPS=[1.8, 2.8], discovery_seed=2026112291,
        evidence='FROZEN_CORRECTNESS_AND_DISCOVERY_DESIGN',
        inherited_numerical_proof='Each action retains its existing independent same-geometry N8 teacher proof. '
            'CompiledLayer and fused-short Hybrid retain their separate full-state teacher proofs.',
        scope='Every action must pass new Natural32 and Natural400 state, lifetime, coverage and task-quality checks '
            'with compute deferral before timing. Stock Token keeps default compilation and uncapped source admission. '
            'Other actions retain source credit4. Report this difference. Keep every failure and discovery outcome. '
            'Select the final static action before held-out timing. No single-start safety or serving-success claim.')
    if design.exists():
        assert json.loads(design.read_text()) == contract
    else:
        write(design, contract)
    for action in selected:
        label = action['label']
        common = dict(cut=action['cut'], pair=(first, first + 1), d_compute_throttle=True,
            execution_family='INSTALLED_DEFAULT_COMPILE' if action.get('stock') or action.get('compiled') else 'MATCHED_NO_COMPILE',
            continuous_source_credit_limit=0 if action.get('stock') else 4,
            source_capsule=capsule, source_capsule_SHA256=sha, wall_timeout=4500, serving_timeout=4000)
        proofs = []
        gate_path = ROOT / 'state' / f'D_compute_family_{label}_gate_v1.json'
        for n, dataset in ((32, 'continuous_qualification32_natural.json'),
                           (400, 'longbench_balanced400_v1.json')):
            name = create(f'Q26_ComputeFamily_{label}_Natural{n}_v1', action['axis'], dataset,
                qps=2.8, seed=2026112201, correctness=True, axis_hash_workers=8,
                axis_capture_hidden_last_only=True, gpu_state_digest=True,
                axis_import_hash_after_model=True, **common, **action_extra(action, dataset))
            assert execute(name), name
            coverage = audit(name)
            assert coverage['passed'] and coverage['decode_full36_observation_complete'], name
            assert all(r['status'] == 'PASS' for r in coverage['imported_prefix_KV']), name
            mixed = no_mixed(name)
            score = quality(name)
            assert score['passed'], (name, score)
            proofs.append(dict(name=name, coverage=coverage, compute_deferral=mixed, quality=score))
            write(gate_path, dict(time=time.time(), passed=len(proofs) == 2,
                full400_passed=len(proofs) == 2, action=action, proofs=proofs,
                inherited_proofs=contract['inherited_numerical_proof'], evidence='CORRECTNESS_ONLY'))
        for qps in (1.8, 2.8):
            name = create(f'ComputeTune_Discovery200_q{qps}_{label}_v1', action['axis'],
                'balanced400_discovery.json', qps=qps, seed=2026112291, discovery_only=True,
                compute_family_gate=gate_path.name,
                scope='New family-specific Natural32/400 proof plus inherited strict N8 proof. '
                    'Strong default-compiled Stock Token remains uncapped. Static tuning is not iso-work '
                    'and does not replace held-out paired server-start safety tests.',
                **common, **action_extra(action, 'balanced400_discovery.json'))
            assert execute(name), name
            m = summarize(name)
            score = quality(name)
            print('COMPUTE_FAMILY_DISCOVERY', label, qps, m['TTFT_p95_ms'], m['ITL_p99_ms'],
                  m['TPOT_p95_ms'], m['SLO_qualified_output_tokens_per_second'], score['passed'], flush=True)
    print('COMPUTE_FAMILY_TUNING_COMPLETE', args.lane, flush=True)


if __name__ == '__main__':
    main()
