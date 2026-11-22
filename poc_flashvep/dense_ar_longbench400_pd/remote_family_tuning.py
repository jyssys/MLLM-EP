"""Retune every strong static family under the common remote-setup rule."""
import argparse
import json
import time
import torch
from .common import ROOT, write, dispatch_allowed
from .compute_family_tuning import candidates, action_extra
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .remote_setup_qualification import assemble
from .metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lane', type=int, choices=(0, 1))
    args = parser.parse_args()
    torch.set_num_threads(1)
    first = (0, 4)[args.lane]
    capsule, sha = freeze('remote_family_static_tuning_v1_20261121')
    selected = candidates()[args.lane::2]
    write(ROOT / 'configs' / f'remote_family_tuning_lane{args.lane}_v1.json', dict(candidates=selected,
        QPS=[1.8, 2.8], seed=2026112291, source_capsule_SHA256=sha,
        numerical_proofs='Retain each independent action-specific N8 teacher proof. Add policy-specific Natural32/400 proofs.',
        selector='Minimize discovery TTFTp95 among correctness/quality and both Native-relative decode point checks. '
            'Use SLO-goodput to break exact ties. Include all failed and inferior candidates. '
            'Freeze final choices before held-out timing. Single-start eligibility is not the final safety CI.',
        evidence='FROZEN_CORRECTNESS_AND_STATIC_DISCOVERY_DESIGN'))
    while not assemble():
        assert dispatch_allowed()
        time.sleep(5)
    for action in selected:
        label = action['label']
        common = dict(cut=action['cut'], pair=(first, first + 1), d_compute_throttle=True,
            defer_D_remote_setup=True,
            execution_family='INSTALLED_DEFAULT_COMPILE' if action.get('stock') or action.get('compiled') else 'MATCHED_NO_COMPILE',
            continuous_source_credit_limit=0 if action.get('stock') else 4,
            source_capsule=capsule, source_capsule_SHA256=sha, wall_timeout=4500, serving_timeout=4000)
        proofs = []
        gate_path = ROOT / 'state' / f'D_remote_family_{label}_gate_v1.json'
        for n, dataset in ((32, 'continuous_qualification32_natural.json'),
                           (400, 'longbench_balanced400_v1.json')):
            name = create(f'Q29_RemoteFamily_{label}_Natural{n}_v1', action['axis'], dataset,
                qps=2.8, seed=2026112201, correctness=True, axis_hash_workers=8,
                axis_capture_hidden_last_only=True, gpu_state_digest=True,
                axis_import_hash_after_model=True, **common, **action_extra(action, dataset))
            assert execute(name), name
            folder = ROOT / 'raw' / name
            receipt = json.loads((folder / 'D_remote_setup_deferral.json').read_text())
            coverage = audit(name)
            assert receipt['passed'] and not receipt['pending']
            assert coverage['passed'] and coverage['decode_full36_observation_complete']
            assert all(r['status'] == 'PASS' for r in coverage['imported_prefix_KV'])
            score = quality(name)
            assert score['passed'], (name, score)
            proofs.append(dict(name=name, coverage=coverage, quality=score,
                compute_deferral=no_mixed(name), pending_READs_at_end=receipt['pending']))
            write(gate_path, dict(passed=len(proofs) == 2, full400_passed=len(proofs) == 2,
                action=action, proofs=proofs, evidence='CORRECTNESS_ONLY'))
        for qps in (1.8, 2.8):
            name = create(f'RemoteTune_Discovery200_q{qps}_{label}_v1', action['axis'],
                'balanced400_discovery.json', qps=qps, seed=2026112291,
                discovery_only=True, remote_family_gate=gate_path.name,
                scope='Action-specific Natural32/400 state proof before timing. '
                    'Keep strong compiled Stock Token and compiled-full Layer candidates. '
                    'This is best-static discovery,not exact iso-work or a serving confidence claim.',
                **common, **action_extra(action, 'balanced400_discovery.json'))
            assert execute(name), name
            m = summarize(name)
            score = quality(name)
            print('REMOTE_FAMILY_DISCOVERY', label, qps, m['TTFT_p95_ms'], m['ITL_p99_ms'],
                  m['TPOT_p95_ms'], m['SLO_qualified_output_tokens_per_second'], score['passed'], flush=True)
    write(ROOT / 'state' / f'remote_family_tuning_lane{args.lane}_complete_v1.json',
          dict(complete=True, candidates=selected, evidence='DISCOVERY_COMPLETE'))
    print('REMOTE_FAMILY_TUNING_COMPLETE', args.lane, flush=True)


if __name__ == '__main__':
    main()
