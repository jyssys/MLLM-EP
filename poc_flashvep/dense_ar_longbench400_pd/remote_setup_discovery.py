"""Measure remote-setup deferral against same-source compute-only controls."""
import argparse
import json
import time
from .common import ROOT, write, dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .run import execute
from .metrics import summarize
from .quality_gate import compare as quality
from .remote_setup_qualification import assemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lane', type=int, choices=(0, 1))
    args = parser.parse_args()
    qps, first = ((1.8, 2), (2.8, 6))[args.lane]
    capsule, sha = freeze('remote_setup_clean_AB_discovery_v1_20261121')
    data = json.loads((ROOT / 'data/balanced400_discovery.json').read_text())
    suffix, work = matched_suffixes([r['input_tokens'] for r in data], 4)
    jobs = []
    actions = [('NativeStrong', 'Native', False, False), ('NativeSetup', 'Native', True, True)]
    for axis in ('Layer', 'Token', 'Hybrid'):
        actions.extend((axis + label, axis, True, setup) for label, setup in (('Throttle', False), ('Setup', True)))
    for label, axis, throttle, setup in actions:
        extra = {}
        compiled = axis in ('Native', 'Token')
        if axis == 'Token':
            extra.update(token_stock_model=True, suffix_by_request={'q' + str(i): s for i, s in enumerate(suffix)},
                         aggregate_work_audit=work, qualification_gate='C3_stock_token_gate_v1.json')
        elif axis != 'Native':
            extra['qualification_gate'] = 'C3_continuous_actions_gate_v1.json'
        if axis == 'Hybrid':
            extra.update(hybrid_suffix_fraction=.125, partition_aligned_chunks=True)
        jobs.append(create(f'RemoteSetupAB_Discovery200_q{qps}_{label}_v1', axis,
            'balanced400_discovery.json', cut=0 if axis == 'Native' else 4, pair=(first, first + 1),
            qps=qps, seed=2026112291, d_compute_throttle=throttle, defer_D_remote_setup=setup,
            execution_family='INSTALLED_DEFAULT_COMPILE' if compiled else 'MATCHED_NO_COMPILE',
            continuous_source_credit_limit=0 if compiled else 4,
            source_capsule=capsule, source_capsule_SHA256=sha, discovery_only=True,
            scope='Within each axis,change only remote-KV setup timing. Keep the same model,compile mode, '
                'source admission,prompts and arrival trace. Token retains its strong original compiled model. '
                'All old failures remain visible. Do not infer a five-percent confidence gate from one start.', **extra))
    write(ROOT / 'configs' / f'remote_setup_discovery_lane{args.lane}_v1.json', dict(jobs=jobs,
        source_capsule_SHA256=sha, evidence='FROZEN_CAUSAL_DISCOVERY_DESIGN',
        final_selection='Keep Original,ComputeOnly,RemoteSetup and Quiescent results. '
            'Select a policy only after correctness and same-trace decode point checks. '
            'Freeze every final action and policy before new held-out server starts. '
            'Final safety must pass both NativeStrong and common-policy Native paired confidence checks.'))
    while not assemble():
        assert dispatch_allowed()
        time.sleep(5)
    for name in jobs:
        assert execute(name), name
        m = summarize(name)
        score = quality(name)
        print('REMOTE_SETUP_DISCOVERY', name, m['TTFT_p95_ms'], m['ITL_p99_ms'],
              m['TPOT_p95_ms'], m['SLO_qualified_output_tokens_per_second'], score['passed'], flush=True)
    print('REMOTE_SETUP_DISCOVERY_COMPLETE', args.lane, flush=True)


if __name__ == '__main__':
    main()
