"""Keep whole-prefill deflection as a strong, separately labeled control."""
import json
import time
from .common import ROOT, write, dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .metrics import summarize
from .quality_gate import compare as quality


def main():
    capsule, sha = freeze('whole_deflection_clean_discovery_v2_20261121')
    jobs = []
    for qps in (1.8, 2.8):
        for label in ('NativeStrong', 'NativeCredit4', 'WholeDeflection'):
            deflection = label == 'WholeDeflection'
            jobs.append(create(f'DeflectionAB_Discovery200_q{qps}_{label}_v2', 'Native',
                'balanced400_discovery.json', pair=(4, 5), qps=qps, seed=2026112291,
                execution_family='INSTALLED_DEFAULT_COMPILE',
                continuous_source_credit_limit=0 if label == 'NativeStrong' else 4,
                whole_prefill_deflection=deflection, d_compute_throttle=deflection,
                source_capsule=capsule, source_capsule_SHA256=sha, discovery_only=True,
                scope='Original full36 model and continuous scheduler. Keep uncapped Native and credit4 Native. '
                    'Deflection changes work placement only after observing current D quiescence and source outstanding work. '
                    'Normal Native handoffs remain continuously admitted. No partial-layer adapter or future arrival/EOS input. '
                    'One discovery start does not prove decode safety or a serving win.'))
    write(ROOT / 'configs/whole_deflection_clean_discovery_v2.json', dict(jobs=jobs,
        source_capsule_SHA256=sha, evidence='FROZEN_DISCOVERY_DESIGN'))
    gate = ROOT / 'state/whole_prefill_deflection_continuous_gate_v2.json'
    while not (gate.exists() and json.loads(gate.read_text()).get('passed')):
        assert dispatch_allowed()
        time.sleep(5)
    for name in jobs:
        assert execute(name), name
        m = summarize(name)
        score = quality(name)
        print('DEFLECTION_DISCOVERY', name, m['TTFT_p95_ms'], m['ITL_p99_ms'],
              m['TPOT_p95_ms'], m['SLO_qualified_output_tokens_per_second'], score['passed'], flush=True)


if __name__ == '__main__':
    main()
