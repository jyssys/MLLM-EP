"""Keep new-prompt reference qualification separate from held-out latency."""
import argparse
import fcntl
import json
import time

from .analyze import metrics, records
from .campaign import run_job
from .common import QUIET_LOCK, ROOT, digest, initialize, resources, save, write
from .configuration import base, standard_capture_sizes
from .run import execute

REFERENCE = 'fresh_reference_Native_N128_O200_repeat1_v1'


def reference():
    names = []
    for repeat in (1, 2):
        name = f'fresh_reference_Native_N128_O200_repeat{repeat}_v1'
        config = base(name, arm='PD_G1', count=128, rate=3., seed=2026111620,
            dataset='fresh_confirmation.json', correctness_only=True, clean_timing=False,
            full_prompt_state_witness=repeat == 1, full_prompt_state_witness_roles=['prefill'])
        assert execute(save(config)), name
        names.append(name)
    name = 'fresh_reference_independent_DP2_FULL_N32_O200_v1'
    config = base(name, budget=8192, count=32, rate=2., seed=2026111620,
        dataset='fresh_confirmation.json', correctness_only=True, clean_timing=False,
        full_prompt_state_witness=True, post_graph_state_witness=True,
        full_prompt_state_witness_roles=['replica0', 'replica1'])
    assert execute(save(config)), name
    names.append(name)
    reference_rows = {r['sample_id']: r for r in records(ROOT / 'raw' / names[0] / 'requests.jsonl')}
    checks = {}
    for candidate in names[1:]:
        rows = records(ROOT / 'raw' / candidate / 'requests.jsonl')
        checks[candidate] = dict(count=len(rows), passed=bool(rows) and all(
            r['input_ids'] == reference_rows[r['sample_id']]['input_ids'] and
            r['output_ids'] == reference_rows[r['sample_id']]['output_ids'] for r in rows))
    source = json.loads((ROOT / 'raw' / names[0] / 'full_prompt_state_prefill.json').read_text())['requests']
    states = {}
    for role in ('replica0', 'replica1'):
        for rid, value in json.loads((ROOT / 'raw' / names[2] / f'full_prompt_state_{role}.json').read_text())['requests'].items():
            states[rid] = dict(KV_equal=value['KV'] == source[rid]['KV'],
                Hidden_equal=value['final_hidden_SHA256'] == source[rid]['final_hidden_SHA256'])
    passed = len(reference_rows) == 128 and all(c['passed'] for c in checks.values()) and len(states) == 32 and all(all(c.values()) for c in states.values())
    write(ROOT / 'analysis/fresh_reference_gate.json', dict(reference=REFERENCE, comparisons=checks,
        states=states, passed=passed, evidence='CORRECTNESS_ONLY',
        scope='Native repeat verifies all128 output sequences. Independent local Full36 verifies32 outputs, valid prompt KV and final hidden. Reference latencies are not performance observations.'))
    assert passed


def freeze():
    path = ROOT / 'state/confirmation_freeze.json'
    hashes = {str(p): digest(p) for p in (ROOT.parent.parent / 'duetserve_agg_decode').glob('*.py')}
    if path.exists():
        frozen = json.loads(path.read_text())
        # Non-execution report/analysis modules can evolve. Model code and
        # declared worker adapters cannot change during confirmation.
        for filename, expected in frozen['worker_hashes'].items():
            assert digest(filename) == expected, filename
        return
    paths = [ROOT.parent.parent / 'duetserve_agg_decode' / name for name in
             ('site_bootstrap.py', 'tp_worker.py', 'tp2.py', 'iteration_trace.py', 'ownership_watch.py')]
    write(path, dict(time=time.time(), all_source_hashes=hashes,
        worker_hashes={str(p): digest(p) for p in paths},
        pool_SHA256=digest(ROOT / 'data/fresh_confirmation.json'),
        arms=dict(paper_DP2=dict(budget=8192, slots=1024, graph='FULL'),
                  strong_DP2=dict(budget=512, slots=16, graph='FULL'), Native='FROZEN_QUALIFIED_PD'),
        QPS=list(range(2, 11)), seeds=[2026111622, 2026111623, 2026111624],
        scope='Freeze before fresh performance. No parameter or SLO tuning uses these results. The finite QPS grid does not establish a global optimum.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('reference', 'evaluate'))
    args = parser.parse_args()
    initialize()
    try:
        if args.phase == 'reference':
            reference()
            return
        assert json.loads((ROOT / 'analysis/fresh_reference_gate.json').read_text())['passed']
        freeze()
        for seed in (2026111622, 2026111623, 2026111624):
            for qps in range(2, 11):
                methods = ('paper', 'strong', 'Native') if seed % 2 == 0 else ('Native', 'strong', 'paper')
                for method in methods:
                    extra = dict(dataset='fresh_confirmation.json', reference_name=REFERENCE,
                                 phase='FROZEN_NEW_PROMPT_CONFIRMATION')
                    if method == 'Native':
                        config = base(f'confirm_Native_N128_O200_q{qps}_s{seed}_v1',
                            arm='PD_G1', count=128, rate=qps, seed=seed, **extra)
                    else:
                        budget, slots = (8192, 1024) if method == 'paper' else (512, 16)
                        config = base(f'confirm_DP2_B{budget}_S{slots}_FULL_N128_O200_q{qps}_s{seed}_v1',
                            budget=budget, slots=slots, count=128, rate=qps, seed=seed, **extra)
                        if slots == 1024:
                            config['explicit_cudagraph_capture_sizes'] = standard_capture_sizes(1024)
                    freeze()
                    run_job(config)
        write(ROOT / 'state/fresh_confirmation_complete.json', dict(time=time.time(), status='COMPLETE'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
