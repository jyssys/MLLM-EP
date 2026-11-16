"""Resume only the prospective user-selected large-budget primary comparison."""
import fcntl
import json
import time

from .campaign import run_job
from .common import QUIET_LOCK, ROOT, digest, initialize, resources, write
from .configuration import base, standard_capture_sizes
from .confirmation import REFERENCE, freeze
from .long_window import configuration as long_configuration, assess


def primary_confirmation():
    assert json.loads((ROOT / 'analysis/fresh_reference_gate.json').read_text())['passed']
    freeze()  # Existing model worker hashes remain unchanged.
    for seed in (2026111622, 2026111623, 2026111624):
        for qps in range(2, 11):
            for method in (('paper', 'Native') if seed % 2 == 0 else ('Native', 'paper')):
                extra = dict(dataset='fresh_confirmation.json', reference_name=REFERENCE,
                             phase='FROZEN_NEW_PROMPT_CONFIRMATION')
                if method == 'Native':
                    cfg = base(f'confirm_Native_N128_O200_q{qps}_s{seed}_v1',
                        arm='PD_G1', count=128, rate=qps, seed=seed, **extra)
                else:
                    cfg = base(f'confirm_DP2_B8192_S1024_FULL_N128_O200_q{qps}_s{seed}_v1',
                        budget=8192, slots=1024, count=128, rate=qps, seed=seed,
                        explicit_cudagraph_capture_sizes=standard_capture_sizes(1024), **extra)
                freeze()
                run_job(cfg)
    write(ROOT / 'state/primary_confirmation_complete_v2.json', dict(time=time.time(), status='COMPLETE',
        starts=54, scope='B8192/S1024 and Native only. Previously completed identical configurations are reused. Old B512 measurements remain supplementary.'))


def long_windows():
    for seed in (2026111631, 2026111632, 2026111633):
        methods = ('DP2_B8192_S1024', 'Native') if seed == 2026111631 else ('DP2_B8192_S1024',)
        for qps in (2., 6.):
            for method in methods:
                cfg = long_configuration(method, qps, seed)
                run_job(cfg)
                assess(cfg['name'])
    write(ROOT / 'state/primary_long_windows_complete_v2.json', dict(time=time.time(), status='COMPLETE',
        starts=8, scope='Predeclared large-budget seeds31/32/33 at QPS2/6. Native seed31 only. No new small-budget experiment.'))


def qualify_output(cap):
    name = f'qual_primary_DP2_B8192_S1024_N32_O{cap}_v2'
    cfg = base(name, budget=8192, slots=1024, count=32, cap=cap, rate=4.,
        seed=2026111601, correctness_only=True, clean_timing=False,
        explicit_cudagraph_capture_sizes=standard_capture_sizes(1024),
        full_prompt_state_witness=True, post_graph_state_witness=True,
        full_prompt_state_witness_roles=['replica0', 'replica1'])
    run_job(cfg)
    from .common import bind_previous
    bind_previous()
    from ..dense_ar_slo_layer_pd.generic_qualification import independent
    expected = independent()
    checks = {}
    for role in ('replica0', 'replica1'):
        for rid, value in json.loads((ROOT / 'raw' / name / f'full_prompt_state_{role}.json').read_text())['requests'].items():
            checks[rid] = dict(KV_equal=value['KV'] == expected[rid]['KV'],
                Hidden_equal=value['final_hidden_SHA256'] == expected[rid]['final_hidden_SHA256'])
    passed = len(checks) == 32 and all(all(v.values()) for v in checks.values())
    write(ROOT / 'analysis' / (name + '_state_gate.json'), dict(passed=passed, checks=checks,
        evidence='CORRECTNESS_ONLY', scope='All36 valid prompt KV planes and all8000 final hidden rows. Output tokens match the frozen reference prefix.'))
    assert passed


def secondary():
    for cap in (16, 64):
        qualify_output(cap)
        for qps in (2, 6, 10):
            run_job(base(f'secondary_primary_DP2_B8192_S1024_FULL_N128_O{cap}_q{qps}_s2026111615_v2',
                budget=8192, slots=1024, rate=qps, count=128, cap=cap, seed=2026111615,
                explicit_cudagraph_capture_sizes=standard_capture_sizes(1024)))
            run_job(base(f'secondary_Native_N128_O{cap}_q{qps}_s2026111615_v1',
                arm='PD_G1', rate=qps, count=128, cap=cap, seed=2026111615))
    write(ROOT / 'state/primary_secondary_complete_v2.json', dict(time=time.time(), status='COMPLETE', starts=12))


def main():
    initialize()
    assert (ROOT / 'state/user_primary_scope_revision_v2.json').exists()
    path = ROOT / 'state/primary_confirmation_freeze_v2.json'
    if not path.exists():
        write(path, dict(time=time.time(), source_SHA256=digest(__file__),
            original_freeze=json.loads((ROOT / 'state/confirmation_freeze.json').read_text()),
            user_revision_SHA256=digest(ROOT / 'state/user_primary_scope_revision_v2.json'),
            QPS=list(range(2, 11)), seeds=[2026111622, 2026111623, 2026111624],
            methods=['Agg-DP2 B8192/S1024', 'Native PD 1P1D'],
            scope='A user-directed prospective scope change. No existing result or SLO is overwritten. All model worker hashes remain frozen.'))
    try:
        for name, function in (('confirmation', primary_confirmation), ('long_windows', long_windows), ('secondary', secondary)):
            done = ROOT / 'state' / ('primary_' + name + '_plan_result_v2.json')
            if done.exists():
                assert json.loads(done.read_text())['returncode'] == 0
                continue
            started = time.time()
            write(ROOT / 'state/primary_plan_active_v2.json', dict(phase=name, start_epoch=started))
            try:
                function()
            except BaseException as error:
                write(done, dict(start_epoch=started, end_epoch=time.time(), returncode=1, error=repr(error)))
                raise
            write(done, dict(start_epoch=started, end_epoch=time.time(), returncode=0))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
