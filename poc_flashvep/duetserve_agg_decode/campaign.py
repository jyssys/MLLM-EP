"""Resume a bounded fixed-setting sweep; stop before unqualified GPU work."""
import argparse
import fcntl
import json
import time

from .analyze import metrics, output_gate
from .common import QUIET_LOCK, ROOT, initialize, resources, save, write
from .configuration import base, standard_capture_sizes
from .run import execute


QUALIFICATIONS = (
    'qual_DP2_B8192_S16_FULL_N32_O200_q4_s2026111601_v2',
    'qual_DP2_B512_S16_FULL_N32_O200_q4_s2026111601_v1',
)


def qualify():
    for name in QUALIFICATIONS:
        receipt = ROOT / 'raw' / name / 'dispatcher_result.json'
        assert receipt.exists() and json.loads(receipt.read_text())['complete'], name
        assert output_gate(name)['passed'], name


def run_job(config):
    name = save(config)
    write(ROOT / 'state/active_job.json', dict(name=name, time=time.time(), status='STARTING'))
    assert execute(name), 'Failed GPU job: ' + name
    value = metrics(name)
    assert value['output_gate']['passed'], 'Output failure: ' + name
    write(ROOT / 'state/active_job.json', dict(name=name, time=time.time(), status='COMPLETE',
        mean_TBT_ms=value['mean_TBT_ms'], ITL_p99_ms=value['ITL_p99_ms'], TPOT_p95_ms=value['TPOT_p95_ms']))
    print('METRICS', name, round(value['mean_TBT_ms'], 3), round(value['ITL_p99_ms'], 3),
          round(value['TPOT_p95_ms'], 3), flush=True)
    return value


def qualify_concurrency(slots):
    captures = standard_capture_sizes(slots)
    name = f'qual_DP2_B8192_S{slots}_FULL_N32_O200_q4_s2026111601_v1'
    run_job(base(name, budget=8192, slots=slots, count=32, rate=4., seed=2026111601,
        correctness_only=True, clean_timing=False,
        explicit_cudagraph_capture_sizes=captures, full_prompt_state_witness=True,
        post_graph_state_witness=True, full_prompt_state_witness_roles=['replica0', 'replica1']))
    from ..dense_ar_slo_layer_pd.generic_qualification import independent
    expected = independent()
    checks = {}
    for role in ('replica0', 'replica1'):
        for rid, value in json.loads((ROOT / 'raw' / name / f'full_prompt_state_{role}.json').read_text())['requests'].items():
            checks[rid] = dict(KV_equal=value['KV'] == expected[rid]['KV'],
                              Hidden_equal=value['final_hidden_SHA256'] == expected[rid]['final_hidden_SHA256'])
    passed = len(checks) == 32 and all(all(c.values()) for c in checks.values())
    write(ROOT / 'analysis' / (name + '_state_gate.json'), dict(checks=checks, passed=passed,
        evidence='CORRECTNESS_ONLY', scope='All36 prompt KV planes and all8000 final Hidden rows match Native. Diagnostic copies exclude this run from clean performance.'))
    assert passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('dp2', 'native', 'confirmation', 'budget', 'budget_coverage', 'secondary', 'state', 'tp2_qualification', 'tp2', 'concurrency', 'paper_dp2', 'profile'))
    args = parser.parse_args()
    initialize()
    qualify()
    try:
        if args.phase == 'dp2':
            # Balance method order. All QPS points share the same Poisson seed.
            for qps in range(2, 11):
                budgets = (8192, 512) if qps % 2 == 0 else (512, 8192)
                for budget in budgets:
                    run_job(base(f'sweep_DP2_B{budget}_S16_FULL_N128_O200_q{qps}_s2026111611_v1',
                        budget=budget, rate=qps, count=128, seed=2026111611))
        elif args.phase == 'native':
            native_qual = 'qual_Native_N32_O200_q4_s2026111601_v2'
            run_job(base(native_qual, arm='PD_G1',
                rate=4., count=32, seed=2026111601, correctness_only=True, clean_timing=False,
                full_prompt_state_witness=True, full_prompt_state_witness_roles=['prefill'],
                imported_native_state_witness=True))
            from ..dense_ar_slo_layer_pd.generic_qualification import independent
            expected = independent()
            source = json.loads((ROOT / 'raw' / native_qual / 'full_prompt_state_prefill.json').read_text())['requests']
            imported = json.loads((ROOT / 'raw' / native_qual / 'imported_native_prompt_KV.json').read_text())['requests']
            checks = {rid: dict(P_KV_equal=value['KV'] == expected[rid]['KV'],
                P_Hidden_equal=value['final_hidden_SHA256'] == expected[rid]['final_hidden_SHA256'],
                D_imported_KV_equal=imported.get(rid, {}).get('KV') == value['KV'])
                for rid, value in source.items()}
            passed = len(checks) == 32 and len(imported) == 32 and all(all(c.values()) for c in checks.values())
            write(ROOT / 'analysis' / (native_qual + '_state_gate.json'), dict(checks=checks, passed=passed,
                evidence='CORRECTNESS_ONLY', scope='Independent Source Full36 KV/Hidden and all36 actually imported D prompt KV planes. No extra model forward.'))
            assert passed
            for qps in range(2, 11):
                run_job(base(f'sweep_Native_N128_O200_q{qps}_s2026111611_v2', arm='PD_G1',
                    rate=qps, count=128, seed=2026111611))
        elif args.phase == 'confirmation':
            for seed in (2026111612, 2026111613):
                for qps in (2, 6, 10):
                    for budget in ((512, 8192) if seed % 2 == 0 else (8192, 512)):
                        run_job(base(f'sweep_DP2_B{budget}_S16_FULL_N128_O200_q{qps}_s{seed}_v1',
                            budget=budget, rate=qps, count=128, seed=seed))
                    run_job(base(f'sweep_Native_N128_O200_q{qps}_s{seed}_v1', arm='PD_G1',
                        rate=qps, count=128, seed=seed))
        elif args.phase == 'budget':
            for qps in (4, 8):
                # S128 avoids the S16 concurrency ceiling. The common capture
                # maximum256 is legal even at the smallest token budget256.
                for budget in (256, 512, 1024, 2048, 4096, 8192):
                    run_job(base(f'budget_DP2_B{budget}_S128_FULL_N128_O200_q{qps}_s2026111614_v1',
                        budget=budget, slots=128, rate=qps, count=128, seed=2026111614,
                        chunk=8000,
                        explicit_cudagraph_capture_sizes=standard_capture_sizes(128)))
        elif args.phase == 'budget_coverage':
            for qps in (4, 8):
                run_job(base(f'coverage_DP2_B512_S128_FULL_CAP512_N128_O200_q{qps}_s2026111614_v1',
                    budget=512, slots=128, rate=qps, count=128, seed=2026111614,
                    chunk=8000, explicit_cudagraph_capture_sizes=standard_capture_sizes(128) + [512]))
        elif args.phase == 'secondary':
            for cap in (16, 64):
                for qps in (2, 6, 10):
                    for budget in (8192, 512):
                        run_job(base(f'secondary_DP2_B{budget}_S16_FULL_N128_O{cap}_q{qps}_s2026111615_v1',
                            budget=budget, rate=qps, count=128, cap=cap, seed=2026111615))
                    run_job(base(f'secondary_Native_N128_O{cap}_q{qps}_s2026111615_v1', arm='PD_G1',
                        rate=qps, count=128, cap=cap, seed=2026111615))
        elif args.phase == 'state':
            from ..dense_ar_slo_layer_pd.generic_qualification import independent
            expected = independent()
            assert len(expected) == 32
            for budget in (8192, 512):
                name = f'state_DP2_B{budget}_S16_FULL_N32_O200_q4_s2026111601_v1'
                run_job(base(name, budget=budget, count=32, rate=4.0, correctness_only=True,
                    clean_timing=False, full_prompt_state_witness=True, post_graph_state_witness=True,
                    full_prompt_state_witness_roles=['replica0', 'replica1']))
                checks = {}
                for role in ('replica0', 'replica1'):
                    for rid, value in json.loads((ROOT / 'raw' / name / f'full_prompt_state_{role}.json').read_text())['requests'].items():
                        checks[rid] = dict(KV_equal=value['KV'] == expected[rid]['KV'],
                            Hidden_equal=value['final_hidden_SHA256'] == expected[rid]['final_hidden_SHA256'],
                            prompt_queries=value['queries'])
                passed = len(checks) == 32 and all(c['KV_equal'] and c['Hidden_equal'] and c['prompt_queries'] == 8000 for c in checks.values())
                write(ROOT / 'analysis' / (name + '_state_gate.json'), dict(checks=checks, passed=passed,
                    evidence='CORRECTNESS_ONLY', scope='All36 valid prompt KV planes and all8000 final hidden rows. Hashes match the independent Native reference at the same prompt order.'))
                assert passed, name
        elif args.phase == 'tp2_qualification':
            for budget in (8192, 512):
                slots = 1024 if budget == 8192 else 16
                for count in (8, 32):
                    config = base(f'qual_TP2_B{budget}_S{slots}_FULL_N{count}_O200_q4_s2026111601_v1',
                        budget=budget, slots=slots, count=count, rate=4.0, correctness_only=True)
                    if slots == 1024:
                        config['explicit_cudagraph_capture_sizes'] = standard_capture_sizes(1024)
                    config.update(arm='TP2', evaluation_arm='TP2',
                        entry_module='poc_flashvep.duetserve_agg_decode.tp2')
                    run_job(config)
        elif args.phase == 'tp2':
            for budget in (8192, 512):
                slots = 1024 if budget == 8192 else 16
                assert output_gate(f'qual_TP2_B{budget}_S{slots}_FULL_N32_O200_q4_s2026111601_v1')['passed']
            for qps in range(2, 11):
                for budget in (8192, 512):
                    slots = 1024 if budget == 8192 else 16
                    config = base(f'sweep_TP2_B{budget}_S{slots}_FULL_N128_O200_q{qps}_s2026111611_v1',
                        budget=budget, slots=slots, rate=qps, count=128, seed=2026111611)
                    if slots == 1024:
                        config['explicit_cudagraph_capture_sizes'] = standard_capture_sizes(1024)
                    config.update(arm='TP2', evaluation_arm='TP2',
                        entry_module='poc_flashvep.duetserve_agg_decode.tp2')
                    run_job(config)
        elif args.phase == 'concurrency':
            for slots in (32, 1024):
                qualify_concurrency(slots)
                for qps in (2, 6, 10):
                    run_job(base(f'concurrency_DP2_B8192_S{slots}_FULL_N128_O200_q{qps}_s2026111611_v1',
                        budget=8192, slots=slots, count=128, rate=qps, seed=2026111611,
                        explicit_cudagraph_capture_sizes=standard_capture_sizes(slots)))
        elif args.phase == 'paper_dp2':
            qualify_concurrency(1024)
            for qps in range(2, 11):
                run_job(base(f'concurrency_DP2_B8192_S1024_FULL_N128_O200_q{qps}_s2026111611_v1',
                    budget=8192, slots=1024, count=128, rate=qps, seed=2026111611,
                    explicit_cudagraph_capture_sizes=standard_capture_sizes(1024)))
        elif args.phase == 'profile':
            for budget, slots, qps in ((8192, 1024, 2), (8192, 1024, 6), (512, 16, 6)):
                labels = ('A1', 'HOST_ONLY', 'A2') if qps == 2 else ('A1', 'HOST_ONLY', 'HOST_EVENT', 'NSYS', 'A2')
                for label in labels:
                    extra = {}
                    if label in ('HOST_ONLY', 'HOST_EVENT', 'NSYS'):
                        extra.update(iteration_trace=True, clean_timing=False)
                    if label == 'HOST_ONLY':
                        extra['iteration_trace_events'] = False
                    if slots == 1024:
                        extra['explicit_cudagraph_capture_sizes'] = standard_capture_sizes(1024)
                    if label == 'NSYS':
                        extra.update(nsys_bounded_capture=True, nsys_roles=['replica0'],
                                     nsys_offset_s=10., nsys_seconds=2.)
                    name = f'profile_DP2_B{budget}_S{slots}_FULL_N128_O200_q{qps}_s2026111616_{label}_v1'
                    existing = ROOT / 'raw' / name
                    if existing.exists() and not (existing / 'dispatcher_result.json').exists():
                        assert not (existing / 'command.json').exists() and not (existing / 'requests.jsonl').exists()
                        write(existing / 'prelaunch_failure.json', dict(evidence='FAILED',
                            scope='The preserved controller log records a pre-launch ownership refusal. No model command or measurement existed. Do not delete or overwrite this namespace.'))
                        name = name.removesuffix('_v1') + '_v2'
                    try:
                        run_job(base(name,
                            budget=budget, slots=slots, count=128, rate=qps, seed=2026111616, **extra))
                    except AssertionError as error:
                        if label != 'NSYS' or resources().processes():
                            raise
                        write(ROOT / 'analysis' / f'failed_nsys_B{budget}_S{slots}_q{qps}.json',
                            dict(time=time.time(), error=repr(error), evidence='FAILED',
                                 physical_kernel_measurement='NOT_MEASURED',
                                 scope='Preserve the failed capture. Run the marker-free A2 control in a fresh process. No CUDA-time claim uses this failure.'))
        write(ROOT / 'state' / (args.phase + '_phase_complete.json'), dict(time=time.time(), status='COMPLETE'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
