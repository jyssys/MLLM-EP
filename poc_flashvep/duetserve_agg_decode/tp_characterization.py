"""Measure within-TP2 load effects, explicitly outside strict TP1-parity comparison."""
import fcntl
import json
import time

from .analyze import metrics, records
from .campaign import run_job
from .common import QUIET_LOCK, ROOT, initialize, resources, save, write
from .configuration import base, standard_capture_sizes
from .run import execute

REFERENCE = 'diagnostic_reference_TP2_N64_O200_v1'


def config(name, budget, slots, count, rate, **changes):
    value = base(name, budget=budget, slots=slots, count=count, rate=rate, seed=2026111611,
        explicit_cudagraph_capture_sizes=standard_capture_sizes(slots), **changes)
    value.update(arm='TP2', evaluation_arm='TP2',
        entry_module='poc_flashvep.duetserve_agg_decode.tp2',
        comparison_eligibility='DIAGNOSTIC_WITHIN_TP2_ONLY_TP1_PARITY_FAILED')
    return value


def main():
    initialize()
    diagnosis = json.loads((ROOT / 'analysis/TP2_numerical_diagnosis.json').read_text())
    assert all(not row['differs_from_first_TP2_requests'] for row in diagnosis['experiments'])
    try:
        ref = config(REFERENCE, 8192, 16, 64, 2., correctness_only=True, clean_timing=False)
        assert execute(save(ref))
        reference_metrics = metrics(REFERENCE)
        # Independent sequential/eager N8 supplies a non-identical execution
        # check. Do not redefine the already failed TP1 reference.
        independent = {r['sample_id']: r for r in records(ROOT / 'raw' /
            'diagnostic_TP2_sequential_B8192_N8_O200_v1/requests.jsonl')}
        reference_rows = records(ROOT / 'raw' / REFERENCE / 'requests.jsonl')
        assert len(reference_rows) == 64
        assert all(r['output_ids'] == independent[r['sample_id']]['output_ids']
                   for r in reference_rows if r['sample_id'] in independent)
        write(ROOT / 'analysis/TP2_diagnostic_reference_gate.json', dict(reference=REFERENCE,
            independent_sequential_N8_pass=True, original_TP1_gate=reference_metrics['output_gate'],
            primary_comparison_eligible=False, evidence='CORRECTNESS_ONLY',
            scope='Within-TP2 reference only. It does not resolve strict TP1 token parity or all-KV/Hidden equality. Fixed200-token work makes a load diagnostic useful, but not an eligible cross-method speedup claim.'))
        for qps in range(2, 11):
            for budget in ((8192, 512) if qps % 2 == 0 else (512, 8192)):
                slots = 1024 if budget == 8192 else 16
                name = f'tp_diagnostic_TP2_B{budget}_S{slots}_N128_O200_q{qps}_s2026111611_v1'
                run_job(config(name, budget, slots, 128, qps, reference_name=REFERENCE,
                    phase='TP2_WITHIN_WORLD_SIZE_DIAGNOSTIC_NOT_PRIMARY_COMPARISON'))
        write(ROOT / 'state/tp_diagnostic_characterization_complete.json', dict(time=time.time(),
            status='COMPLETE', primary_comparison_eligible=False))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
