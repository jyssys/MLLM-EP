"""Preserve strict TP1 parity failures without presenting them as eligible curves."""
import fcntl
import json
import time

from .analyze import metrics, records
from .common import QUIET_LOCK, ROOT, initialize, resources, save, write
from .configuration import base, standard_capture_sizes
from .run import execute


def main():
    initialize()
    initial = 'qual_TP2_B8192_S1024_FULL_N8_O200_q4_s2026111601_v1'
    assert (ROOT / 'analysis' / (initial + '_output_gate.json')).exists()
    jobs = [
        (initial + '_repeat2', 8192, 1024, 4., {}),
        ('diagnostic_TP2_sequential_B8192_N8_O200_v1', 8192, 16, .15,
         dict(ar_async_scheduling=False, ar_cudagraph_mode='NONE')),
        ('diagnostic_TP2_B512_S16_FULL_N8_O200_q4_v1', 512, 16, 4., {}),
    ]
    try:
        results = []
        for name, budget, slots, rate, extra in jobs:
            config = base(name, budget=budget, slots=slots, count=8, rate=rate,
                correctness_only=True, clean_timing=False,
                explicit_cudagraph_capture_sizes=standard_capture_sizes(slots), **extra)
            config.update(arm='TP2', evaluation_arm='TP2',
                          entry_module='poc_flashvep.duetserve_agg_decode.tp2')
            complete = execute(save(config))
            assert complete, name
            # A failed original-output gate is the result of this diagnosis.
            # Do not bypass run_job's strict gate for performance experiments.
            value = metrics(name)
            rows = records(ROOT / 'raw' / name / 'requests.jsonl')
            first = {r['sample_id']: r for r in records(ROOT / 'raw' / initial / 'requests.jsonl')}
            changes = [r['sample_id'] for r in rows if r['output_ids'] != first[r['sample_id']]['output_ids']]
            results.append(dict(name=name, original_output_gate=value['output_gate'],
                differs_from_first_TP2_requests=changes, complete=complete))
        write(ROOT / 'analysis/TP2_numerical_diagnosis.json', dict(time=time.time(),
            initial=initial, experiments=results, evidence='CORRECTNESS_ONLY',
            exact_reduction_cause='NOT_IDENTIFIED',
            scope='Repeat, sequential eager, and small-budget interventions. Different TP reduction geometry can change BF16 rounding. This is a hypothesis, not an exclusive causal attribution. No failed original-output arm is eligible for the primary performance figures.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
