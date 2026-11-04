# Probe admission and completion denominators

Evidence class: source audit and measured event reconciliation.
This audit does not change any kernel, queue, service, credit, or latency threshold.

## Mismatch in the original validator

`decode_probe.py` treats `D_probe_count` as a maximum admission count.
The runtime can defer a local probe when Decode load or KV credit is insufficient.
An unadmitted probe is not an unfinished GPU request.

The original correctness helper required `completed == D_probe_count`.
This requirement conflated the admission limit with the number of admitted requests.
Its old verdict remains in `retained_native_probe_correctness.json`.
No old raw result or quota verdict is overwritten.

## Measured example

Run: `recovery_huginn_allphase_slots8_q1p971_joint_s1_profile512`.

| Quantity | Value |
|---|---:|
| Background requests offered/completed | 512/512 |
| Maximum local-probe admissions | 12 |
| Actual local-probe admissions | 6 |
| Actual local-probe completions | 6 |
| Local probes completed in the common capture interval | 1 |
| Native output mismatches among completed probes | 0 |
| Expected/observed recurrent token rows | 33,803,264 / 33,803,264 |
| P kernel union in that interval | 95.543202% |
| P finite-service proximity | 106.756077% |
| Device-qualified queue half-window growth | 1.093750 requests |
| D TPOT p95 in that interval | 133.964551 ms |
| D ITL p99 in that interval | 145.978274 ms |
| Matched response-wall profiling change | 0.460896% |
| D credit-refusal sample fraction | 28.906250% |
| Whole-run ITL p99 | 145.969898 ms |
| Whole-run maximum ITL | 1511.944346 ms |

The maximum ITL remains reported. The frozen contract is a p99 contract, not a maximum-ITL guarantee.
This exploratory run does not count as one of the new independent confirmation starts.

## Version 2 completion contract

The corrected validator requires the following conditions:

1. At least one probe was admitted.
2. Every admission ID has exactly one completed request ID.
3. Admission and completion counts agree with the runtime's close receipt.
4. Each completed request retains its admitted seed and input IDs.
5. Every completed output and depth matches the qualified native reference.
6. The full recurrent row count matches all background and admitted probe work.
7. At least one exact probe completes in the common observation interval.

Missing, duplicate, zero-work, and inconsistent completion cases fail.
The configured maximum remains reported separately. Unused admission capacity is not hidden.
All background requests must still complete. No offered background request is excluded from SLO accounting.

The original PoC specification requires real exact D-local Prefill inside the common interval.
It does not require the scheduler to reach its optional admission ceiling.
The correction therefore fixes the denominator. It does not relax a frozen performance threshold.

The new confirmation uses fresh namespaces and three independent starts.
It retains the original 12-admission ceiling, 512 background requests, and all existing SLO criteria.
The admission guard and workload remain unchanged from the exploratory all-phase eight-slot control.
New results use `ADMITTED_EXACT_V2` and retain the separate old quota verdict.

CPU regression tests cover an admitted subset, missing completion, duplicate completion, zero work, and an inconsistent close receipt.
GPU confirmation remains required before method expansion.
