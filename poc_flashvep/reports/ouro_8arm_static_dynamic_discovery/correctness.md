# Correctness and validity

Updated UTC: 2026-10-21T08:36:29.235050+00:00.

## Independent axes

The historical P512 strict numerical gate remains FAIL.
No new benchmark-derived numerical tolerance replaces it.
TP4 task accuracy is NUMERICALLY_UNQUALIFIED_EXPLORATORY.
Valid vocabulary IDs and finite logits permit this exploratory task lane. They do not establish tensor equivalence.

| Arm | Probe requests | Exact output | Exact depths | Bitwise KV | KV scope |
|---|---|---|---|---|---|
| L4 | 4 | 4 | 4 | 4 | All 96 logical slots per request |
| O4 | 4 | 4 | 4 | 4 | All 96 logical slots per request |
| LD | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | Pending physical probe |
| OD | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | Pending physical probe |

The PP comparisons use the same custom Single backend and actual owner-local KV tensors.
Forced-depth probes and live-gate probes remain different lanes.
The official CDB versus custom comparison, when present, is `numerical/dynamic_reference_comparison.json`.
Its backend warning must remain visible if outputs or gate decisions differ.
The official cache-copy checks and prototype tests independently check last_exited current-row semantics.

## Preserved failures

The initial new Graph path had an input-lifetime error. The corrected path retains the mask input tensor.
This correction does not explain the historical eager TP4 numerical failure.
An older Single task record used different whitespace tokenization. It remains preserved and excluded from canonical paired comparisons.
The first native TP1 task run has invalid TTFT/TPOT. It is not the timing baseline.
Earlier raw logs and failed probes remain available. See `failures.md`.

## Scope limits

Probe parity does not prove long-generation quality equivalence.
The quality CSV measures actual answers separately.
CUDA kernel service sums are not exclusive E2E component shares.
Native setup memory is not a measured full-task HBM peak.
