# C07 review remedies before GPU execution

## Scope

Two review remedies now apply to the C07 eager fixture.
Only its helper, worker, tests, and this contract changed.
The runner, trial order, physical pass, page lease, and timing intervals remain unchanged.
No archive helper changes.
No GPU run occurred.

The initial [fixture contract](c07_one_pass_fixture_contract_v1.md) remains the scope reference.
Its initial source ledger remains unchanged.
This contract records the later review remedies.

## Exact BF16 comparison

[compare_bf16_words](../c07_fixture.py:71) checks payload metadata before word equality.
[The worker comparison](../c07_fixture_probe.py:384) uses that helper for every output and KV comparison.
The helper requires these properties:

- Identical actual and reference shapes.
- BF16 dtype for both tensors.
- Identical positive payload element counts.
- Identical BF16 word counts after the INT16 views.
- One INT16 word for every BF16 element.

Only then does the helper compare words.
Broadcasting cannot make a different shape pass.
A non-BF16 payload cannot pass through an equal raw-byte interpretation.
The result retains actual and reference shapes, dtypes, and word counts.
Any contract mismatch raises `C07ValidationError`.
The helper never uses a numerical tolerance.

## Local validation and TP ordering

[check_local_validation](../c07_fixture.py:91) converts ordinary Python validation exceptions into structured `UNRESOLVED` records.
The record contains the validation stage, exception type, and message.
The handled types are `ValueError`, `TypeError`, `AssertionError`, `KeyError`, `IndexError`, and `AttributeError`.
`C07ValidationError` inherits `ValueError`.
The known cache-allocation contract violation now uses that type.

[c07_service](../c07_fixture_probe.py:435) uses this wrapper at initial, per-trial, and final validation.
It then calls the existing all-rank parity reduction.
A healthy peer therefore reaches the same reduction after an ordinary local validation failure.
The initial failure stops all timed trials.
A per-trial failure stops subsequent trials.
The original trial ordinals and reduction order remain unchanged.
No validation call moves inside a timed interval.

The wrapper does not catch execution, synchronization, event, or collective errors outside validation.
It also does not catch an unclassified `RuntimeError` from the validator.
CUDA/runtime errors do not become successful recovery records.
A fatal GPU failure can still prevent collective completion.
The external guard remains responsible for termination.
This remedy does not claim recovery from fatal CUDA errors.

## CPU evidence

The metadata regressions failed against the old comparison.
The old service also allowed a local `ValueError` to escape before its parity reduction.
The changed code passes 23 CPU tests in both interpreters.
The [suite](../tests/test_c07_fixture_cpu.py) includes these new checks:

- Broadcast-compatible shape mismatch, non-BF16 dtype, and word-count mismatch.
- No word equality operation before those metadata checks pass.
- Initial, per-trial, and final local validation failures.
- Matching reduction order for a failing rank and a simulated healthy peer.
- Propagation of a simulated CUDA `RuntimeError` without a recovery claim.
- The successful BF16 comparison path.

Six additional cases passed with actual Torch CPU tensors in the native interpreter.
Every tensor explicitly used `device='cpu'`.
Those cases covered shape mismatch, both dtype mismatches, empty payloads, equal words, and different words.
Syntax checks also passed.
These checks do not execute FA3, GPU collectives, or the physical fixture.
GPU parity and performance remain untested.

## Source and actual time

The changed source hashes are:

```text
56b7458c261aae24c5534328c05908967d52a18a2b6d4e4d4545b5a57e335e30  c07_fixture.py
e744aa497c0a6a6c55c01d15d227078aa26cb7ca70ae3b51f3144a8d8c1d1fec  c07_fixture_probe.py
aefa63c93fd67143eecde79954ae6220c4faa77b58aa6f546225e2264417074b  tests/test_c07_fixture_cpu.py
```

The segment started at `2026-10-19 15:13:58 UTC`.
The implementation and initial complete CPU checks ended at `2026-10-19 15:18:01 UTC`.
That interval is 243 seconds.
Contract drafting, lint, and repeat CPU checks follow that interval.
