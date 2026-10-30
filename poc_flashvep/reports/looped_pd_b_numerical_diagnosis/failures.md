# Diagnostic limitations and failed attempts

## Frozen failures remain valid

The original B reports and their rejected correctness gates remain unchanged.
Frozen trace comparison reproduces the recorded failures exactly.
New numerical corrections do not retroactively validate old performance runs.

## Trace-key correction

An early norm trace omitted the occurrence index for repeated module calls.
Huginn calls `transformer.ln_f` twice in one coda pass.
The omitted index produced an artificial coda difference.
That difference is excluded. Later traces include the occurrence index.
The confirmed core norm finding does not use that duplicate key.

## Greedy reporting correction

Early operator traces stored the first `topk` index as `top1`.
Top-k tie order does not define greedy argmax.
Those fields remain preserved but are excluded as greedy-output evidence.
Later traces record `torch.argmax` separately.

## Analysis implementation errors

The first extended phase-decomposition analysis reused the name `deltas` for an array and a list.
The script failed before publishing its extended result.
The correction separates the bootstrap array from the component list.
The corrected script checks exact reconstruction of request TTFT from host phases.

One added unit test initially contained assertions from the preceding test.
That test failed with `NameError`. The assertions were restored to their correct test.
No GPU model result depended on that test-code error.

## Resource acquisition race

One resource-stop command overlapped the start of an owned standalone diagnostic on GPU5.
The ownership guard refused to declare the pair empty. It did not escalate or stop that diagnostic.
Both numerical probes continued. Their timing is not used as clean serving performance.
Future model-worker launches also check for occupied GPUs and reserved burners before starting.
Lease acquisition and standalone launch must run in sequence.

## UCX shutdown diagnostics

Several completed runs print `cuDevicePrimaryCtxGetState` error code4 during process teardown.
The request outputs and terminal COMPLETE records precede these messages.
Their logs remain available. This task does not claim to resolve the teardown error.
They do not replace the independent output, work-count or byte-identity checks.

## Scope limits

Teacher forcing establishes numerical counterexamples on real failure histories.
It is not free-running benchmark accuracy.
Heavy tracing can change scheduling. Instrumented and clean runs remain separate.
Original HF agreement uses a small frozen-reference comparison and remains a separate gate.
No large serving-capacity campaign belongs to this diagnosis.
Any RCC-PD discussion is an untested follow-up proposal. It is not implemented here.

## Independent Huginn qualification rejects the selective candidate

The original200-request seed passes after Down and norm corrections.
The independent seed2026109001 fails199/200 for both the control repeat and B comparison.
The first differences occur at sample85/output63 and sample85/output98 respectively.
The400-request closed-loop control repeat fails393/400 before B starts.
The B arm is therefore NOT_RUN in that400-request gate.
The remaining failure is not accepted as harmless BF16 noise.

## E2E partition analysis correction

The first new partition assertion counted the first-token observer interval twice.
It combined engine-relative decode time with observer-relative token delivery.
The assertion failed before publishing the table.
The corrected partition uses the observed first token as one shared boundary.
It checks exact reconstruction of each request's measured E2E time.

## Wait-only controller replacement

The follow-up controller was replaced while its state was WAITING_FOR_QUIET_LANE.
Its UID, command line and waiting state were checked before termination.
No GPU model process was stopped.
The replacement adds the same queued-readiness change to ordinary PD_G1 as a generic control.
