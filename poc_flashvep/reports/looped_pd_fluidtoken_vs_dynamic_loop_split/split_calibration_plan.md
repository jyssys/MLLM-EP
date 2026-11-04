# Split calibration and attribution plan

Status: CPU preparation. The method GPU gate is not yet complete.

## Correctness before latency

The first Huginn partial cut is R31 of R32. Its matched token prefix contains 31/32 of the prompt.
For L2048, the token prefix contains 1984 tokens. The suffix contains 64 tokens.
The first Ouro partial cut is R3 of R4. Its matched token prefix contains 3/4 of the prompt.
For L4096, the token prefix contains 3072 tokens. The suffix contains 1024 tokens.

These plans match recurrent token-row fractions. They do not guarantee equal attention FLOPs or GPU time.
The earlier v2 configuration files remain unchanged and unexecuted.
The conservative v3 Huginn cut starts with one offloaded loop instead of four.

Each model has five correctness controls:

1. Atomic native Full-PD.
2. Staged native Full-PD.
3. Staged snapshot Full-PD.
4. Staged snapshot partial Loop Split.
5. Staged snapshot partial Token Split.

The controls use the same eight preserved prompts and output caps within each model.
They use a common D workspace and a conservative KV pool for state verification.
These heavy correctness runs are not clean latency results.
Serving requires a separate common-memory calibration. The conservative pool must not become an artificially weak baseline.

## Fair calibration

Token boundaries do not round to the execution chunk size.
Both methods receive legal large token budgets.
The reused packed-prefill hook selects `RecurrentGraphs`, not `DeviceRecurrentGraphs`.
Its graph key uses the actual row count, sequence count, and maximum query length.
It does not apply the device-owned graph class's power-of-two row buckets.
Thus a 64-token suffix does not automatically become a 256-row graph in this hook.
Both methods receive the same graph-entry budget and memory budget.
Warmup must cover their actual shapes. Unseen shapes and measured captures require explicit reporting.
A token method must not incur avoidable padding or capture costs solely because the loop method uses large rows.

Huginn's initial matched offload fractions are 1/32, 2/32, 4/32, and 8/32, subject to correctness and Decode interference.
Ouro has only three nontrivial loop boundaries. Its initial matched offload fractions are 1/4, 2/4, and 3/4.
Smaller token suffixes can provide a separate granularity control. No equivalent fractional loop is invented.
Unsafe candidates stop before serving expansion.

## Attribution limits

The following statements are structural accounting, not measured benefits.
For equal recurrent token rows, a loop cut can retain larger matrices while executing fewer remaining depth steps on D.
A token suffix instead executes the full recurrent depth on fewer rows.
This can change weight traffic, launch count, attention work, and matrix efficiency.
The experiment must measure these effects. Token-row equality alone does not establish FLOP equality.

A generic logical-layer split can select the same loop boundary.
At the same boundary, with the same state transfer and execution policy, these mechanisms are equivalent.
A Token-versus-Loop speed difference cannot establish exclusive loop-specific novelty.
Both PD workers already hold the complete model for their normal roles.
Shared weights therefore do not automatically establish an additional model-residency saving in this comparison.

The D-local 16-token probe is not a migrated serving request.
Its success does not establish that a full-prompt recurrent continuation fits the same Decode slack.
The smallest valid loop continuation must receive its own measured cost and SLO test.
