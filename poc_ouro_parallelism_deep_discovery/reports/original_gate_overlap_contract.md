# Original-gate overlap and baseline-strength contract

## Result

Mandatory R2 can overlap the R1 gate readback and host CDF update without changing the selected adaptive policy.
The runtime must preserve the actual R1 hazard and consume hazards in their original order.
It must not use an exit decision before that decision exists.
R2 is mandatory under minimum R2, so its launch does not need the R1 decision.

The current selected synchronous CDB path does not implement that overlap.
It synchronizes the compute stream before host gate routing.
The existing delayed-consumption path is not an exact replacement for the selected same-step policy.
Its transport also promotes logits before sigmoid, which can change a near-threshold decision.

This is generic adaptive-baseline engineering, not a new recurrent execution axis.
No GPU implementation or timing result occurs in this audit.
All model-output, live-exit, and performance checks remain `NOT_RUN` here.

## Pinned runtime

CDB uses commit `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The campaign wrapper loads Base 1.4B or Base 2.6B with BF16 weights and hidden states.
It checks the original gate weight and bias against the corresponding checkpoint.
It selects `last_exited`, q=0.5, minimum R2, maximum R4, no replay, and no delayed consumption.
See campaign `runtime.py::load_engine:35–59`.

The Base pins are `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` and `1ed04250da1a9936042725d302e81c8fa2ab5abd`.
This audit reads the wrapper. It does not rerun its model-loading assertions.

## Actual blocking and buffer boundaries

| Location | Actual dependency | Consequence |
|---|---|---|
| `model_runner.py::compute_recurrent_batch:661–664` | Host waits for the prior metadata-copy event before reusing host staging | This protects staging memory, not an independent gate decision |
| `model_runner.py::compute_recurrent_batch:669–693` | Metadata copies, hidden gather, recurrent launch, and bank update use the compute stream | Program order protects the next pass's data input |
| `model_runner.py::compute_recurrent_batch:695–699` | Selected same-step path synchronizes the compute stream | The host cannot route before model work finishes |
| `exit_policy.py::should_exit:161–166` | GPU sigmoid, FP32 conversion, then `.item()` | Scalar extraction blocks the host on the hazard value |
| `model_runner.py::_copy_exit_signals_to_host:799–800` | The non-async helper calls `.float().cpu().clone()` | This path supplies no asynchronous host packet |
| `model_runner.py::_copy_exit_signals_to_host:807–818` | Async path snapshots FP32 logits, records an event, and copies on D2H | The host packet needs its completion event |
| `requests.py::PendingGateResult.signal:142–150` | `ready_event.synchronize()` precedes scalar access | The wait covers that copy, not automatically the later compute stream |
| `model_runner.py::_stage_coda_batch_sync:993–1005` | Coda synchronizes before CPU token access | Removing a gate boundary does not remove all token-boundary waits |

CDB creates separate compute and D2H streams at `continuous_batching/input_outputs.py:518–524`.
Separate streams permit overlap with the required dependencies. They do not prove physical concurrency or a latency gain.
Copy engines, compute kernels, memory traffic, and host work can still contend.

## Graph scope and mutable outputs

The recurrent graph key contains the stage ID, bucket size, gate-enabled flag, and scalar KV slot.
See `model_runner.py::_run_recurrent_stage:901–907`.
Capture and replay use the compute stream at lines 908–938.

The captured body calls one physical-stack pass, its loop norm, and its original gate.
It copies the returned hidden and gate values into static output buffers.
See `model_runner.py::_forward_recurrent_stage:941–965` and `adapters/ouro.py::recurrent_step:248–264`.
CPU CDF routing is outside that graph.
Metadata staging, the hidden-bank update, and terminal-copy routing are also outside that captured body.

The recurrent buffers use `(batch_size, dtype)` as their allocation key at `model_runner.py:841–850`.
Different KV-slot graphs can therefore share the same gate-output buffer for one bucket.
The returned `exit_signals` is a view of that mutable buffer at lines 700–702.
Keeping that view does not retain the R1 value after a later replay overwrites it.

A gate packet needs its own snapshot before the next graph can overwrite the output.
The current async helper creates such a snapshot, but its dtype conversion needs a separate exactness correction.
A packet must retain its request, token, recurrent step, batch-row mapping, and completion event.
Its storage must remain alive until host consumption finishes.

## Gate arithmetic must remain unchanged

The selected source path performs this sequence:

```text
BF16 gate weights and hidden -> BF16 gate logit
BF16 logit -> GPU sigmoid with BF16 output
BF16 hazard -> FP32 conversion -> Python float
Python-float CDF/survival update in recurrent order
```

The BF16 gate-output buffer comes from `model_runner.py:847–850`.
The source computes `torch.sigmoid(signal).float().item()` at `exit_policy.py:163`.
The FP32 conversion occurs after sigmoid, not before it.

The existing async helper instead converts logits to FP32 before D2H.
`PendingGateResult.signal` returns that FP32 host scalar.
Applying the existing policy to it computes sigmoid on the host FP32 value.
The arithmetic is therefore not identical to the selected source path.

An exact-overlap design should preserve the baseline-rounded hazard before transport.
For example, compute the BF16-output GPU sigmoid, then promote that rounded hazard to FP32 for the packet.
Every BF16 value is exactly representable in FP32.
The host then consumes the hazard directly and performs no second sigmoid.
It retains the same Python-float CDF and survival update order.

This is a proposed arithmetic contract, not a completed GPU-equivalence result.
Scalar versus batched sigmoid execution and any compiler fusion still need hardware numerical checks.
Copying BF16 logits to the CPU and computing sigmoid there also needs a backend-equivalence check.
Preserving the dtype alone does not prove equal GPU and CPU kernels.

## Synthetic precision counterexample

The CPU test uses BF16 logits -3.9375 and -0.0390625.
It compares sigmoid-before-promotion against promotion-before-sigmoid.

| Quantity | Baseline-rounded path | FP32-before-sigmoid path |
|---|---:|---:|
| R1 hazard | 0.0191650390625 | 0.01912403665482998 |
| R2 hazard | 0.490234375 | 0.4902356266975403 |
| R2 CDF | 0.5000040531158447 | 0.49998437925790296 |
| Exit at q=0.5, minimum R2 | Yes | No |

These are synthetic CPU values, not observed checkpoint logits or GPU exit decisions.
They show why promotion-before-sigmoid cannot receive an automatic exactness claim.
The test runs PyTorch `2.11.0+cu129` with CPU tensors and empty CUDA visibility.
It runs no CUDA operation and loads no model.

## Exact mandatory-prefix schedule

Let h1 denote the normalized R1 output.
Let H1 and H2 denote immutable packets containing the actual baseline-rounded hazards.

```text
compute stream: R1 -> snapshot H1 -> stage R2 inputs -> R2 -> snapshot H2
D2H stream:         wait H1-ready -> copy H1
host:                               wait H1-copy -> update CDF1
D2H stream:                                                   wait H2-ready -> copy H2
host:                                                         wait H2-copy -> update CDF2 -> route
```

R2 depends on h1 and the matching R2 history, not on CDF1.
Its minimum-depth requirement makes it mandatory.
CDF2 depends on the actual R1 and R2 hazards in that order.
The host can update CDF1 while R2 executes if the launch interface returns without a full R2 wait.

Both launch and consumption need separate interfaces or equivalent staging.
Skipping only the R1 wait while leaving a blocking R2 call can hide readback but still block host CDF1 work.
Simply enabling the existing delayed-consumption setting changes more than transport.

The R1 packet must become immutable before R2 overwrites the shared gate output.
Its D2H completion need not precede the R2 launch.
No R1 output selection is possible under minimum R2.
Therefore, mandatory-prefix overlap needs no extra retained h1 for a possible R1 exit.
It still needs h1 until R2 consumes or snapshots its input.

For B rows, one FP32 hazard packet contains 4B payload bytes on device and 4B payload bytes on host.
Two pending packets require two such retained payloads.
Tensor allocation, events, row descriptors, and staging metadata add further capacity and overhead.
These are payload counts, not measured allocation sizes.

## Later eligible-depth lifetime contract

At R2 and R3, the current decision can select the current normalized hidden state.
If the next pass does not launch provisionally, the existing bank can retain that state until routing.
If the next pass launches provisionally, it must not destroy the possible exit state.

For each eligible row, retain these objects:

1. The immutable normalized h_r used by the original gate and possible LM head.
2. The gate packet and recurrent-order policy state for this token.
3. The terminal pass's per-layer KV rows until terminal publication completes.
4. A distinct hidden version and exclusive append-row leases for provisional r+1 work.
5. Completion events for every read, copy, and provisional writer before reuse.

For one BF16 normalized row, h_r occupies 4096 payload bytes.
For B rows, it occupies 4096B payload bytes.
The CDB hidden-bank write at `model_runner.py:689–693` otherwise overwrites the active slot after each pass.
Retaining only `buffers.hidden_out` is insufficient because later replay can overwrite that buffer too.

If the gate exits at r, the LM head must consume h_r, not h_(r+1).
The runtime must discard provisional continuation and publish r's terminal KV into deeper logical slots.
The copies must complete before a later token reads those slots.
They must not overwrite rows while a provisional writer or reader still uses them.

If the gate continues, the runtime may accept provisional work only if its input and historical versions match.
It must then consume the next actual gate in order.
At maximum R4, the routing rule forces exit.
Unused later gate values cannot change the selected output.

The current delayed path consumes a prior gate after executing the current pass.
It routes that current pass to coda at `continuous_api.py:1116–1122,1232–1246`.
It does not retain and select the prior h_r under this same-step contract.
That path therefore cannot supply this baseline-strength result by a configuration toggle alone.

## Baseline classification and fixed-depth caution

This optimization changes host/device orchestration around an unchanged original gate.
It exploits a known minimum-depth requirement, not an independent new recurrent input.
The same principle applies to other adaptive models with mandatory prefixes.
Its classification is `GENERIC_ADAPTIVE_BASELINE_ENGINEERING`.

The campaign's fixed CDB wrapper selects q=1 and minimum R4.
It still requests gate signals by default and uses the synchronous-signal predicate.
Those gate outputs do not determine the forced R4 output.
Combining mandatory passes or omitting unused gate routing would strengthen that baseline.
Such changes do not establish a recurrence-specific architecture speedup.

The source rejects `use_early_exit_gate=False` while `exit_threshold` remains non-None.
See `continuous_api.py::_validate_model_kwargs:1739–1745`.
A fixed-depth fast path needs an explicit configuration or code change that still selects R4.
It is not an available flag change under the current wrapper contract.

The inspected native vLLM fixed path already executes R4 without invoking its declared exit gate.
A proposed architecture must compare against that competent fixed-depth path where numerical qualification permits.
An adaptive overlap result does not establish fixed-R4 headroom.

## Bounded CPU tests and next action

The CPU logical model compares serial routing with mandatory-prefix routing over 343 hazard triples.
Both select the same R2/R3/R4 depth in all cases.
It also demonstrates why a mutable output alias is not a retained snapshot.
These tests do not execute CUDA streams, gate readbacks, model logits, or actual KV writes.

The reproducible script is `scripts/check_original_gate_overlap_cpu.py`.
Precise file/function/line locators and raw results appear in `artifacts/original_gate_overlap_contract.json`.
No external code, old worktree, coordinator file, or GPU job changes in this task.

The next implementation should first preserve the baseline-rounded GPU hazard and separate launch from host consumption.
Then compare online exits, selected logits, and terminal KV against the selected same-step path.
Only after those tests should it measure uninstrumented latency and actual hidden/buffer overhead.
Until then, exact overlap is a source-backed contract with bounded CPU support, not a measured systems result.
