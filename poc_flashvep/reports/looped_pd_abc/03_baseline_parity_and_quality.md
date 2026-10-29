# Baseline fidelity and quality protocol

Status: bounded qualification completed. Full benchmark accuracy is not established.

## Corrected qualification results

Evidence: `MEASURED`. Each row uses 100 natural questions and greedy generation with EOS enabled.
The output cap is 32 tokens. This is not complete-answer evaluation.

| Model | Standard PD / native RLT | A / native RLT | B / matched staged PD | Native RLT / official reference | Wilson 95% CI for reference agreement |
|---|---:|---:|---:|---:|---|
| Ouro R4 | 100/100 | 100/100 | 100/100 | 88/100 | 80.19–93.00% |
| Huginn R32, corrected EOS/RNG | 100/100 | 100/100 | 100/100 | 89/100 | 81.37–93.75% |
| Dory R2, adapter v2 | 100/100 | 100/100 | 100/100 | 80/100 | 71.12–86.66% |

The reference intervals describe output agreement, not answer accuracy.
All reference comparisons use identical input token IDs.
Ouro and Huginn also use the same per-request seeds.
Dory uses greedy generation without stochastic latent initialization.
Huginn has 97/100 matching output lengths. Ouro and Dory have 100/100 matching lengths.
The reference code uses Transformers 4.57.6 for Ouro and Huginn.
Dory uses the unchanged official-card-linked vLLM model source.
The RLT attention backend is FA4. The official paths use different execution backends.

These results do not establish bit-exact equivalence with the official implementations.
They also do not establish an accuracy regression.
The experiment retains both facts. It does not interpret internal RLT parity as official quality reproduction.
Subsequent performance results remain conditional on this reference-fidelity limitation.

The corrected native pilots reached the cap in 96/100 Ouro requests, 39/100 Huginn requests, and 100/100 Dory requests.
GSM8K test accuracy and HumanEval accuracy remain `NOT_MEASURED`.

The transfer checks contain three verified imports per model and track, including one warmup import.
All 18 A/B checks passed SHA-256 identity for valid KV and the registered hidden buffer.
B also passed reversed-fragment transfer order.
These heavy checks do not contribute clean serving latency measurements.

Machine-readable evidence: `parity.csv`, `reference_agreement.csv`, `transfer_identity.csv`, and `state/*_reference_divergences.json`.

## Measured initial fidelity

| Model | Configuration | Exact outputs | Reference | Scope |
|---|---|---:|---|---|
| Ouro R4 | Standard PD | 100/100 | Native RLT | Natural prompts, greedy, EOS enabled, cap 32 |
| Ouro R4 | A, cut 2 | 100/100 | Native RLT | Same bounded protocol |
| Ouro R4 | B | 100/100 | Matched staged G1, C256 PD | Same bounded protocol |
| Huginn R32 | Standard PD | 100/100 | Initial native RLT | Initial upstream EOS/RNG protocol |
| Huginn R32 | A, cut 16 | 100/100 | Initial native RLT | Initial upstream EOS/RNG protocol |

These counts test bounded generation. They do not measure GSM8K accuracy.
The prompts contain questions only. The harness does not supply answer labels.
The source is the existing pinned local validation manifest.
The benchmark is not the GSM8K test set.

Each prompt uses its checkpoint tokenizer and chat template.
The seed is `20261029 + sample_id`. Temperature is zero.
Prefill and decode both use the complete fixed depth.
The harness preserves EOS and records length termination separately.

## Huginn reference mismatch

The checkpoint generation config declares EOS IDs `[65505, 65508]`.
Upstream RLT uses only ID 65505 from its model config.
In the initial native pilot, 60/100 outputs contain ID 65508 before the final stop.
Those requests continue beyond a stop required by the official generation config.
The experiment preserves these raw outputs and labels the mismatch.

The official initialization also draws normal values before it overwrites them with truncated normal values.
Upstream RLT omits that first RNG draw.
Equal seeds therefore do not imply equal initial latent states.
Chunk-dependent random draws introduce another risk when prompt chunks change.

The explicit `ABC_HUGINN_REFERENCE=1` compatibility flag addresses these differences.
It uses both official EOS IDs.
It prepares the complete prompt's initial state before token chunking.
It preserves both official RNG draws and the resulting generator state.
Later decode tokens use the same initialization sequence.
PD transfers that generator state before destination decode.

This flag changes the runtime protocol, not the checkpoint weights or recurrence count.
The corrected qualification has separate `parity2` run names.
The initial 100/100 results cannot replace that qualification.

## Dory qualification

The adapter preserves the official BF16 checkpoint and its R2 setting.
The official reference uses unchanged Dory model code from PR head `876edc971507f602b18d1ec523c174a068cc02d7`.
Compatibility wrappers preserve the nested RoPE map under Transformers 5.16.
They do not change either checkpoint theta.

An initial four-prompt comparison matched 2/4 complete output sequences.
That comparison used official vLLM batch size 4 and RLT batch size 1.
It does not identify the cause of the mismatches.
The matched B1, 100-prompt comparison completed.
Adapter v1 matched 86/100 outputs. Adapter v2 matched 80/100 outputs.
Packing projections and using native RoPE did not solve reference fidelity.
No Dory lossless serving gain is accepted from either reference comparison.

A separate four-prompt diagnostic isolates full-prompt versus split coda geometry within adapter v2.
Those two paths produced identical outputs on 4/4 prompts.
Both matched the original reference on 3/4 prompts.
Their first-logit differences were nonzero despite matching generated tokens.
This small diagnostic does not explain all 20 reference mismatches.
It does not justify a fidelity or accuracy claim.

## Accuracy limits

GSM8K test accuracy and HumanEval accuracy remain NOT_MEASURED.
The current bounded parity pilot cannot establish official paper quality.
Generated-code execution requires a separate safe evaluation sandbox.
The final report must preserve these limits even if native RLT parity passes.

## Ouro post-hoc gate audit

The pinned official implementation computes all four recurrent loops before it selects output hidden states.
Its q=1 selection uses a BF16 cumulative gate probability when the model runs in BF16.
An earlier cumulative value can round to 1.0.
Thus q=1 does not guarantee selection of the final R4 hidden state mathematically.
This behavior is separate from actual compute skipping.

A synthetic CPU counterexample uses gate logits of 2.2 at all four steps.
Its cumulative values are:

| Arithmetic | Step 1 | Step 2 | Step 3 | Step 4 | Selected output depth |
|---|---:|---:|---:|---:|---:|
| FP32 | 0.90024954 | 0.99004984 | 0.99900746 | 1.0 | 4 |
| BF16 | 0.90234375 | 0.9921875 | 1.0 | 1.0 | 3 |

Evidence class: synthetic gate inputs with measured CPU tensor operations.
These are not checkpoint routing statistics or GPU serving measurements.
The example cannot explain the observed 88/100 reference agreement by itself.
`test_analysis.py` checks the arithmetic and the audit helper.

The queued official-model audit records actual gate values on selected diagnostic prompts.
It compares the unchanged q=1 path with explicit zero-based `exit_at_step=3`.
If it observes early post-hoc selection, it creates a separate 100-prompt explicit-R4 reference.
It does not overwrite or relabel the original q=1 reference.
The diagnostic sample includes earlier mismatches and cannot estimate population frequency.

Source: [pinned official Ouro implementation](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py).

The actual checkpoint audit completed on 16 diagnostic prompts.
The sample includes all 12 earlier mismatch cases and four controls.
All 16 default paths selected R4 at every observed output position.
Default outputs reproduced the earlier official reference on 16/16 prompts.
Explicit final-state selection also matched those default outputs on 16/16 prompts.
Thus the rounding counterexample does not explain these 12 observed mismatch cases.
It remains a mathematical possibility, not the measured cause here.

The audit did not run the conditional 100-prompt explicit-R4 reference.
Its trigger found no earlier selection on the diagnostic paths.
The original 88/100 reference agreement remains unchanged.
The remaining model or backend discrepancy is unresolved.
Raw evidence is `raw/ouro_reference_gate_audit/diagnostics.jsonl` and its status file.

The audit ran between complete benchmark jobs.
The experiment paused only the non-inference launcher during that interval.
It never paused an active benchmark worker.
Owned burners ran on the other seven devices during this non-timing audit.
The experiment stopped those burners before it resumed clean serving jobs.
The launcher resumed automatically. The safety guard did not need its fallback signal.
