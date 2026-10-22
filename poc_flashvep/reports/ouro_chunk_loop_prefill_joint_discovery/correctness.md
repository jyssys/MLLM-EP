# Correctness and quality scope

## Preserved model contract

The model is pretrained base Ouro-1.4B, BF16, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The checkpoint SHA256 is `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
The runtime executes 24 shared physical layers at four recurrent depths.
Each depth/layer pair has its own KV namespace.
Prompt positions, causal masks, RoPE, normalization, residual order, weights, and sampler remain native RLT operations.

The G2 adapter changes only prefill scheduling and frontier progression.
It executes two consecutive loops from the current frontier, then returns control.
It clips a final group to the remaining depth.
It does not reduce model depth or implement Dynamic Early Exit.
The decode path remains Static R4.

## CPU state-machine tests

Before the patch, the stock split-prefill CPU tests reported 33 passes and 28 GPU deselections.
After the patch, the complete stock CPU suite reported 800 passes, 34 skips, and 322 GPU deselections.
The new grouped test file reported eight passes.

The new tests cover:

- Budgets smaller than the requested chunk, partial final chunks, and chunk capacity greater than the prompt.
- G1/G2/G4 output agreement with full-R4 issued work.
- A one-loop final group after a frontier at R3.
- Cancellation, staged-state cleanup, and request-ID reuse.
- Rejection of unsupported group sizes and grouped Mixed decode batches.

Tiny random models are CPU test fixtures only.
They are not evidence for pretrained accuracy or serving performance.
The JUnit records and exact commands remain in this directory and [reproduce.md](reproduce.md).

## Pretrained physical-GPU probes

All probes used physical GPU4.
`correctness_probes.csv` contains 43 executions and 31 non-reference comparisons.
Atomic was the reference within each same-backend, same-chunk probe set.

| Probe lane | Prompt tokens | Chunk tokens | Modes | Non-reference comparisons |
| --- | --- | ---: | --- | ---: |
| Stock Triton sanity | 32, 128, 512 | 128 | Atomic, Split, Mixed | 6 |
| Patched Triton | 32, 128, 512, 2048 | 128 | Atomic, Split, Mixed, G2 | 12 |
| Patched FA4 | 32, 128, 512, 2048 | 128 | Atomic, Split, Mixed, G2 | 12 |
| Long FA4 probe | 8192 | 1024 | Atomic, G2 | 1 |

All 31 comparisons had:

- Exact generated token IDs and recorded R4 exit-depth sequences.
- Exact final-prefill hidden state and logits.
- Exact K and V contents for every prompt position in all 96 depth/layer namespaces.
- Zero maximum absolute difference and zero relative L2 difference for the compared tensors.

The diagnostic relative-L2 limits were 0.03 before measurement.
No failed limit was relaxed. Exact equality was stronger than the required limit here.
These same-RLT comparisons do not independently validate the entire RLT model against a different HF engine.

The probe synchronizes the native runner before cloning hidden state and KV.
It never reads a tensor while its producer remains incomplete.
KV snapshots cover the completed prompt before its first sampled output.
They do not compare every decode-updated KV element after the eight-token probe generation.
The eight generated tokens are bounded numerical probes, not accuracy tests.

## Full-R4 work and lifecycle

`full_R4_work_validation.csv` contains 1,551 request-work checks and zero failures.
For every recorded request, issued prefill token rows times executed group width equaled four times the prompt length.
Recorded generated-token depths also remained four.
These are actual scheduler descriptors, backed by separate pretrained GPU probes.
They are not an independent count of every GPU layer kernel.

The patch advances every completed depth frontier in a group.
The native executor prepares each loop's KV metadata and preserves its existing GPU dependency events.
Only one chunk per request remains staged.
The retained activation survives until the next legal group consumes it.
First-token coda executes only after full-prompt R4 completion.

The adapter rejects a grouped Mixed configuration.
This prevents a prefill group from advancing unrelated decode rows through two loops.
Stock Mixed remains an independent, unmodified baseline.

## Batched natural-output fidelity

The corrected confirmation ran 600 request executions.
Each request generated 128 free-running tokens and reached the length cap.
There were zero EOS completions in this lane.

Grouped and Split matched exact capped sequences in **112/120** paired held-out request executions.
They matched depth sequences and output lengths in **120/120**.
`natural_output_parity.csv` records request IDs and first divergent token indices.

Changing scheduling can change GEMM batch geometry and numerical order in BF16.
That is a possible cause of these eight sequence differences. This PoC did not isolate it completely.
Therefore, the batched lane does not pass a universal token-exact parity claim.
Equal output length also does not establish equal per-layer arithmetic or identical generated content.
Throughput comparisons remain natural-serving comparisons, not fixed-token hardware replay.

Observer and graph controls preserved outputs and R4 depth sequences in the matched four-request checks.
See `observer_and_control_measurements.csv` for the precise reference and denominator per control.
These checks do not establish quality for different unseen prompts.

## Accuracy status and decision consequence

GSM8K64, MATH50, and full benchmark accuracy are **NOT_RUN**.
The capped serving lane cannot establish complete-answer correctness or quality noninferiority.
No accuracy PASS, accuracy delta of zero, or model-quality preservation claim follows from these measurements.

The fixed-quantum continuation gate already failed on registered performance outcomes.
We did not expand into a quality-certified Adaptive Joint method or multi-GPU runtime.
If the narrow tail-SLO observation is investigated later, complete natural answers and paired quality evaluation are mandatory.

## Summary

| Claim | Status |
| --- | --- |
| Grouped prefill CPU state progression | PASS in the recorded tests |
| Isolated pretrained same-RLT token/hidden/logit/prompt-KV parity | PASS, 31/31 comparisons |
| Recorded full-R4 issued prompt work | PASS, 1551/1551 checks |
| Universal batched natural-output parity | NOT_PASS, 112/120 Grouped/Split sequences match |
| Complete reasoning benchmark quality | NOT_RUN |
| Adaptive-policy correctness | NOT_RUN |
| Multi-GPU correctness | NOT_RUN |
