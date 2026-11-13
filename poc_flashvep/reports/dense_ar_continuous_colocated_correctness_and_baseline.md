# Continuous Colocated DP2: correctness and baseline qualification

## Current result

`CORRECTNESS_ONLY`: standard Continuous Colocated DP2 passed N32 against the unchanged independent Native reference.
All output tokens matched.
All36 valid prompt KV planes matched byte hashes for each request.
All8000 final Hidden rows matched byte hashes for each request.
Both replicas returned all KV credits after normal completion.
This state diagnostic does not establish serving performance.

The original Chunk512 N64 reproduction also passed 64/64 output sequences after the correction.
Before the correction, the archived reproduction failed 8/64 sequences.
The new minimized reproduction failed 2/8 sequences before the correction.
It passed 8/8 afterward.
No reference token, state tolerance, weight, or SLO changed.

## Frozen environment

| Field | Value |
|---|---|
| Model | Qwen/Qwen3-8B |
| Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Arithmetic | Original BF16, exact Attention, Full36 |
| Runtime | vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1 |
| GPU mapping | Full-model replicas on GPU0 and GPU1 |
| KV budget |42 GiB per GPU |
| Prefix caching | Disabled |
| Router | Round-robin |
| Primary scheduler | Standard vLLM Continuous Batching |
| Initial token budget |8192 |
| Input/output |8000/16 for initial state qualification |

The environment manifest records GPU UUIDs and actual model configuration.
Every run preserves its immutable configuration, command, source ZIP, and source hashes.
The original checkout, staged index, external runtime, and previous evidence remain unchanged.

## Root cause

`MEASURED`: fused residual-add RMS Normalization first differed at model Layer23, indexed from zero.
It was not the Attention output or the first MLP GEMM.

For diagnostic request q1, Native and Continuous used token334 at position8000.
Native executed one query.
Continuous combined that query with another request's512 prefill queries.
All36 original prompt KV hashes matched.
Decode intermediate values matched through Layer 23 Attention and its output projection.
The following Normalization differed in 1/4096 elements.
The maximum absolute difference was 0.001953125.

The difference propagated through later layers.
The final151936 logits differed in 119675 elements.
The maximum absolute logit difference was 0.125.
Native assigned token3851 a logit of 20.875 and token1 a logit of 20.75.
Continuous assigned both tokens20.75.
Greedy selection chose token1 in Continuous instead of 3851.
This explains the changed second output token.
It does not permit accepting the changed sequence.

## Primitive causal test

`CORRECTNESS_ONLY`: replay used the saved, identical Layer23 operands and original Normalization weight.
Only the number of independent rows changed.
The runtime retained `VLLM_BATCH_INVARIANT=1`.

| Rows | Observed CUDA threads per block | Output for the victim row |
|---|---:|---|
|1/2/4/8/16/32/64/128/255 |1024 | Byte-identical to Native |
|256/512/513/1024/8000/32000 |256 | Byte-identical to the mismatching Continuous result |

A separate short profiler capture measured these actual launch dimensions.
The changed reduction geometry explains the batch-dependent BF16 result.
The capture is diagnostic only.
Its kernel durations are not clean serving latency or physical idle measurements.
The Python environment flag alone did not ensure invariant execution in the installed CUDA extension.

## Rejected explanations

| Test | Result | Supported conclusion |
|---|---|---|
| Single victim, Chunk512 | Original16-token sequence matched | Chunking alone did not reproduce this victim's failure |
| Mixed requests, Eager | Same q1/q4 mismatches | CUDA Graph replay was not necessary |
| Runtime observation extension disabled | Same q1/q4 mismatches | The observation extension was not necessary |
| Fixed random Qwen-shaped FA3 operands | Invariant in the tested shapes | This Attention fixture did not reproduce the model failure |
| TRITON_ATTN | Other output mismatches | Not a qualified replacement |
| FLASHINFER | Rejected batch-invariance configuration | Not a qualified replacement |
| FA2 primitive | Unsupported PTX toolchain | No usable FA2 result |

Related upstream Attention failures remain a separate hypothesis class.
They do not explain the first differing operation in this captured request.
See the [upstream FA3 batch-invariance issue](https://github.com/vllm-project/vllm/issues/51902).

## Correction

`phase_stable_norm.py` separates only residual-add Normalization by token phase.
Decode rows use the existing Native Decode reduction path.
Prefill rows use the existing Native full-prompt reduction path.
Short prefill segments pad only independent Normalization rows to select that same path.
The correction does not pad Attention, change KV positions, or add model tokens.

The scheduler remains continuous.
All GEMMs retain their mixed batch.
Attention retains its original execution path.
The correction does not use a cohort barrier for Colocated.
It does not change BF16 precision, model weights, activation functions, greedy sampling, or work accounting.
All compared methods enable the same optional correction.
Its generic runtime effect must remain separate from Dynamic policy gains.

## Evidence

The evidence directory is `dense_ar_decode_safe_elastic_layer_pd_20261113/`.

- `analysis/archived_chunk512_red_repro.json`: original8/64 output failure.
- `analysis/first_decode_tensor_comparison.json`: first differing layer and logits.
- `analysis/first_decode_submodule_comparison.json`: Normalization and MLP boundaries.
- `raw/rms_saved_operands_batch_probe_v1/rms_fixture.json`: saved-operand causal replay.
- `analysis/rms_actual_kernel_launch_geometry.json`: actual CUDA launch dimensions.
- `analysis/continuous_N32_full_state_gate_v2.json`: corrected request-only state gate.
- `raw/normfix_Continuous_chunk512_original_N64_v1/`: full original reproduction after correction.

The first state-analysis draft mistakenly included9 warm-up IDs.
Those IDs did not belong to the32 measurement requests.
Version2 excludes them and preserves all 32 request checks.
The earlier draft remains available and is not the qualification verdict.

## Strong baseline selection

The discovery evaluated token budgets128/256/384/512/768/1024/2048/8192 where the runtime permitted them.
It also tested Round-robin against least-outstanding routing.
The final validation compared maximum sequence counts8/16/32.
Each replica retained42 GiB KV.
No custom cohort barrier constrained Colocated.

The selection used maximum Balanced goodput on viewed validation data.
Decode attainment and generation throughput resolved ties.
The choices froze before any fresh confirmation request.

| Output cap | Token budget | Maximum sequences per replica | Routing |
|---|---:|---:|---|
| 16 | 512 | 32 | Round-robin |
| 64 | 512 | 16 | Round-robin |
| 200 | 384 | 16 | Round-robin |

The primary family uses no model compilation and retains full Decode CUDA graphs.
An optional compilation attempt completed but mismatched4/8 output sequences.
It did not qualify.
The exact cause of that compiled failure remains `NOT_IDENTIFIED`.
No primary curve includes its latency.

## Fresh serving observations

`MEASURED`: Input8000/Output16/QPS4.8 used three independent N128 starts.
Colocated TTFT p95 averaged 3026.389 ms.
Native PD averaged 10937.125 ms.
The Colocated/Native ratio was 0.27671, with 95% CI [0.19016,0.35318].

Colocated ITL p99 averaged 35.836 ms against Native 26.981 ms.
The ratio was 1.32821[1.26900,1.38182].
Colocated TPOT p95 averaged 29.323 ms against Native 25.418 ms.
The ratio was 1.15363[1.08310,1.22664].
Thus, Colocated improved TTFT but failed the Native-relative Decode 5% criterion in this cell.
This result does not mean every Colocated request failed every absolute SLO.

The three starts retained exact output sequences against the independent fresh Native reference.
Their marker-free metrics are separate from the N32 full-state diagnostic.
See [the sweep report](dense_ar_four_method_qps_sweep_results.md) for all output caps and QPS points.

## Baseline limitations

This is the strongest selected baseline within the tested exact, non-compiled family.
It is not a claim about globally optimal stock vLLM performance.
The phase-specific correction can add packing, indexing, and Normalization work in mixed batches.
The report does not hide that qualification cost.
It does not charge that cost to Layer Split as a new algorithmic benefit.

The frozen Static and Dynamic arms share the same optional numerical correction and generic logging improvement.
Natural EOS, bursty arrivals, and every possible attention backend remain outside this confirmation.
The [final report](dense_ar_decode_safe_elastic_layer_pd_final_summary.md) preserves those limits.
