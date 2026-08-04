# FlashVEP Batch 16/32 Quick PoC

Date: 2026-08-04

Decision: **HOLD**

Scope: Phase 1b TP2/DP2/EP4 reuse only; Phase 2A and live overlap were not started.

## Executive result

Batch 16 produced a valid low-overhead measurement. The median local-expert
critical time grew from 0.4491 ms at Batch 1 to 1.9950 ms, and the selected
layer-sum timestamp oracle grew from 1.0823x to 1.1750x. Communication relative
to expert compute fell from 5.4518x to 3.1155x. However, expert fraction was
only 21.55%, inside the HOLD band, and the 17.50% optimistic gain was only
slightly larger than the 13.56% profiling overhead.

Batch 32 executed successfully without OOM and preserved the expected DPEP
path, but its profiler overhead was 61.81%. Its stage values are retained as
diagnostic observations only and are excluded from the gate.

## Environment and fixed workload

- Model: exact local Qwen3-VL-30B-A3B-Instruct snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- GPUs: physical GPU 4, 5, 6, 7 only
- Parallelism: TP=2, DP=2, EP=4, PP=1; BF16; sequence-parallel MoE
- Image/prompt: fixed 896x896 input, 799 prompt tokens per request
- Generation: `max_tokens=1`; every request returned token ID 1986
- Prefix caching: disabled
- Selected layers: 12, 24, 36
- Warmups/measured iterations: 3/8
- Batch 16/32 token budgets: 8,192/16,384
- vLLM: asynchronous scheduling and chunked prefill, eager execution

Before launch, GPUs 4-7 each had about 81.1 GiB free. Model loading consumed
15.81-15.91 GiB per GPU. The fixed 1 GiB KV cache reported capacity for 21.33
concurrent 1,024-token sequences per DP engine, so the Batch 32 local batch of
16 fit without changing the model, image, dtype, or token count.

## Request distribution and actual batching

The runner created independent prompt objects and sent half of the global batch
to each DP process.

| Global batch | DP0 real requests | DP1 real requests | Prompt tokens | Routed assignments/layer | Result |
|---:|---:|---:|---:|---:|---|
| 16 | 8 | 8 | 12,784 | 102,272 | all 16 outputs consistent |
| 32 | 16 | 16 | 25,568 | 204,544 | all 32 outputs consistent |

This was not one-request-at-a-time execution. The asynchronous scheduler split
each submitted global batch into model microbatches:

- Batch 16: 2-3 active model calls per measured iteration; the split varied by
  iteration, but each DP rank always accumulated exactly 8 real requests.
- Batch 32: two active model calls in every measured iteration, with real
  request splits `[1,1]` then `[15,15]` across DP0/DP1.

Stage times below sum the sequential, real-request-containing model calls for
each rank and then use the four-rank critical value. Pure idle model calls are
excluded. TP padding and idle-DP dummy assignments are accounted separately.

## DPEP path proof

Both batch sizes recorded the same actual runtime path on all selected layers
and all four ranks:

- prepare/finalize: `MoEPrepareAndFinalizeNaiveDPEPModular`
- all-to-all manager: `AgRsAll2AllManager`
- dispatch: `dispatch_all_gatherv`
- combine: `combine_reduce_scatterv`
- final sequence-parallel TP combine: `tensor_model_parallel_all_gather`
- local expert backend: `TritonExperts`
- local/global experts: 32/128 per EP rank

Each final audit contained 12 runtime-path records (3 layers x 4 ranks) and
4,248 observed dispatch plus 4,248 combine collective calls. Thus the workload
did execute DPEP dispatch/combine collectives; it was not a local-only MoE path.

## Stage comparison

All entries are medians in milliseconds over layers 12/24/36 and eight
iterations. Batch 1 is recomputed from the Phase 1b raw analysis using the same
three layers. The dagger marks an invalid high-overhead diagnostic.

| Batch | Layer | Attention | Norm/router | Dispatch | Expert max | Combine drain | Full MoE | Expert fraction | Comm/expert |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.6011 | 0.8184 | 0.1844 | 1.6621 | 0.4491 | 0.7715 | 2.3690 | 17.19% | 5.4518x |
| 16 | 9.6168 | 2.1043 | 0.4745 | 3.4940 | 1.9950 | 2.7910 | 7.1516 | 21.55% | 3.1155x |
| 32† | 8.8516 | 2.3332 | 0.5084 | 1.5484 | 2.8914 | 2.5938 | 5.4869 | 32.99% | 1.4370x |

Batch 16 scaling versus Batch 1 was 3.697x for layer wall time, 2.102x for
dispatch, 4.443x for local expert, and 3.618x for combine drain. Expert compute
therefore became materially longer and communication-to-expert improved by
42.9%, but dispatch plus combine still consumed substantially more time than
the expert window.

At Batch 16, all three selected layers repeated the same pattern:

| Layer | Expert max | Expert fraction | Comm/expert | Existing oracle | Extended optimistic oracle |
|---:|---:|---:|---:|---:|---:|
| 12 | 2.0725 ms | 21.65% | 3.1589x | 1.1627x | 1.3593x |
| 24 | 2.0002 ms | 21.36% | 3.0864x | 1.1935x | 1.3443x |
| 36 | 1.9688 ms | 21.71% | 3.1581x | 1.1910x | 1.3459x |

## Expert timing interpretation

The earlier 0.447 ms result remains plausible for Batch 1, but it is the
boundary of the fused local-expert kernel, not dispatch, combine, or the full
MoE layer. The comparable Batch 1 three-layer median is 0.4491 ms. Under the
real Batch 16 DPEP workload the same boundary rises to 1.9950 ms (4.443x), which
confirms that larger batch creates a more substantial expert-compute window.

## Oracle and uncertainty

The existing timestamp oracle retains first-tile fill and the complete
DPEP-combine-through-TP-all-gather drain. The extended optimistic oracle also
allows additional expert/combine hiding while respecting measured prelude and
drain lower bounds.

| Batch | Existing selected-layer oracle | Extended optimistic oracle | Profiler overhead | Gate use |
|---:|---:|---:|---:|---|
| 1 | 1.0823x | not used | 8.86% | reference |
| 16 | 1.1750x | 1.3552x | 13.56% | valid |
| 32† | 1.2425x | 1.4233x | 61.81% | excluded |

Batch 16 clears the 1.15x oracle threshold, including all three selected
layers, but its nominal 17.50% existing-oracle gain exceeds measured profiler
overhead by only 3.94 percentage points. That is insufficient margin for the
GO requirement that expected gain clearly exceed profiler uncertainty.

## Blockers and decision

- No OOM or worker crash occurred.
- Batch 16 profiling is valid under the 20% overhead limit.
- Batch 32 profiling hit the immediate-stop condition at 61.81% overhead. No
  alternative Batch 32 configuration or retry was attempted.
- vLLM warned that no tuned H100 MoE configuration existed for the observed
  `E=32,N=768` shape, so the selected Triton backend used a default config.
- The Batch 32 stage trend is encouraging, but it cannot be used as evidence
  while its observer effect is this large.

Final decision: **HOLD**.

Batch 16 has expert latency above 1 ms, a lower communication-to-expert ratio,
and a >=1.15x oracle on all representative layers. It does not meet the 25%
expert-fraction GO threshold, and gain-versus-profiler margin is weak. Batch 32
would appear to meet the compute-window thresholds, but its stage data is
invalidated by profiling overhead. These facts match the specified HOLD cases,
not a defensible GO or NO-GO.

Single recommended next task: perform one same-configuration Batch 32
revalidation that records only the two demonstrated active scheduler offsets
and must keep profiler overhead below 20%; do not start Phase 2A unless that
measurement validates the stage trend.

## Artifacts and changes

Canonical result directory:
`poc_flashvep/results/batch16_32_quick_poc_20260804_131743/`

Final analyses:

- `batch16/analysis_final.json` (valid; uses adjacent
  `baseline_check_requests.json` and `profile_v3_requests.json`)
- `batch32/analysis.json` (invalid high-overhead diagnostic)

Minimal implementation changes:

- `poc_flashvep/scripts/phase1b_tp2dp2.py`: opt-in global-batch construction,
  balanced DP request distribution, and per-request output validation
- `poc_flashvep/flashvep/instrumentation_phase1b.py`: opt-in multi-offset
  capture and batched real/padding/idle workload accounting
- `poc_flashvep/scripts/run_batch16_32_quick_poc.sh`: fixed Batch 16/32 runner
- `poc_flashvep/scripts/analyze_batch16_32_quick_poc.py`: microbatch-aware,
  selected-layer stage/oracle analyzer

Earlier incomplete Batch 16 captures in the timestamped directory are
preserved for audit and excluded from the final metrics. No prior Phase 0/1/1b
artifact was overwritten.
