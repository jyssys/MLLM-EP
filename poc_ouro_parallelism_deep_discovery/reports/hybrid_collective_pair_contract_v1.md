# Native custom-AR versus stock AG/RS: physical pair contract

Status: prelaunch contract. This is a bounded component falsifier, not an architecture implementation.

## Changed assumption

The older C02 fixture uses a remote complete MLP and NCCL All-Reduce.
It does not measure the source-owned attention/TP-FFN refinement.
That refinement has a direct generic prior and an explicit [ownership contract](dp_attention_tp4_ffn_refinement_v1.md).
This new probe checks one missing cost: the proposed collective substitution against vLLM's enabled custom All-Reduce.
It changes no candidate card or model path.

## Frozen scope

- Physical P2 uses released GPUs4/5. Physical P4 requires released GPUs4/5/6/7.
- Only verified owned burners can stop. The ongoing2.6B quality arms must finish first for P4.
- Hidden width is2048, BF16, with useful rows C=1,2,4,8,32,128.
- The two Base configs must both confirm the hidden width. No model weights load.
- Each arm uses seven warmups,16pairs per captured graph, and31trials.
- Trial order alternates between the two arms.
- Every rank participates, including inactive row owners.
- Equal-count padding remains in all tensors and timings.
- The outer guard bounds each P2/P4 probe to1800seconds.

## Compared work

CUSTOM_AR_PAIR executes two dependent vLLM CustomAllreduce calls.
STOCK_AG_RS_PAIR executes NCCL All-Gather followed by Reduce-Scatter.
The latter allocates `ceil(C/P)` rows per owner and retains padded rows.
The source code records every input/output tensor capacity.
Those capacities are not measured link bytes.

The two fixtures do not represent identical complete mathematical layers.
They omit the attention, MLP, norms, gates, packing, and scheduling between the collective boundaries.
The comparison tests communication service only.
It cannot validate a replacement's model output or task quality.

## Functional checks

Before timing, every rank checks integer-valued BF16 payloads.
The expected All-Reduce sums, gathered row order, and Reduce-Scatter output are exact in this fixture.
The native fast path must remain enabled. A fallback or disabled path fails admission.
Both allocator environment aliases remain job-local, with `expandable_segments:False`.
The worker checks for a prior Torch import before setting them.
Existing native graph-buffer registration uses the inspected CustomAllreduce API.

## Timing and interpretation

Each trial records the complete pair on a CUDA stream.
The reported value selects that trial's maximum rank-local duration.
It does not add independent component maxima or claim synchronized global timestamps.
Host time remains a separate diagnostic.
The fixture does not use Nsight or hardware-counter profiling.

A slower stock AG/RS pair can reject a free-communication premise for that named implementation and shape.
It cannot reject all fused hybrids, changed packet batching, or a different physical ownership optimum.
A faster pair is only a component opportunity. It is not an E2E gain or a recurrence-specific mechanism.
No model engine follows from either result alone.

## Launch gate

Require CPU layout tests and an independent source review before GPU execution.
Use the existing campaign process guard and fresh output names.
Preserve any failed launch, correctness check, or partial timing record.
Each rank retains available CPU timing rows after an ordinary exception.
Failure logging makes no new CUDA query. Process-group setup also enters cleanup protection.
The group timeout is120seconds. Outer-guard completion remains authoritative.
Do not change the protocol after a result.
