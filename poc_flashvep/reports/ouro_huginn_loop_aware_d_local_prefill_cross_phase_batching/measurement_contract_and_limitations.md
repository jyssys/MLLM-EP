# Measurement contract and limits

## Serving and exactness

Marker-free runs supply the primary serving latency and goodput values.
State-hash runs qualify outputs, valid prompt KV planes, and prompt-last hidden states.
State-hash latency is not clean serving latency.
Native reference parity does not establish original HF equivalence.

N32 submits 32 requests together. GPU capacity can admit fewer requests at once.
Huginn retains its full physical per-depth KV allocation.
Unused physical planes do not create additional admission credit.

The workload preserves the existing ShareGPT and LongBench controlled prompt pool.
Output caps and `ignore_eos=True` force the reported output lengths.
These outputs are not a natural EOS distribution.
Prefix caching stays disabled.
The campaign does not establish generalization to new prompt text.

## Placement and scheduling

A D-local request never starts prefill on P.
It allocates local prompt KV and completes both prefill and decode on D.
Its input still reaches D through the host control path.
The absence of intermediate P-to-D KV does not mean zero input transport.

The fixed subset places request indices 0, 4, 8, and subsequent multiples of four on D.
Both generic serving arms use that subset.
The local active limit is one in the clean placement controls.
Blocked local requests retain local ownership.
This control is not the official Kairos placement algorithm.
It is not a calibrated deadline-prediction policy.

Ready decode has priority in the separate-execution controls.
Already submitted prefill work can still delay newly ready decode.
Priority at submission does not preempt a GPU sequence.
The optional output-credit control limits submissions between observed output steps.
That control does not guarantee each stream's hard deadline.

## Warm-up fairness

V5 is the primary generic serving comparison.
Native and V5 enable the same full-context decode-geometry warm-up.
They retain the common eight-entry graph budget.
Local paths also warm their actual local prompt geometry.
Warm-up requests do not enter serving statistics.

V4 disabled the additional decode-geometry warm-up in D-local arms.
Those results remain tuning diagnostics.
They must not replace the warm-matched primary comparison.

## GPU evidence

CUDA-event intervals enclose model-stream work and possible gaps.
They are not exclusive kernel sums.
Kernel-free intervals do not establish deadline-safe insertion windows.
DMA and other device resources can remain active.

The collector self-test checks timestamp coverage, dropped records, and output integrity.
It does not qualify serving overhead.
Each serving trace needs a separate local-tail interference check.
An overall response span within 5% does not qualify a locally distorted decode tail.
Unqualified traces remain diagnostic evidence only.

SM active cycles, achieved bandwidth, and actual weight-read bytes remain unmeasured.
Kernel names and counts come from a separate profiled microbenchmark.
Its profiled durations do not replace clean serving measurements.
Packing and attention enter the full-core event enclosure.
Their isolated critical-path costs remain unmeasured.

## Batching controls

Full-core primitives start from legal ready recurrent states.
They compare the resulting hidden rows and written query KV bytes.
They do not hash every historical KV byte after every replay.
Whole-request D-local state qualification is a separate test.
Cross-depth online N32 qualification must not be inferred from the generic N32 test.

The dense same-depth and cross-depth labels replay the identical input and graph.
Depth is not an argument to the dense projection.
Timing differences between those labels do not establish an arithmetic benefit.
Different-depth readiness can still change scheduling opportunities.
The closed-loop control measures that separate question.

The qualified numerical patch retains 256-row tiles for sensitive projections.
Ouro uses the patched down projections.
Huginn also uses the patched adapter.
The patch adds work for small inputs.
Batching can amortize that work.
This runtime effect must not be attributed only to recurrent weight sharing.
All compared arms retain the patch.

The initial closed-loop control uses two decode streams and one new local prompt.
It has no open-loop arrival rate.
Its cohort time is not PD throughput or sustainable serving capacity.
Three warm repetitions are not three independent model starts.
Independent starts require separate source and process receipts.

## Attribution and novelty

Native RLT already supports mixed-phase, mixed-depth recurrent batches.
The PoC reuses that implementation.
Generic same-depth batching can provide most of the measured batching benefit.
Broader depth eligibility is also a CDB mechanism.
Combining existing deflection and CDB mechanisms does not establish a new contribution.

ExactLoop optimizations are not applied in this campaign.
The campaign does not establish complementary headroom after those optimizations.
It also does not compare this model with an untied AR architecture.
No structural impossibility claim follows from a failed bounded PoC.
