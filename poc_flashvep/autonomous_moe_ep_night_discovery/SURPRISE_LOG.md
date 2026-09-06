# Surprise log

## S00 — the observer may be the first confound

The inherited online hook synchronizes a CUDA event and copies top-k IDs to
CPU on every MoE layer.  The stock HT path deliberately returns asynchronous
DeepEP events.  Before searching for a runtime phenomenon, this instrumentation
assumption must be falsified.

## S01 — eager DP still has a per-step CPU phase rendezvous

The runner always calls `coordinate_batch_across_dp` for DP>1. With async
scheduling vLLM intentionally chooses the CPU process group, so eager/no-DBO
avoids DP padding but does not avoid the collective. This is a hidden global
coordination assumption and generated H20/H31/H32/H35.

## S02 — MoE metadata is reconstructed through host memory 48 times per step

DeepEP returns a Python expert-count list; vLLM makes a fresh CPU int32 tensor
and copies it to the GPU at every layer. It may be benign, but it is a
high-frequency source-derived candidate (H33), unlike another routing-shape
scalar.

## S03 — generic and MoE sampled tails strongly co-occur

In the first fresh H03 trace, 3.55% of logical steps are simultaneously above
the p95 for sampled attention and sampled MoE time, versus 0.25% under
independence (14.2x enrichment). The rank correlation is only 0.27. This
weakens an EP-specific tail story and motivates matched whole-step controls;
its direct request mass is not yet established.

## S04 — H09 common latency-regime transition

While the live randomized H09 chunk-residue control was still running, both
variants moved from roughly 683 ms request medians to 450–480 ms after about
300 pairs. The variants moved together, so this is not evidence for chunk
residue; it is a fresh synchronized state transition (allocator/KV/scheduler
or GPU clock regime candidate) with potentially large latency mass.

## S05 — H14 common decode-state expansion

Near the end of the live H14 long-context block, both randomized variants
expanded from roughly 9.0 s to roughly 13.6 s within adjacent waves. The
variants again moved together, so this is not a long-context A/B effect. It is
additional evidence for a serving-state regime transition and motivates
segment-level postmortem on host gaps, step cadence and sampled stage timing.

## S06 — metadata allocation is small but has rare host outliers

The fresh metadata observer captured 287,219 calls. Its typical host cost is
about 0.11 ms (p99 about 0.15 ms), but a small number reached 13.9 ms. This
supports a high-frequency tax candidate (H33/H38) without implying that the
median tax is a material E2E optimization opportunity; the outlier/host-gap
association is the causal follow-up.

## S07 — dummy work was real but not expensive at the request level

The corrected fresh-worker hook showed that an apparently idle DP group does
not disappear: its two workers executed 6,399 explicit dummy-batch RPCs in
H36. However, the paired useful-request wave changed by only -0.26%. This
separates **observability of participation** from **a user-visible latency
opportunity** and closes the simplest idle-DP explanation for the earlier
wave-only anomaly.
