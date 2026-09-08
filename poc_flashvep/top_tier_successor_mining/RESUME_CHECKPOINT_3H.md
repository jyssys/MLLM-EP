# Resumed GPU work — 2026-09-08 14:04 KST

Physical GPUs4–7 only. No finalist/winner yet. No Kimi run or successor method.

## Completed

- Official MoDES1024/grid100 frontier, official-norm SERE HF quality and B16
  controls,32 captured VL inputs, native full48-layer text and corrected VL sanity.
- Six clean original-port EP conditions:ChartQA/GQA × B1/B4/B16,128 requests,
  three paired cohort repetitions,all six policies. These are six engines, not
  three independent engines per condition. All runtime proofs confirm actual
  DeepEP HT,TP2/DP2/EP4,BF16,eager,DBO/prefix cache off on4–7.
- MoDES port-cost falsification:official source computes modality masks once per
  model forward; our first port repeated this per layer. Correcting the lifetime
  preserves all96 paired output sequences per policy and improves request E2E by
  9.65%/9.55% versus the uncached70/85 ports. This is **not a new method**.
- Metadata-only profiler/control confirms small-M `isin` uses `unique`/stream
  synchronization. Standalone operator timings are not serving speedups.
- B16 scheduler/route diagnostic:actual decode membership median3.5–4,not16.
  Selected eight-layer/first-eight-step MoDES70 sample drops0.036% of decode
  assignments versus about78% in prefill. Sample scope includes warmup; do not
  extrapolate this fraction to all layers/requests.6,288 four-rank stage joins
  accepted;240 trailing incomplete joins excluded,not imputed.

## Strongest negative / validity correction

SERE B1 quality loss is real but largely removed by S4/rho and larger cohorts.
MoDES aggressive ChartQA loss is already in its original paper. Both facts make
a successor claim harder, not easier. The uncached MoDES latency penalty cannot
be used as evidence against the paper:it partly came from our port.

## Running / next

Fixed32-output B1/B16 and prefill-only MoDES controls are running sequentially.
Then rerun the corrected MoDES port on all six128-request conditions,execute
native Libra remaining VL groups plus an exact-current-route lookahead oracle,
and add a large real-text prefill control. Only then rank all three. Kimi and a
successor prototype remain conditional on material,nontrivial headroom.

## Accounting

Recorded completed intervals since11:03 KST:9.18 GPU-resident hours across4–7,
about2.29 hours per card. Includes model loading/CPU control and separately
marked port-debug runs; excludes the currently running job and unrecorded
aborts. This is not a CUDA-active-time measurement. Burn counted as research:0.
Live telemetry covers4–7 only. Main baseline environment remains unchanged.
