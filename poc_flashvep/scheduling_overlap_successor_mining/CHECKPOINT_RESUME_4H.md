# Resumption checkpoint — 2026-09-09 14:27 KST

Four hours since the authorized physical GPUs 4–7 became available at 10:27.
This is elapsed resumption time, not CUDA busy-time. No burn overlaps research.

## Completed evidence

- Layered full-workload-warmup existing-cap/chunk screen: twelve engines,
  0.657% additional finite-policy E2E envelope, held-out choice zero, maximum
  additional attained SLO goodput 0.0301%. Arbitrary partitions not bounded.
- NanoFlow large-prefill portfolio: eighteen engines, all twelve paired first
  tokens agree. Plain wins both M4096 and M8192; finite envelope zero. Native
  MoE auto-search remains a source-supported port boundary, not method failure.
- Large real-image vLLM transfer: six engines / 480 requests; 120 exact actual
  prompt-token visual/text pairs. Common-state observer effects range widely,
  so these stage traces are not clean E2E or modality-causal evidence.
- FastPP native static-partition causal control: all three pairs regress E2E,
  despite the positive stage-max proxy. Source/quality limitations remain explicit.
- Twelve FastPP CPU tests, five COMMON tests, NanoFlow lifecycle and actual
  activation-copy/parent-identity checks pass; all study Python files compile.
  Runtime metadata, source pins, patch hashes and authorized GPU UUIDs saved.

## Active and next

FastPP final five-knob/full-workload-warmup screen has **12/15 complete engines**
at the accounting snapshot. It still runs on all four authorized GPUs. NanoFlow
B4/B64/B256 four-plan decode screen waits for its completion. CPU analysis of
each screen is queued behind its completed manifest; no partial oracle is
reported as final. Finish correctness-gated comparisons before the ranking.

No candidate passes the material non-trivial successor gate yet. Do not run Kimi,
design a successor around existing config tuning, or fill a paper Introduction
with an unmeasured MLLM failure. Equal screening documents now exist for all
three; limitations are not silently relabelled PASS.

## Accounting at 14:26:29

Whole-study derived request rows: **29,000**. Conservative recorded live-interval
union: **8,618.33 s (143.64 min)**; **8.7981 GPU-hours**;
**2.1995 four-GPU-equivalent hours**. These include pre-pause authorized study
measurements, not just the current four-hour resume. Model loading, build/JIT,
burn, failed startups and some unrecorded diagnostics are excluded. Native
Layered two-GPU time is not multiplied by four. This is not hardware busy-time.

## Strongest remaining uncertainty

Native full VL/PP×EP transfer and NanoFlow searched-optimal MoE plans are not
implemented. The successful alternate native/trace diagnostics do not justify a
universal three-paper NO_GO. If the remaining bounded controls remain negative,
final classification must preserve this partial-port coverage rather than
claiming an exhaustive impossibility result.
