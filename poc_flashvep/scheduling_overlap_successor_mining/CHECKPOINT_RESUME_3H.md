# Resumption checkpoint — 2026-09-09 13:27 KST

Approximately three hours since GPUs became available again at 10:27. This is
not three hours of CUDA busy time. The user explicitly asked to keep continuing.
Only physical 4–7; native Layered uses TP2 on 4/5. Burn never overlaps experiments.

## Completed / strongest observations

- Layered: 21 uninstrumented graph engines (two screens), 2,016 measured
  requests, plus three eager mechanism diagnostics (288 requests). The new
  four-existing-knob/full-workload-warmup envelope adds only 0.657% mean E2E and
  at most 0.0301% fixed-trace attained goodput over the best static setting.
  Restart variation is retained; no 30–40% lucky-block headline.
- FastPP: 18 dense baseline restarts, nine native Qwen3 restarts and six actual
  partition-control engines. The measured-cost partition loses E2E in all three
  pairs despite a positive stage proxy. Native PP4 is not EP4.
- NanoFlow: actual overlap is verified but no clean positive plan result yet.
  New M8192 resource-control same-prefix/HF comparison localizes its one output
  mismatch to a zero-gap HF tie. Mismatching performance remains excluded.
  Native final logits really have shape 8192×151936 for four next-token outputs;
  this is a trivial baseline implementation concern, not a novel successor.
- Qwen3-VL: earlier six clean/lite-observer engines yielded 1,248 measured
  requests and 14,399 measured four-rank logical MoE joins. Clean measurements
  remain the E2E reference. New four-real-image/natural-text matched inputs are
  ready; warmup-selector and identity-join CPU regressions pass.

## Accounting

Whole-study recorded interval union: **7,386.38 s (123.11 min)**;
**7.4292 GPU-hours**, **1.8573 four-GPU-equivalent hours**. This includes prior
authorized study measurements before the overnight pause, not just this resume.
Model load/JIT, burn, failed startups and uncovered pilots are excluded.
Per-device union avoids counting Layered's two-device runs as four-device runs.

## Active and next

1. Native NanoFlow plain/split2/split4 large-prefill portfolio: 18 fresh engines,
   three restarts each at M4096/M8192; 15 complete at this checkpoint.
2. Queued six-engine clean/lite Qwen3-VL larger matched-image control after it.
3. FastPP full-workload warmup with PP-only/static chunks/greedy/ALP, three
   randomized restarts and identical KV cap.
4. Bounded clean decode plan portfolio if needed to close the original manual
   pilot's repetition gap; no native full Qwen-VL NanoFlow port.
5. Equal-milestone review and honest ranking, distinguishing unsupported native
   ports/full search from measured method failure. No finalist gate is met yet;
   Kimi and successor prototype are not justified at this point.
