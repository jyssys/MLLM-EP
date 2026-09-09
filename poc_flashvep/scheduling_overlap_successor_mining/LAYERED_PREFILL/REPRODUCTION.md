# Native reproduction in progress — 2026-09-09

Pinned official Qwen3-30B-A3B, BF16, physical GPUs 4/5, TP2 (not EP).
FlashAttention and native Layered extension builds completed. CPU-hidden import
checks passed for torch 2.8.0+cu128, nanovllm.ops and vllm_flash_attn.

## First live eager correctness/mechanism sanity

Independent native chunked512 and layered8192/group16 engines completed:
three identical four-request warmups, four correctness requests and 48 identical
steady real-content requests with the original arrival/output schedule.

- All requests complete; no client stream coalescing observed.
- Short arithmetic and capital answers agree with the HF reference.
- All four smoke outputs are exactly equal across the two scheduling modes.
- Longer outputs agree exactly on only 13/48 requests. This triggers numerical
  and same-policy restart controls. Do not accept timing as successor evidence.
- Descriptive mean E2E: chunked 13.4976 s, layered 12.9456 s (+4.09% reduction).
  This is one eager pair, not the paper's graph-enabled baseline or an oracle.

## Official graph-enabled controls

`graph_screen_20260909_v1`: three randomized independent restart blocks with
chunk512, layered4 and layered16; identical steady/bursty request sets. All nine
engines completed, yielding 864 measured requests. Existing group knobs are the
first trivial-fix control, not a method. No MLLM native port is claimed.

Median paired mean-request E2E reductions versus chunk512: cap4 bursty 19.665%,
cap4 steady 3.471%; cap16 bursty 3.011%, cap16 steady 3.586%. The third cap16
restart was substantially slower in both workloads; it is retained, not removed.
Long free continuations also differ across identical-policy restarts, whereas
short correctness and the first 32 output characters agree in the compared
steady requests. These are descriptive timings, not a quality-matched successor.

Source and runtime logs establish that `--num-stages` is a **selection cap**:
cap16 uses 1/2/4/8/16 according to token count; cap4 uses 1/2/4. It is not a fixed
group count per request. Source supports caps 4/12/16/24, not an invented cap8.
The native selection counts are saved in `native_scheduler_choices.json`.

Across the two tested caps, cap4 is best in both workload medians. The observed
best-static versus per-workload cap envelope is 0%, including leave-one-restart-
out selection. This is NOT an upper bound on untested partitions or VL serving.
Bursty cap4 improves TTFT while worsening maximum ITL: the joint SLO result
depends strongly on whether the TBT threshold is 50 or 100 ms. This is consistent
with the paper's acknowledged TTFT/TBT tradeoff, not new novelty by itself.

Next bounded controls: native eager layer/group and expert-use diagnostics;
full-workload warmup; remaining existing caps. Instrumented CUDA costs and
uninstrumented request performance remain separate.
