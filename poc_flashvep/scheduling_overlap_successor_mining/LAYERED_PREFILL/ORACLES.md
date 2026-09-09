# Layered grouping: restricted envelope versus request oracle

The native CLI's stage count selects a length-dependent table; it is not the
group count forced on every request. Actual selected values are recorded by
`analyze_scheduler_choices.py`. Unsupported cap8 is not silently treated as a
valid alternative. Official caps are 4/12/16/24.

For the completed cap4/cap16 graph screen (three independent engine pairs),
cap4 wins both measured workload median mean-E2E costs. Hence the per-workload
oracle adds 0% over best-static cap4. Leave-one-restart-out choice is also 0%.
This is restricted to two caps and current inputs, not all possible grouping.

Nine all-token SLO thresholds similarly give no per-workload versus static
advantage in this two-cap set: the best cap changes with the SLO itself, not
with the two tested arrival regimes. Cap16 is preferable at tight TBT50, cap4
at TBT100/200. This is the original documented TTFT/TBT tradeoff. A successor
must beat the best appropriate existing setting, not just chunk512.

Native eager and common Qwen3-VL measured-layer-cost partitions are labelled
minimax group-cost diagnostics. No stage maximum is directly substituted into
request E2E: a request still visits every group, decode interference and admission
can change, and event overhead is material. Direct native MLLM request gains
would require a faithful VL port; measured vLLM costs alone do not establish them.

Additional official caps and larger chunk/full-workload warmup controls are
prepared before final screening closure. No new online grouping method or
Kimi/prototype promotion is justified by the current restricted envelope.

## Native measured-cost partition diagnostic, September 9

`analyze_partition_proxy.py` reuses the exact contiguous minimax CPU routine on
22 complete 48-layer rank/phase/prefill-active profiles. No missing layer is
filled or inferred. Output is `mechanism_resume_20260909_v1/native_partition_proxy.csv`.
Decode group maxima do not improve in this diagnostic. Prefill group proxy
reductions reach 25–40%, while mixed cap4 profiles reach 13–14%; these numbers
are **not direct request E2E gains** and not even a portable clean graph cost
model. Conditional layer medians can change when the grouping changes its
batch composition. The independent FastPP static-partition control demonstrates
why treating such a minimax proxy as a request oracle is unsafe.

## Completed full-workload-warmup cap/chunk control

`full_warmup_cap_screen_20260909_v1` has twelve completed engines: cap4/12/24
and chunk2048, each three independent restarts, 1,152 measured requests total.
Same arrival traces and full workload warmup precede each measurement; no
mechanism hooks, graphs enabled. Cap4 versus chunk2048 has median paired mean-E2E
reduction +1.15% bursty and -1.62% steady. A slow third chunk restart gives wide
restart-bootstrap intervals (-1.30..31.79% bursty, -6.21..39.54% steady); it is
retained, not selected as a large-win headline. Cap24 is worse in every pair.

The four-existing-knob, median-cost lower envelope gives **0.657%** additional
request E2E over best-static cap4. Leave-one-restart-out selection gives 0% in
all three held-out blocks. Across all nine predeclared TTFT/all-token-TBT SLOs,
additional attained-goodput envelope is at most **0.0301%**. Tight TBT favors
cap12; looser TBT favors cap4 or chunk2048. This is a change with the SLO target,
not substantial workload-dependent successor headroom. These finite envelopes
do not rule out all possible partitions or a native MLLM successor.
