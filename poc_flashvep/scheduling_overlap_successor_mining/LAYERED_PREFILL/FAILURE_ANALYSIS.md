# Failure mining result

The proposed premise “one fixed group count” is inaccurate: the official
scheduler already selects the actual count from length-dependent tables. Equal
contiguous membership and shared active-cohort stage count remain assumptions,
but an assumption is not itself a performance failure.

The original mechanism direction is reproduced: cap4 versus chunk512 gives
19.67% median paired bursty E2E reduction, with an ITL tradeoff. This original
gain must not be relabelled as successor headroom. A broader, full-workload-warmed
cap4/cap12/cap24/chunk2048 screen gives only **0.657%** additional finite-policy
mean-E2E opportunity; held-out policy selection gives zero in all three blocks.
Additional attained SLO goodput is at most 0.0301% on the predeclared grid.

No material, non-trivial MLLM failure has been established. The 25–40% native
prefill group-makespan proxy is not a request saving. Long-output quality remains
bounded by short-answer sanity and restart diagnostics, not a benchmark study.

Current conclusion: the tested existing-knob successor branch is low-headroom;
untested arbitrary partitions and a native MLLM port are not ruled out.
