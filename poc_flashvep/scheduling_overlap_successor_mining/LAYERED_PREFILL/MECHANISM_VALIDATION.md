# Native Qwen3 mechanism, 2026-09-09

Three separate eager diagnostic engines, 48 steady + 48 bursty requests each:
chunk512, official layered cap4, official layered cap16. All 48 layers timed;
rank-zero expert histograms sampled every 16 invocations. Exact request headers
propagate into native sequence IDs. Same-device stack durations join across the
two physical GPUs 4/5; no subtraction of cross-device absolute CUDA timestamps.

| Diagnostic | complete two-rank stacks | sampled mixed-layer routes | with active prefill | median extra experts needed only by prefill, when active |
|---|---:|---:|---:|---:|
| chunk512 | 1,219 | 2,400 | 2,400 | 42 |
| layered cap4 | 1,156 | 816 | 240 | 64 |
| layered cap16 | 1,217 | 2,448 | 171 | 67 |

Layered execution genuinely suppresses prefill work in inactive layer groups.
The fewer active-prefill observations have larger routed working sets, as
expected when prefill work is concentrated. This verifies the original method's
mechanism direction; it is not a new successor failure. Counting all mixed
layers without the active-prefill condition would misleadingly report zero
median prefill-exclusive experts for layered execution.

Each captured expert histogram conserves M × top-k assignments. Logical routed
weight-use bytes are a proxy, not measured HBM traffic. Capture is bounded to
1,500 invocations, so absolute route counts across configurations are not full-run
traffic totals. Routing-copy steps are excluded from unsampled layer profiles.

Eager per-layer event overhead substantially changes stack latency. The graph
request screen is the performance source, not these 60–70 ms instrumented spans.
One incomplete cap4 stack is excluded; no duplicate TP rank is counted twice.

Artifacts: `layered_runs/mechanism_resume_20260909_v1/`, including raw JSONL,
`sampled_expert_use.csv`, `joined_stacks.csv`, `layer_profiles.csv`, summaries.
