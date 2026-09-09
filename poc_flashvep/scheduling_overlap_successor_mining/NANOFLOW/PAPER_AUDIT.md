# NanoFlow paper audit

Source: https://www.usenix.org/system/files/osdi25-zhu-kan.pdf, OSDI 2025; technical body pp.2–15 read. Primary FP16 Llama2-70B/8×A100; additional Mixtral and dense models. Throughput-oriented, not universally latency-optimal.

Nano-batches break cross-operation batch dependencies; execution-unit allocation overlaps resource-complementary kernels. Two-stage MILP chooses structure then interference-aware resources. Async request management and optional KV offload contribute separately; overlap-only ablations are smaller than cross-framework headline gains.

Key assumptions for faithful transfer:

- Abundant requests and approximately stable dense token budget, with external autoscaling at low demand.
- Re-search is explicitly permitted after significant model/workload change. A never-replan strawman is invalid.
- Pairwise GEMM-memory/network interference is assumed transferable to three-way overlap; measured mapping variation was small in original evaluated shapes.
- Original MoE uses TP grouped GEMM, not EP. Qwen DeepEP traces cannot be labelled native NanoFlow EP execution.
- Extra weight reads must fit beneath compute slack; low-load regression is acknowledged, not a new failure.
- Async EOS retirement assumes long outputs; measure any short-output cost before promoting it.

No Qwen-VL full port before a measured direct headroom gate. Static/portfolio/per-regime oracles must include scheduling and plan-transition costs.
