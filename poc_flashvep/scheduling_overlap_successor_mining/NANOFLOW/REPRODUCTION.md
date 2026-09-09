# Native NanoFlow reproduction and scope

Reference: official `dev-h100` revision
`915790ea862d1ddd52a8871282c1eb5be88f1391`, isolated Python3.10 / torch2.8-cu128,
FlashInfer0.3.1, native extension build. Actual supported Qwen1.5-MoE-A2.7B EP4,
FP16, checkpoint `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9`.
24 layers, 60 routed experts, top4, 15 experts per rank plus the shared branch.
Hidden states are replicated; native expert-output reduction uses NCCL. This
is not the project's DeepEP HT path or the original 8×A100 dense headline setup.

## Completed native sanity

Five original-entry smoke conditions (eager, graph, one-part plan, two-part eager,
two-part graph) match the independent HF reference's 82 output tokens. Scoped
compatibility edits include current KV layer-range constructor and preserving
the original activation function in `copy_nano`. Fixing sigmoid→SiLU substitution
is a port correctness repair, not a research contribution.

Same-prefix B4/B16, two independent engines per graph/split2 condition, then
independent HF full-prefix logits: numerical differences also occur within an
identical plan, and HF argmax disagreements are confined to near ties. See
`NUMERICAL_DIAGNOSIS.md`; no benchmark quality-equivalence claim.

Native splitter, expert copies, operation dependencies, selected stream IDs and
kernel tags are saved in each `.plan/` directory. CUDA/NVTX timeline confirms
split2 overlaps compute and resident NCCL kernels; `NSIGHT_MECHANISM.md` explains
why this did not improve the profiled B16 request step.

## Fidelity boundaries

- `nano_plan_adapter.py` exposes existing splitting/executor APIs; it does not
  implement a new scheduler or a searched-optimal plan. Only FFN is split.
- The native FusedMoE profiling/database hooks are missing. The dense-paper MILP
  cannot be fed a zero-cost expert placeholder and called a faithful MoE search.
- Manual 1/2/4-way plan comparisons and a native 112/16 green-context knob are
  bounded diagnostics. Category stream pools do not prove cross-category SM
  disjointness merely because requested counts add to 128.
- Initial decode graph capture/plan creation is charged to fixed-cohort E2E and
  separately excluded in steady-ITL statistics. A first-use cost is not assumed
  to recur in a production warmed-plan engine. Conversely, a warm-stage proxy
  is not reported as directly measured full-request improvement.
- Simultaneous cohorts are actual submitted requests, but not continuous-batching
  arrivals. Dynamic shape transitions and a full Qwen3-VL port remain unclaimed.
- The parent probe uses the native per-step barrier and receives each token
  before submitting the next step. It does not reproduce the paper's full
  asynchronous request-management or double-buffered lookahead contribution.
  Verified within-step operation overlap must not be expanded into a claim
  that every component of the paper's optimized serving pipeline is active.
- Pure-prefill first-use exposed missing initialization of plan-unlisted
  attention operations. Failed pilot timings are excluded; native default-tag
  initialization is CPU-tested and passed fresh GPU smoke plus the 18-engine
  M4096/M8192 three-plan screen. See PREFILL_PORTFOLIO.md.

Output trees are under `nanoflow_runs/`; all commands use the native launch
wrapper that forces physical GPUs 4–7. Build/load/JIT and numerical diagnostics
are not included as clean performance measurements.
