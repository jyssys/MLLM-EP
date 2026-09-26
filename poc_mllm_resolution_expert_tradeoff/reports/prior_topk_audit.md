# Closest-prior boundary audit

This run does not claim that visual expert-top-k reduction is new.

| Prior | Relevant mechanism | Status in this run |
|---|---|---|
| FastMMoE | Reduces visual `K_v` and renormalizes retained weights; also contains a separate token-pruning component | Uniform visual-k component is represented by C2 and labeled `REIMPLEMENTED_COMPONENT_CONTROL`, not a full paper reproduction |
| MoDES | Skips actual experts within the routed candidate set using modality/layer policy | Boundary audited; full policy not ported and is `NOT_RUN` |
| MACS | Capacity control, overflow reroute/drop and local expansion change executed expert work | Boundary audited; not a direct equal-budget baseline here |
| ReaLB | Primarily adjusts EP-rank precision | Related EP work, not a direct k baseline |

The novel question under test is narrower: whether jointly allocating image
sampling density and visual expert intensity using measured EP critical-path
cost selects a materially better point than resolution-only, expert-only,
independent two-axis, or total-pair cost controls.  If the EP-aware selector
does not beat those controls, the result is reported as generic compression,
even if absolute latency improves.

Primary sources audited on 2026-09-26 are
[FastMMoE v2](https://arxiv.org/abs/2511.17885),
[MoDES v2](https://arxiv.org/abs/2511.15690),
[MACS v3](https://arxiv.org/abs/2605.05225), and
[ReaLB v3](https://arxiv.org/abs/2604.19503). Repository/commit compatibility
and any missing official implementation are reported rather than filled with
fabricated results.

The official FastMMoE repository was pinned at commit
`b355b15a46601be24e440e1449df82de08e3684d`.  Its published code supports
InternVL3.5 and DeepSeek-VL2, not Qwen3-VL-30B-A3B; it also couples expert
activation reduction with its own layer schedule and token pruning.  Directly
running it would change checkpoint/runtime/protocol, so C6 is intentionally the
isolated uniform visual-k component under the common Qwen baseline.  A full
FastMMoE or MoDES port is `NOT_RUN`, not silently scored as this component.
