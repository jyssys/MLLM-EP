# Vision-token heterogeneous Top-K PoC

This workspace evaluates whether removing low-importance visual-token expert
assignments from the currently overloaded EP rank improves the critical path
beyond semantic-only Top-K reduction at the same assignment budget.

The working contract is the externally supplied specification at
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/bottleneck_reproduction_vision_heterogeneous_topk_poc_spec.md`.
Its SHA-256 and the source repository revision are recorded in each result
manifest.  The existing `poc_flashvep/` tree is treated as read-only evidence.

Primary commands are documented in `EXPERIMENT_LOG.md`.  GPU commands refuse
to run unless `CUDA_VISIBLE_DEVICES=4,5,6,7`.

## Outcome

Final decision: **NO-GO** for EP-tail-aware vision heterogeneous top-k.

The bottleneck was reproduced: at 8K/16K, expert compute accounts for
35.8--39.7% of clean TTFT and the count-proportional perfect balance upper
bound is 9.5--10.6%. At the same vision-assignment budget, the tail-aware
oracle reduces max-rank assignments 2.40x more than semantic-only selection
at 10% removal. That positive geometry does not survive the joint quality and
economic gate: across three replay contents the median projected TTFT gain is
6.94% at 10% removal with 13.16% worst-rank local MoE relative-L2. A 20%
budget reaches only 8.30% projected TTFT with 19.69% relative-L2.
An all-48-layer nine-image correctness diagnostic further found that a 10%
tail-aware budget preserved the first token but only 6/9 four-token greedy
sequences, versus 9/9 for semantic-only removal at the same budget.

Start with:

- `reports/bottleneck_reproduction.md`
- `reports/vision_heterogeneous_topk.md`
- `reports/final_decision.md`
- `results/final_analysis/`
