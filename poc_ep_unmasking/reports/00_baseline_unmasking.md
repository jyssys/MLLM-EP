# 00 — Stock decoder and EP4 baseline

Working contract: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/training_free_ep_aware_unmasking_poc_spec.md`.
The attached spec SHA256 is
`4fc7e597932413d467187d3f5b1a4d8bc7cf8f72fe1dc9103abaa0a3d6f51d5a`.

The downloaded model is `/home/esjung/models/LLaDA2.0-flash-744c3f8`;
`config.json` SHA256 is
`ac35e9dc8313f49b2600da2a8b3ec31c53c01d22a5943ec26c8e21f8e208ab07`.
The external dInfer checkout has HEAD `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`
plus existing local substrate changes. This PoC did not modify that checkout.

All launches used `CUDA_VISIBLE_DEVICES=4,5,6,7`; benchmark logical
0/1/2/3 mapped to physical 4/5/6/7. Their UUIDs, process inventory, free
memory, NV18 links, and logical mapping are saved per run in `results/`.
Before each model launch no other compute process occupied these GPUs.

The current bridge is dense TP4/routed EP4, DP1/PP1, BF16, normal DeepEP.
Each rank owns 64 contiguous experts of the 256 routed experts. Source token
rows are partitioned across four ranks before top-k=8 routing; DeepEP dispatch
sends branches to owner ranks, the existing fused expert executes them, reverse
combine returns them, and an exact gather restores the dense TP view. A
representative measured layer-16 invocation had 224 global rows/56 source rows
per rank and owner receives `[196,153,216,157]`; rank 0 sent assignments
`[121,57,180,90]` to ranks 0–3, proving remote dispatch rather than TP-only
execution. Shared expert computation remains source-local.

The stock threshold decoder at temperature 0 computes selected-token softmax
confidence. At threshold 0.9 it finalizes all eligible masked positions above
0.9; when none qualify it lowers the effective cutoff to just below the
maximum so at least one position is finalized. It excludes predictions equal
to the MASK ID. Transfer count is therefore usually one, not a fixed
`steps=B` schedule. The generated sequence uses block length 32, prefix KV
cache, and generation budget 128.

For the bounded eight-request control, submitted=8/mini=8 is the largest
feasible single wave. Earlier large-pool work found larger mini sizes favorable;
these numbers must not be represented as a submitted-32/mini-32 serving result.

| Task | Clean BCT | NFE | Bounded score | Observer-heavy status |
| --- | ---: | ---: | ---: | --- |
| GSM8K 8 | 7.4620 s | 106 | 5/8 | 9.7159 s; exact same answers/lengths, +30.2% timing overhead |
| HumanEval 8 | 5.7934 s | 55 | 2/8 | partial route/confidence trace; compiled reduction failed at iteration 50 |

The HumanEval observer failure was an Inductor `invalid configuration argument`
after a dynamic-size MoE reduction. A stock clean rerun succeeded, so the
partial HumanEval trace is usable for descriptive slack only; it is not a
complete trajectory or benchmark-quality result. Teardown warnings after
successful clean runs are not performance failures.
