# Adaptive Recurrence-Cut PD: Dynamic Loop vs Token vs Logical-Layer Migration

**PoC specification — evidence-first, fixed-depth, training-free**  
**Date:** 2026-10-30  
**Working method name:** ARS-PD / Dynamic A  
**Project:** `/home/esjung/MLLM-EP-github`  
**Report destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/`  
**GPU budget:** physical GPU IDs 0–7, total 8 GPUs available; compare *equal budgets* of 2, 4, and 8.

## 0. Executive research question and falsifiability

**Q1 (adaptation):** When prefill pressure, decode pressure, and SLO slack change, does the best *recurrent-loop cut* change enough to give an online Dynamic A policy meaningful headroom above the best static cut?

**Q2 (axis):** Under matched token/loop work, hardware, scheduler, and correctness contracts, what are the differences among **token-prefix migration**, **recurrent-loop migration**, and **generic logical-layer migration** in (a) compute granularity, (b) KV/activation transfer, (c) decode interference, and (d) online SLO-goodput?

**Q3 (loop-specific advantage):** Is there an advantage due to *shared recurrent weights, regular repetition and depth-indexed state*, rather than merely being able to cut at an arbitrary graph/layer boundary? A general logical-layer cut is a superset of loop-boundary cuts if all cut points and costs are identical. **Never claim loop cuts are mathematically more flexible than arbitrary logical-layer cuts.** An advantage would need lower implementation/control overhead, more effective bounded choices, better state management, or a different deployment/memory trade-off, with measured evidence.

This study is not a new co-batching paper. Keep **Native Mixed / cross-depth GEMM co-batching OFF** in all primary arms. Diagnose and disclose any unavoidable scheduler differences; do not claim this baseline matches the full Sarathi-Serve optimization.

## 1. Verified project inheritance

- Frozen previous A/B/C PoC commit: `b82eabd20a4ec168a5ad9a119c8401f97a565e78`.
- Frozen B SLO campaign commit: `fea20edef840ed357ee0493c2f0dd69668c66c1f`.
- Verified numerical correction release used in RCC Stage 1: `3b7a722898dab5f6dd797b474c1ff461659d8fa1`; confirm actual deployed runtime and the newest authoritative report before any run.
- RCC Stage 1 commit: `b53a8fed36a1706ac74754b53753f20edb8b913e`.
- Existing A split implementation: `poc_flashvep/looped_pd_abc/split.py` (`SplitConnector`). The original A uses a fixed cut per launched run and **rejects prompts longer than its entire configured single chunk**. `A_sep` is recurrence split plus separate phases, not Native Mixed; RCC Stage 1 confirmed A + Native Mixed already physically co-batches cross-depth rows.
- Reference engine (read-only): `/home/esjung/MLLM-EP-github/poc_flashvep/ref_loopserve`.
- Private active RLT checkout: `/home/esjung/external/looped-pd-abc-rlt`.
- Preserve original source/weights, logs, raw data, Git dirty/staged state and CUDA numerical corrections. Do not silently revert a qualified patch.

**Mandatory correctness precondition:** Before GPU studies, read the *final* numerical-diagnosis report, identify its pinned source hashes and unresolved fidelity limits, and rerun small matched-runtime control checks. Ouro and Huginn qualify independently. A `CORRECTNESS_READY` internal matched-runtime gate is not proof of exact original-HF outputs or downstream benchmark accuracy. Any new migration path needs fresh token/state/hidden/KV validation.

## 2. Models, hardware, and semantics

| Model | Pinned public source | Recurrent fixed depth | Context limit | Notes |
|---|---|---:|---:|---|
| Ouro Base 1.4B | ByteDance/Ouro-1.4B | R4 | 65,536 | 24 physical recurrent layers; original BF16 checkpoint |
| Huginn-0125 | tomg-group-umd/huginn-0125 | R32 | 4,096 | 4 recurrent physical layers plus 2 prelude and 2 coda; preserve injection state |

Obtain exact HF revision hashes from `poc_flashvep/looped_pd_abc/common.py`; do not infer or update revisions. Fixed R32 is the **primary Huginn configuration**, not R16. No dynamic exit, dropping FLOPs, model approximation, retraining or weight changes. BF16 and the qualified runtime numerical path remain fixed across all methods.

Huginn with max output 512 admits at most **3584 prompt tokens**, subject to exact tokenizer and context accounting. For output targets 384 and 448, maximum prompts are 3712 and 3648, respectively. A separate common-support subset must contain examples valid for *both* models; optionally report the extra 3585–4096 prompt-length cells as **Ouro-only**, never silently truncate.

## 3. Precisely define three migration axes

Represent a fixed-depth prefill as a valid causal/recurrent computation DAG over **token position × recurrent depth × physical transformer sublayer**. A cut is legal only if its consumer-visible inputs and valid KV/state are materialized, transferred, and acknowledged.

### 3.1 Token-prefix migration (`Token-Static`, `Token-Dynamic`)

Choose a prefix length `c` at token-chunk boundaries. P finishes **all R loops** for prefix tokens `0..c-1`, transfers the valid all-depth prefix KV and required per-request metadata, and D computes all R loops for remaining prompt tokens `c..L-1` before subsequent full-depth decoding. This is **not** ordinary chunked prefill entirely on P: it is actual P→D **prefill-compute migration**. Preserve causal dependencies and depth-specific KV. Previous prefix tokens' *intermediate prompt hidden rows* are normally not needed after their all-depth computation if the suffix remains, but final logits/last-token hidden and model-specific state may require additional transfer; instrument all such bytes, never assume zero hidden bytes categorically.

For `c=0` D performs full prefill (decode-local prefill). For `c=L`, P does full prefill (`PD_G1`). Support a real prefix/suffix continuation protocol; do not fabricate Token-Dynamic by mere routing or labeling. Frozen chunk sizes: initially `{128, 256, 512}`, with a 1024 diagnostic if viable; optimize without changing the method's claimed comparison semantics.

### 3.2 Recurrent-loop migration (`Loop-Static`, `Loop-Dynamic`)

Choose cut `n∈{0,...,R}`. P computes **all prompt tokens** through recurrent depth `n`; D resumes remaining `R−n` loops on the same prompt, then decodes all subsequent tokens at depth R. Transfer semantically valid KV below the cut **and the intermediate hidden states for all prompt positions**, plus required injection, RNG, position, cache-frontier and prelude/coda state. These are the original A semantics. Endpoint `n=0` = decode-local prefill, `n=R` = full-prefill PD. Do not count loops n/R as inference FLOP reductions; total prescribed work stays identical. **Dynamic** means per-request or per-safe-boundary selection using only online-visible P/D load, KV memory and TTFT/TPOT SLO slack, not an offline cut fixed across the entire workload and not model early exit.

### 3.3 Generic logical-layer migration (`Layer-Static`, `Layer-Dynamic`)

Define the **logical unfolded model graph** explicitly, including Ouro's physical sublayers repeated over R4 and Huginn prelude/core/coda and injection dependencies. A generic layer cut chooses a legal **transformer sublayer boundary**, possibly inside a recurrent iteration, not merely an integer loop boundary. P and D each retain the qualified identical full-model weights (matched to Loop mode); transfer the correct intermediate activation for *every affected token*, plus all semantically valid KV/state and metadata. A generic layer cut at a loop boundary **must produce identical semantics and essentially the same migration protocol as Loop Split**; make this an invariant/test. If unrolled-layer execution or partial intra-core transfer is not implementable correctly, report `NOT_IMPLEMENTED` and a separately measured/validated lower bound; **do not substitute a conventional static PP deployment or a dummy duplicate Loop arm and call it Layer-Dynamic**.

Interpretation: the Layer policy has a weakly larger search space than Loop if cuts/costs are otherwise identical. Therefore *Loop-Dynamic cannot outperform the exact optimal Layer-Dynamic oracle under identical costs*. Investigate practical implementation/serialization/batching/metadata overhead and restricted policy complexity as the actual potential distinguishing advantages. An AR model reference, if added, is diagnostic and must not be used for a cross-architecture latency superiority claim.

### 3.4 Work-preserving semantic conditions

Identical prompt IDs, model weights, fixed depth, termination policy, scheduler mode, numerical patch, GPU pair, P:D layout and workload arrival trace. Decode tokens always execute all R loops. Apply the current numerical determinism fix to all arms, with new online-batching parity tests. Separate `producer byte identity` from `cross-arm mathematical equivalence`. Record transfer cost and KV ownership even when exposed handoff appears small.

## 4. Required methods and fairness

**Exactly eight mandatory baselines plus one candidate** (nine arms). No Native Mixed GEMM co-batching on any primary arm.

| Arm | Definition | Remarks |
|---|---|---|
| `DP` | N independent colocated full-model replicas | Same GPU budget. No inter-GPU KV. |
| `DP_Chunked` | DP with token-chunked prefill, *separate execution* of prefill/decode work; interleave at scheduling boundaries but **no fused prefill+decode GEMM** | Clearly weaker than full Sarathi-Serve co-batching, and not equivalent to actual token-prefix migration. |
| `PD_G1` | Standard 1P1D or pooled P:D, all prompt loops on P, decode on D | Chunking/wavefront settings matched where possible. |
| `Loop_Static` | Best frozen recurrent cut per workload and model | Tune on disjoint calibration, freeze on test. |
| `Token_Static` | Best frozen token-prefix cut | Same independent calibration protocol. |
| `Token_Dynamic` | Online token-prefix cut selected from P/D load and SLO slack | Actual cross-GPU prefix/suffix continuation. |
| `Layer_Static` | Best frozen legal unfolded logical-layer cut | Include intra-loop cuts. |
| `Layer_Dynamic` | Online legal unfolded logical-layer cut | Correct state-transfer implementation mandatory. |
| `Loop_Dynamic` | **Our candidate: online recurrence cut**, optional D-side admission credits and fallback to PD_G1 | Do not call existing A sweep dynamic. |

Controls, diagnostics and ablations may be added only with clear labels outside the primary nine-arm matrix. For user-requested no-mixed setup, use the engine's `refill`/separate-phase execution uniformly when supported. Confirm `DP_Chunked` does not turn on Native Mixed inadvertently. No `A_native_mixed` arm in the main matrix. The `PD_G1` connector and A split must have comparable chunking; disclose any impossible equivalence rather than hiding it.

**Important cost fairness:** for axes with different discrete boundaries, compare at *matched actual P-side compute, actual D-side compute, transferred KV/state bytes, and exposed decode delay*, not just “half the prompt = half the loops”. Use per-worker measured elapsed kernel times, not inferred FLOPs alone. `Loop_Dynamic` must beat the **best eligible tuned static loop cut**, not merely an arbitrarily selected cut.

## 5. Workload design (same long-context plan, corrected model validity)

The prior 0.8–4K input and ≤512 long output plan is retained. Two intentionally different trace classes must be labeled:

1. **Controlled long generation:** model-tokenized ShareGPT prompts combined, where text exists legally, with real public code/document prompts or trace-driven length targets; `ignore_eos=True` and per-request `max_new_tokens` targets. This is a synthetic *length-controlled stress test*, not a natural answer-length distribution.
2. **Natural EOS:** the same underlying prompts but no forced generation; `ignore_eos=False`, `max_new_tokens=512`. Report actual EOS/cap stop fractions, mean output length and benchmark quality separately. Do not guarantee mean 400+ in natural EOS.

| Workload | Input profile (model-tokenized) | Output | Arrival/goal |
|---|---|---|---|
| `W1_LG` primary | 35% 0.8–1.5K; 40% 1.5–3K; 25% 3–4K (within context validity) | 25% 384; 50% 448; 25% 512, controlled; target mean 448 | Open-loop Poisson; representative diverse long-generation stress |
| `W1_NAT` | Same legal prompts | Natural EOS; cap 512 | Realized conversational length control |
| `W2_PHEAVY` | 2–4K, context-valid | 128–256 | Prefill pressure, D slack, split sensitivity |
| `W3_BURST` | Regular 0.8–2K, bursts 3–4K | 256–512 | Explicit P overload and D slack transitions; **primary adaptation test** |
| `W4_DHEAVY` | 0.8–1.5K | 400–512 | D saturation, fallback/Decode-SLO protection |

Prompt source: **ShareGPT text** for chat, and legitimate accessible public code or other real long documents. Azure-Code may supply *length/arrival statistics* where permitted; do not invent or label Azure trace statistics as actual prompt text. Record source revision/hash/license, tokenizer, filters, common-support manifest, actual prompt/output lengths, per-request stop reasons and exact arrival trace.

Keep separate calibration vs held-out requests, including independent arrival seeds and load points; calibrate each model's TTFT/TPOT SLO using controls only. Primary metric: **maximum tested stable SLO-constrained goodput**, joint TTFT + request-mean TPOT with ≥90% satisfied requests and a frozen queue-growth check. Report full finite-rate grid; never imply an infinite-horizon capacity guarantee. Secondary metrics: TTFT, TPOT, actual ITL, E2E p50/p95/p99, p99 tail interference, window throughput and output tokens/s.

## 6. Stage plan and explicit gates

### Stage 0 — correctness + source/novelty audit (CPU first)

Verify released numerical patch and compatibility; inspect `A`, `PD_G1`, `DP`, `refill` and current chunking. Map the unfolded logical-layer graph and every state boundary. Read DistServe, Splitwise, Sarathi-Serve, MLSys 2026 Layered Prefill and July 2026 load-aware prefill deflection as *non-identical but related* prior work. Preserve an evidence table separating **cross-GPU prefill compute migration**, **in-worker layered scheduling**, **pipeline parallelism**, and **layerwise KV streaming**. Execute no new candidate performance experiment before the new migration parity gates.

### Stage 1 — fixed loop-cut performance landscape

Start with small correct pilots (e.g., input 128/512, output caps 32/128) and expand to 512/1K/2K/valid 3–4K input and outputs 128/256/512. Use several deliberately controlled P-heavy, balanced and D-heavy conditions **at the same input length**. Initial cuts: Ouro `0,1,2,3,4`; Huginn `0,4,8,16,24,28,32`, adding neighbors only if warranted. `0`/`R` must have safe explicit endpoint execution. Keep Native Mixed disabled. Measure physical GPU/P/D time, queue, transfers, decode ITL and SLO attainment. Check whether the optimal cut shifts across load states. If not, mark `NO_GO_ADAPTATION_HEADROOM` and stop Stage 2–4 GPU development unless there is a separately justified scientific measurement.

### Stage 2 — dynamic-cut headroom, without pretending counterfactual measurements

Fit a small deterministic cost model from *measured* per-cut P time, D time, transfer and queue/capacity traces. Validate simulator predictions out-of-sample against **unseen** fixed-cut serving runs and quantify errors. Compare (a) full PD, (b) calibrated best static Loop Cut, (c) state-dependent fixed-*regime* cut, (d) an **optimistic oracle or upper bound clearly marked SIMULATED**. Never add independently measured per-request best latencies and call it goodput: decisions change congestion and other requests' deadlines. Freeze a quantitative Stage-2 GO threshold before seeing held-out results (suggestion: predicted ≥10% goodput headroom over best Static Loop with calibrated non-degenerate decode-SLO safety, then demand confirmation in real online runs). If prediction uncertainty spans zero and no credible positive region remains, stop early.

### Stage 3 — token, loop and generic layer migration (mandatory if Stage 2 GO)

Implement/validate legal multi-chunk migration. Do **not** merely remove A's single-chunk guard. Compare `Token_Static/Dynamic`, `Loop_Static/Dynamic` and `Layer_Static/Dynamic`, alongside the three nonmigration controls. Use the same model, same full replicated weights and 2GPU pair. For layer cuts, verify intermediate physical-layer serialization, layer-by-layer KV frontier and Huginn prelude/core/coda handling. Include unit tests asserting a generic logical-layer cut exactly at a recurrent-loop boundary is equal to Loop Cut under the same destination policy, within qualified numerical tolerance; exact bytes if same kernels and geometry.

Report a **2D Pareto plot** of (actual migrated prefill compute or P pressure relief, added D-side interference) and annotate KV bytes, intermediate hidden bytes, max ITL and goodput. Check hypotheses:
- H-A: loop cuts preserve large token compute shape compared with small chunk sizes at similar migration granularity;
- H-B: bytes differ (not assumed lower), and often small *exposed* handoff can coexist with large payload/HBM impact;
- H-C: online loop cuts might attain a better *cost/performance frontier* or require less boundary-state work than generic layer cuts. A superset layer oracle cannot be beaten on raw optimal-choice space by a subset loop oracle; any implementation advantage must be measured and explained.

Separate `NOT_IMPLEMENTED` from `NO_GAIN`. If a proper generic Layer-Dynamic baseline cannot be built, do not publish a complete loop-exclusive superiority claim.

### Stage 4 — online SLO-safe Dynamic A (only after Stage 1–3 GO)

Choose cut per request or at a safe uncommitted boundary, using only current P queue, D active decode token-step costs, expected continuation time, HBM, transfer cost and request deadlines. Reserve bounded D-side admission credits **before committing a migration**. If D gets busy, continue P work or delay migration at a legal boundary; do not rollback committed KV. `n=R` fallback is mandatory. No RL. Evaluate online 2GPU first; scale to 4/8GPU only after correctness and incremental result. Start 4GPU at 2P2D, 8GPU 4P4D, then include alternative legal P:D ratios as diagnostic equal-budget controls if relevant; avoid pooled parallel tests when physical fabric interferes with clean timing.

## 7. Tests, measurement and stop rules

**Correctness:** pinned checkpoint/BF16/fixed depth; exact input IDs, full recurrent-work accounting, token/stop parity at matched qualified execution, valid KV and hidden fingerprints, request/depth/coda frontier safety, EOS and output caps; independent seeds and multi-chunk prompts; producer-vs-receiver byte identity. Classify numerical batch invariance vs true memory/state corruption. Original HF vs RLT fidelity and GSM8K/MATH/HumanEval accuracy remain separate evidence categories and must never be silently equated to internal exact parity.

**GPU timing:** profiler-only attention/MLP/transfer/CUDA graph/stream activity with GPU synchronization discipline and correct critical-path accounting, plus separate clean end-to-end runs. Host phase percentages are not GPU kernel percentages. No sum of overlapping transfer windows. `NOT_MEASURED` rather than fabricated zero when a counter was disabled. Quiet runs no owned burners on measured fabric; never terminate unrelated processes. Keep full raw evidence and failures.

**Mandatory model-specific verdicts:** `CORRECTNESS_BLOCKED`, `NO_GO_ADAPTATION_HEADROOM`, `NO_GO_AXIS_ADVANTAGE`, `NO_GO_NOVELTY`, `CONDITIONAL`, `GO_EVIDENCE` (with actual samples and limits). Missing Stage 3/4 means `NOT_RUN`, not empirical failure. Apply separately to Ouro and Huginn. An implausibly favorable simulator or oracle alone is not a GO. Use repeat independent starts (≥3 at decisive points) and uncertainty by start, not token pseudoreplication.

**Checkpoints and deliverables:** after every stage publish audit+evidence before progressing. Store new uniquely named run namespace; preserve all old files. Reports under `poc_flashvep/reports/`, including `adaptive_recurrence_cut_vs_token_layer_split_poc_spec.md`, `dynamic_a_static_cut_headroom.md`, `dynamic_a_chunk_loop_layer_axis_comparison.md`, `dynamic_a_online_slo_serving_results.md`, `dynamic_a_final_summary.md`, `dynamic_a_reproduce.md`. Publish raw/frozen contracts + code revisions + plot data, safe commit/push on new branch, no force push. Never change checkpoint precision, context limits, target length policy or thresholds based on observed candidate outcomes.

## 8. Prior-art reading and exact claim boundary

- DistServe (OSDI 2024): https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin — PD GPU-pool isolation and SLO-goodput.
- Splitwise (ISCA 2024): https://arxiv.org/abs/2311.18677 — layerwise asynchronous KV **transfer**, not necessarily prefill compute migrated to decode worker.
- Sarathi-Serve (OSDI 2024): https://www.usenix.org/conference/osdi24/presentation/agrawal — chunked prefill and stall-free mixed scheduling, with pipeline configurations.
- Layered Prefill (MLSys 2026): https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html — layer-group **in-engine scheduling** vs chunked prefill; MoE evidence not directly a Dense Looped PD compute-migration result.
- Load-aware Prefill Deflection (2026): https://arxiv.org/abs/2607.02043 — D capacity as an alternative to static PD, chunk scheduling under Decode SLO.
- DuetServe (ICML 2026): https://proceedings.mlr.press/v306/gao26m.html — aggregated vs isolated prefill/decode trade-off.

**Final publication bar:** a claim that “loop cuts beat chunk and layer cuts” requires legal, measured, same-budget comparator implementations and a causal mechanism that survives strong alternatives and the generic-layer-subsumption argument. It is acceptable and informative to find no uniquely loop-specific advantage.
