# MLLM MoE Hierarchical Cross-Module Rebatching PoC

## Decision

**FINAL STATUS: `NO_HIERARCHICAL_HEADROOM`**

The proposed three-level hierarchy is algebraically possible, but its measured
economic opportunity is too small. Across three independent warmed observer
restarts, the optimistic full-hierarchy oracle reduces direct BCT by a median
**1.37%** and at most **6.42%**. Adding only an idealized HBM-copy lower bound
for the two exact hidden-state yields gives **1.35% median**. The extra MoE
value beyond independent vision/LM stage batching is **0.50% median**, and the
extra vision value beyond BatchGen-style LM rebatching is **1.37% median**.

Those figures fail the 8% implementation gate, the 5% MoE-specificity gate and
the 8% MLLM-specificity gate. No scheduler prototype or Kimi run was performed.

## 1. Scope: offline batch inference, not interactive serving

The primary unit is a closed set of requests submitted as one offline job:

\[
\mathrm{BCT}=\max_i(t^{completion}_i)-t^{GPU\ cohort\ ready}.
\]

All requests on both DP shards are rendered before a common barrier and before
any GPU submission. This is materially different from interactive serving:
the scheduler may reorder work and hold exact hidden state to maximize whole-job
throughput, whereas interactive serving is constrained by per-request TTFT and
TPOT. BatchGen itself targets this offline setting and reports gains in BCT,[^1]
while RPS-Serve and ModServe primarily address online SLO and stage-resource
problems.[^2][^3] Nothing in this report is an interactive-latency claim.

## 2. Prior-art conclusion before implementation

BatchGen already provides sequence coroutines, intra-forward `YIELD`/`COMBINE`,
and attention→MoE regrouping.[^1][^4] Its OSDI paper selects intra-forward
yield points statically per model and explicitly calls a post-vision-encoder
yield for vision-language models future work. Therefore neither “use different
attention and MoE batches” nor “yield once after the encoder” is novel.

Current BatchGen main was audited at
`ca2aaca8a24fa110bb48b86f1876fe25b5cf3267` separately from the artifact pin
`df221143d0520ea37d2dd35b23f2915ae5f92678`. Current main has Kimi-K3
scaffolding but its model path explicitly rejects vision media placeholders;
no working Qwen3-VL hierarchy was found. vLLM-Omni currently implements real
asynchronous chunks across composite stages such as Thinker→Talker→Code2Wav,[^5]
not internal Qwen3-VL encoder→attention→MoE regrouping. ModServe and ElasticMM
already occupy generic modality/stage disaggregation,[^3][^6] and BlendServe
already studies offline request ordering under competing resource and prefix
objectives.[^7]

The exact potential novelty gap was consequently narrow:

> a natural, high-mass conflict among vision compatibility, attention
> compatibility and sparse-expert compatibility that requires more than static
> stage separation or BatchGen's LM-internal regrouping.

The GPU results do not support that gap. Full details are in
`mllm_moe_hierarchical_rebatching/SOURCE_AND_PRIOR_ART_AUDIT.md` and
`BATCHGEN_PAPER_VS_MAIN.md`.

## 3. Runtime and evidence contract

### Hardware and model

- Physical GPUs: NVIDIA H100 80GB HBM3, indices 4/5/6/7 only.
- UUIDs:
  - 4: `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139`
  - 5: `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730`
  - 6: `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28`
  - 7: `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b`
- Driver 570.211.01, CUDA 12.9 runtime, PyTorch 2.11.0+cu129.
- Model: Qwen3-VL-30B-A3B-Instruct, BF16, model snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Runtime: vLLM 0.20.0, TP2/DP2/EP4, DeepEP high-throughput, DBO off,
  eager execution, 2 GiB KV cache per GPU.

Every observer process emitted and asserted:

```text
TP=2, DP=2, EP=4
enable_ep=true
prepare_finalize=DeepEPHTPrepareAndFinalize
experts=TritonExperts
backend=deepep_high_throughput
DBO=false
CUDA_VISIBLE_DEVICES=4,5,6,7
```

Clean BCT and SCT runs contain no CUDA hooks. Observer runs use same-device
events and reduce duplicate rank rows by the maximum rank-local duration. They
are diagnostic only; observer wall time is never headline performance. The
observer's per-policy stage CV ranges roughly 15–45%, reinforcing this split.

### Model facts verified from config

- 48 sparse layers, 128 routed experts, top-8.
- Hidden size 2,048; MoE intermediate size 768.
- Vision tower depth 27; packed variable-resolution inputs.

### GPU accounting

There were 16 successful target-runtime runs: 31.71 minutes of meaningful live
four-GPU wall time, or 2.114 four-GPU-hours. One wrong-interpreter smoke launch
is retained but excluded. Burn is excluded. The headroom-first rule stopped
additional GPU work after the optimistic oracle failed, rather than padding the
run time with low-information repetitions.

## 4. Workloads and controls

The 128-request real-image pool contains 32 requests in each class:

- text-only control;
- single real image;
- two real images;
- long text plus a real image.

Across every grouping policy the closed set is identical:

| Quantity | Value |
|---|---:|
| Requests | 128 |
| Actual model-input prompt tokens | 45,893 |
| Images | 128 |
| Pixels | 47,600,528 |
| Question words | 9,907 |

Policies preserve the request count, pixels, image histogram, resolution set,
prompt-token total and output budget. The primary batch-128 control fixes DP
membership independently of policy and balances actual prompt tokens to
22,947 vs 22,946 (max/mean 1.00002). The roles are:

- P0 natural/source interleaving;
- P1 vision-aligned ordering;
- P2 LM-length-aligned ordering, also a conflict screen for vision/MoE;
- random;
- coarse single/multi-image grouping;
- coarse text/image grouping.

The experiment does **not** manufacture a route-conflicted synthetic pool.
Natural routing/load is measured after scheduling. Since the natural/full
hierarchy oracle already fails the 8% gate, spending GPU time to engineer a
synthetic route correlation could not establish the required natural headroom.

Batch sizes 16/32/64/128 were screened at max-token-1. Batch 128 was then run
with full-shape warmup in repeated clean and observer runs. Fixed-16 and natural
EOS up to 32 tokens were collected as secondary controls.

## 5. Confounds found and removed

### 5.1 Frontend overlap

The first smoke interleaved image preprocessing and engine submission, making a
vision grouping look 7–8% better. The harness was changed so both DP workers
render the complete cohort, meet a common barrier, and only then submit. The
original result is excluded.

### 5.2 DP membership

Naive rank-stride assignment made the P0 batch-128 shard contain 16,627 prompt
tokens while the other held 29,266 (max/mean 1.275). The apparent policy effect
changed substantially after policy-independent greedy balancing. All primary
policies use the 22,947/22,946 assignment.

### 5.3 First large shape

Two tiny requests did not warm a 128-request/45.9K-token job. Independent
restart policy CV reached 17%. Two complete-plan warmups were added. The warmed
clean policy CV remains 7.3–18.1%, so paired medians and observer restart
oracles are reported rather than a single best run.

## 6. Clean BCT results

### 6.1 Max-token-1, warmed batch 128

| Policy | Median BCT | CV | Relative to P0 |
|---|---:|---:|---:|
| P0 global/natural | **1.129 s** | 18.1% | — |
| P1 vision | 1.211 s | 12.4% | +7.3% |
| P2 LM length | 1.651 s | 11.7% | +46.3% |
| Random | 1.235 s | 8.6% | +9.4% |
| Single/multi image | 1.307 s | 7.3% | +15.8% |
| Text/image | 1.238 s | 12.3% | +9.7% |

The striking P2 penalty is real enough to be an operational warning: ordering
requests by LM length can create adverse chunked-prefill/admission waves even
when global work and DP load are fixed. It is not a hierarchical-rebatching
opportunity. The natural fixed order avoids it, and no module-boundary method
is needed to recover the loss.

### 6.2 Batch-size screen

| Batch size | Best policy in screen | Best BCT | P0 BCT |
|---:|---|---:|---:|
| 16 | LM length | 2.262 s | 2.586 s |
| 32 | P0 | 1.768 s | 1.768 s |
| 64 | Vision | 1.390 s | 1.398 s |
| 128 | Vision (un-warmed screen) | 1.145 s | 1.219 s |

The ranking changes with cohort size, so total token count is not a complete
performance model. However, after full-shape warmup no nontrivial grouping is a
stable clean winner, and the module-mixing lower envelope remains small.

### 6.3 Output-length controls

- Fixed 16: best text/image BCT 2.498 s vs P0 2.571 s, a 2.8% diagnostic gain.
- Natural EOS, max 32: best text/image BCT 3.977 s vs P0 4.205 s, a 5.4%
  diagnostic gain.

These are below the 8% gate. Moreover, full greedy sequences agree across
groupings for only 105/128 fixed-16 requests and 107/128 natural-EOS requests.
Fixed-16 first tokens agree for 126/128. These multi-token timings are therefore
excluded from positive speedup evidence; they serve only as negative
output-regime screens.

## 7. Module cost atlas

The rank-critical same-device CUDA atlas yields the following medians over
three observer restarts:

| Policy | Vision | Attention | MoE total | LM forward | Route rank max/mean |
|---|---:|---:|---:|---:|---:|
| P0 | 306.6 ms | 189.8 ms | 618.9 ms | 842.3 ms | 1.258 |
| Vision | 296.3 ms | 153.1 ms | 651.4 ms | 857.3 ms | 1.188 |
| LM length | 330.3 ms | 251.2 ms | 721.6 ms | 1,055.4 ms | 1.358 |
| Random | 358.4 ms | 204.2 ms | 583.9 ms | 804.2 ms | 1.253 |
| Single/multi | 236.1 ms | 136.0 ms | 499.9 ms | 691.3 ms | 1.177 |
| Text/image | 264.7 ms | 176.3 ms | 514.0 ms | 770.8 ms | 1.201 |

The key causal observation is not the exact magnitudes, which carry observer
tax, but the winner pattern:

- attention winner: single/multi in **3/3** restarts;
- MoE winner: single/multi in **3/3** restarts;
- vision winner: text/image, P0 and single/multi once each.

Thus attention and MoE do not request conflicting groupings in this regime.
Vision sometimes differs, but the difference has little direct critical-path
mass. Route HHI and rank max/mean correlate with diagnostic MoE time, yet
correlation does not translate to incremental schedule headroom.

## 8. Strongest-baseline oracle

For each observer restart, let `V(g)`, `A(g)` and `E(g)` be measured vision,
attention-unit and MoE costs under grouping `g`. The oracle preserves exact
dependencies and does not assume inter-module overlap:

```text
P3 BatchGen LM-only = min_g [V(g)+A(g)] + min_g E(g)
P4 stage-only       = min_g V(g) + min_g [A(g)+E(g)]
P5 full hierarchy   = min_g V(g) + min_g A(g) + min_g E(g)
P7 feasible lower  = P5 + two ideal BF16 hidden-copy lower bounds
```

The direct saving is divided by the same restart's BCT; rank rows are not
summed. Results:

| Oracle | Median direct BCT | Maximum |
|---|---:|---:|
| P3 BatchGen-style LM-only | 0.00% | 1.69% |
| P4 independent vision/LM stages | 1.37% | 4.57% |
| P5 full hierarchy | **1.37%** | **6.42%** |
| P7 favorable feasible lower bound | **1.35%** | **6.40%** |

One exact global BF16 hidden tensor is 187.98 MB
(45,893×2,048×2 bytes). At an optimistic 1.5 TB/s, a pure HBM copy is 0.125
ms per boundary; real metadata, buffer ownership, waiting and restoration can
only reduce P7. The proposal is killed by lack of removable work, not by state
materialization overhead.

Incremental specificity:

- full hierarchy over P3: **1.37% median** (vision-specific increment; required 8%);
- full hierarchy over P4: **0.50% median** (MoE-specific increment; required 5%).

## 9. Trivial-fix and causality attack

The strongest static controls are enough:

- natural interleaving is the best warmed clean max-token-1 policy;
- a single/multi-image bucket is the common attention/MoE stage winner;
- actual-token DP balancing removes the largest false effect;
- full-shape warmup removes the first-use story;
- vision, length, text/image and random alternatives fail to provide a stable
  >8% gain.

Because the perfect full hierarchy itself is below 8%, asking whether a simple
bucket recovers 80% of it is almost moot: the simple grouping already chooses
the same attention and MoE winner, leaving only a 1.37% median vision increment.

## 10. Fifteen-hypothesis requirement

Twenty-two distinct causal hypotheses were evaluated. They cover frontend
submission, DP membership, warmup state, six grouping policies, batch-size
interaction, total-volume sufficiency, three-way module conflict, observer tax,
vision/LM conflict, routing's independent value, P3/P4/P5/P7 oracles, yield
memory, output length, and output correctness. Detailed `EXPECTED`, `OBSERVED`,
failed assumption, new fact and child decisions are in `RESEARCH_TREE.md`.

## 11. Direct answers to the ten research questions

1. **Is offline MLLM batch inference materially different from interactive
   serving?** Yes in objective and allowable reordering. This PoC measures
   closed-set BCT only and makes no TTFT/TPOT serving claim.
2. **Do vision, attention and MoE actually want different groupings?** Not
   materially. Vision occasionally differs; attention and MoE choose the same
   simple policy in 3/3 observer restarts.
3. **Is total token count enough to predict BCT?** No. Identical global and DP
   token totals still show order-dependent BCT, especially the adverse
   LM-length order. That fact does not imply hierarchical headroom.
4. **Does hierarchical rebatching beat stage-only MLLM batching?** No. Its
   median MoE-specific increment over P4 is 0.50%.
5. **Does it beat BatchGen-style LM-only rebatching?** No materially. The median
   vision increment over P3 is 1.37%.
6. **Does MoE routing add independent value?** Routing affects measured MoE
   cost, but contributes only 0.50% median schedule oracle after the stage
   baseline. The required independent BCT value is absent.
7. **Can simple buckets recover the oracle?** Yes in the only meaningful sense:
   a coarse single/multi grouping already minimizes attention and MoE; natural
   interleaving is the best clean warmed baseline. There is no large residual
   oracle left to recover.
8. **What is direct BCT headroom?** P5 optimistic median 1.37%, max 6.42%; P7
   favorable feasible median 1.35%, max 6.40%.
9. **What gap remains beyond BatchGen and vLLM-Omni?** An implementation gap
   remains for internal Qwen3-VL encoder→attention→MoE coroutines, but measured
   BCT value is negligible. The generic abstraction is already anticipated by
   BatchGen and adjacent to vLLM-Omni/ModServe/ElasticMM.
10. **Is there a clean paper-level method?** No. The central conflict is absent,
    headroom is below gate, and the residual behavior is dominated by simple
    ordering/DP/warmup controls.

## 12. Limitations and falsifiability

- This is a single-node four-H100 Qwen3-VL result. A much larger multi-node
  fleet, video-heavy encoders, or a model whose attention and MoE grouping
  objectives genuinely diverge could change the answer.
- The exact ALIGNED/CONFLICTED route correlation was screened through natural
  policy orderings rather than a synthetic histogram-preserving route
  construction. This is conservative for a natural-workload claim; a synthetic
  positive could not overcome the already-small natural direct oracle.
- Observer stage variability is high. The conclusion relies on a no-go upper
  bound across three restarts, not on exact stage speedup.
- Greedy divergence under long generation deserves numerical reproducibility
  investigation, but it does not rescue the BCT oracle and is outside this
  scheduler PoC.

## 13. Recommendation

Do not port BatchGen to Qwen3-VL for this hierarchy, do not implement adaptive
yield selection, and do not promote to Kimi. If revisiting the problem class,
first find a model/workload where measured attention and MoE winners diverge
and where a direct stage-only-to-hierarchy BCT oracle exceeds 8%; only then is a
native engine port warranted.

## Artifacts

- Working contract: `poc_flashvep/reports/mllm_moe_hierarchical_rebatching_poc_spec.md`
- Analysis and source: `poc_flashvep/mllm_moe_hierarchical_rebatching/`
- Raw results: `poc_flashvep/deepep_revalidation/results/mllm_moe_hierarchical_rebatching_poc_20260909_233848/`
- Machine-readable tables: `MODULE_COST_ATLAS.csv`,
  `MATCHED_POOL_CONTROLS.csv`, `SCHEDULE_BASELINES.csv`,
  `HIERARCHICAL_ORACLES.csv`, `BCT_RESULTS.csv`, `SCT_RESULTS.csv`,
  `MEMORY_RESULTS.csv`, `GPU_TIME_LOG.csv`.

## References

[^1]: [BatchGen: An Architecture for Scalable and Efficient Batch Inference, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/xu-tairan)
[^2]: [RPS-Serve](https://arxiv.org/abs/2603.26498)
[^3]: [ModServe](https://arxiv.org/abs/2502.00937)
[^4]: [MoE-Gen](https://arxiv.org/abs/2503.09716)
[^5]: [vLLM-Omni async chunk](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/async_chunk.md)
[^6]: [ElasticMM](https://arxiv.org/abs/2507.10069)
[^7]: [BlendServe](https://arxiv.org/abs/2411.16102)
[^8]: [HeteroServe](https://arxiv.org/abs/2603.12707)
[^9]: [M*: A Modular, Extensible, Serving System for Multimodal Models](https://arxiv.org/abs/2606.12688)
