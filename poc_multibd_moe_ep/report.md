# MultiBD × physical MoE EP discovery

## Contract and reproducibility

Same non-DMax checkpoint `SJTU-DENG-Lab/MBD-Code-LLaDA2-mini-16B@83434493ea9f026cbc139d681d7ae04ffefe052f`; official Diffulex
`mbd-lms@9d1a04e794351f18039b3234a10e8b07e15bcbbf`. The primary intervention is ONLY `buffer_size=R`
for R=1/2/3/4. No model training, router/expert approximation, or controller.
Physical GPUs 4,5,6,7 map in order to logical 0/1/2/3 on four H100 80GB GPUs.
This is the explicit 2026-10-13 user override from original GPUs 2/3/6/7.
New hardware confirmation session: `gpu4567_det_20261014`. Historical calibration
and interrupted held-out evidence are preserved; absolute old/new timing is
not pooled. See `artifacts/environment_gpu4567_det_20261014.json` for UUIDs/topology.
Topology is dense TP4 / strided source-SP4 / physical EP4 / DP1 / PP1.
All six pairs within the selected H100 80GB set report NV18 fabric connectivity
in `nvidia-smi topo -m` (NVLink/NVSwitch, not a PCIe-only EP measurement).
Physical 4/5/6/7 map respectively to logical CUDA 0/1/2/3 and UUIDs
`6076e2f2-5b63-3761-5586-56ceb7df8139`, `a1a1cfcf-93a1-3544-9a5e-e58144b68730`, `e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28`, `4cc26b88-19fc-1988-f9e0-17858aa7a99b`.
The supported EP-disabled control is `TP4/EP1 TPFusedMoE`: every TP rank
receives the full token batch, retains 64 complete contiguous experts, and
sums partial expert outputs through TP all-reduce. It does not split each
expert's intermediate dimension, and does not invoke DeepEP sparse dispatch.
This is the native supported alternative, not a claim of zero communication.
The actual loaded model has 20 layers, 19 sparse layers, hidden 2048,
256 routed experts, top-8, intermediate 512, one shared expert. Each owner
has 64 complete experts/layer (384 MiB/layer); weights are not replicated.
See `artifacts/environment.json`, pins, ownership, and physical validation.

## Runtime integrity

Pinned upstream rejects EP configs despite including experimental EP modules.
The bounded bridge removes ONLY that config guard and repairs DeepEP normal
owner-local IDs (0..63), which upstream mistakenly treated as global IDs.
The failed unmodified physical proof is retained. Fixed real Dispatch →
owner-local GEMM → Combine passed numerical tolerance fixed before held-out.
Official scheduler, sampler, Block Buffer, attention and KV-cache state machine
are not reimplemented. This is Diffulex, NOT SGLang serving: only the previously
validated Python environment has SGLang in its directory name.

R is buffer CAPACITY, not necessarily the number of active blocks. Dummy,
to-cache and active rows are reported separately where captured. Natural EOS
is enabled, temperature 0, block/page32, naive non-DMax sampling, native
thresholds add=.9, semi=.9, accept=.95. Caps: 4096 output tokens, NFE1024,
model context4096, scoped process watchdogs. No failed artifact is removed.

After resuming on 4/5/6/7, a same-R native-repeat acceptance mismatch was
found on frozen HumanEval/11: final 122 tokens and NFE41 matched, but three
intermediate steps differed. The initial session was stopped, not credited
as valid performance. One-request counterbalanced diagnostics used six
repeats per setting plus trace. Fixed cuBLAS workspace and deterministic
operators (automatic scratch poison-fill disabled for DeepEP's asynchronous
buffer contract) passed that diagnostic without changing tokens, thresholds,
precision or routing. The final sweep freezes that recipe for ALL R and
both backends and checks acceptance parity after every problem. This is a
runtime reproducibility repair, not an EP optimization or an acceptance rule.
See interrupted evidence and `results/gpu4567_20261013/determinism_no_fill_audit.json`.
Because both diagnostic settings passed with fixed workspace, the exact
original non-determinism source remains unresolved; index-add alone is not
claimed as the proven cause. The failed poison-fill diagnostic is preserved.

## Workload and measurement

Metadata-only seed3407 freeze: 8 calibration + 24 frozen held-out problems,
half HumanEval and half MBPP. Prompts are official reproduction templates.
Full official EvalPlus 0.3.1 HumanEvalPlus v0.1.10 / MbppPlus v0.2.0 tests
are evaluated in private mount/network/PID namespaces plus resource limits.
The source problem examples alone are NOT the reported Plus quality.

Clean inference restores original unwrapped D/E/C functions. Held-out clean
and event-only runs are counterbalanced within the same loaded model; trace
is a separate heavy lane. Three request warmups and three measured clean/event
repeats; one heavy trace. Probe barriers and all serialization are outside
inference timing. Earlier separate-process calibration had drift and some
cross-rank flush races; all logs are retained and marked, not headline samples.

Prefix caching is enabled and each prompt is primed: headline E2E is therefore
WARM-PROMPT generation, not cold uncached serving latency. Full-request NFE
includes native prefill/refinement forwards; TPF/TPS denominators use returned
final output tokens, not padded buffer rows or raw acceptances later cut by EOS.
Upstream `is_truncated` includes normal EOS: only actual context/NFE/output/
repetition caps exclude quality. Raw output and all stop flags are retained.

Stage timing selects ONE cumulative-critical rank/request and uses its D/E/C
intervals, not a sum of unrelated rank maxima. Sparse routed D/E/C excludes
shared expert and full-source hidden/router-logit all-reduces; whole MoE events
capture those separately. Logical bytes use one BF16 hidden/output row per
remote source-token/destination, validated against actual DeepEP receive rows.
Logical payload excludes protocol/meta overhead and is NOT measured wire/HBM
traffic. Heavy-trace wall time is never headline latency.
Dispatch/Combine are dispatcher API intervals, not pure transport kernels:
Dispatch includes layout/count coordination and expert-major preparation;
Combine includes restoring slot order, FP32 local accumulation, conversion
back to BF16 and real reverse collective completion. Owner execution includes
the native packed expert GEMMs and routing-weight multiplication. GPU events
are resolved only after the request; no per-layer host synchronization is
added by the observer. Full-buffer restoration uses the existing native
all-reduces, rather than an assumed-free oracle layout step.

## Held-out primary table

| R | N | TPF | Step ms | TPS | PCE | EP ms/commit | D+C bytes/commit | Active experts | Rank max/mean | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 3.500 | 98.910 | 35.424 | 1.000 | 13.635 | 3784946.718 | 66.496 | 1.623 | 0.667 |
| 2 | 24 | 4.643 | 98.059 | 47.875 | 1.012 | 10.911 | 5904256.800 | 82.022 | 1.528 | 0.542 |
| 3 | 24 | 4.713 | 108.504 | 44.005 | 0.932 | 12.406 | 8751842.006 | 83.975 | 1.559 | 0.542 |
| 4 | 24 | 4.715 | 104.024 | 46.358 | 0.976 | 11.232 | 11633403.283 | 84.605 | 1.630 | 0.542 |

## Paired conversion ratios, mean [request-bootstrap 95% CI]

| R | TPF gain | TPS gain | PCE | EP ms/commit ratio | Logical D+C bytes/commit ratio |
|---|---|---|---|---|---|
| 1 | 1.000 [1.000, 1.000]; n=24 | 1.000 [1.000, 1.000]; n=24 | 1.000 [1.000, 1.000]; n=24 | 1.000 [1.000, 1.000]; n=24 | 1.000 [1.000, 1.000]; n=24 |
| 2 | 1.326 [1.242, 1.437]; n=24 | 1.341 [1.238, 1.461]; n=24 | 1.012 [0.970, 1.044]; n=24 | 0.796 [0.730, 0.878]; n=24 | 1.550 [1.462, 1.633]; n=24 |
| 3 | 1.345 [1.264, 1.453]; n=24 | 1.247 [1.156, 1.337]; n=24 | 0.932 [0.880, 0.980]; n=24 | 0.909 [0.813, 1.011]; n=24 | 2.287 [2.164, 2.403]; n=24 |
| 4 | 1.346 [1.265, 1.454]; n=24 | 1.312 [1.208, 1.434]; n=24 | 0.976 [0.927, 1.014]; n=24 | 0.817 [0.749, 0.895]; n=24 | 3.039 [2.864, 3.200]; n=24 |

Ratios compare each problem to its own R=1 result before aggregation.
Stage ratios are observed values only; interpret them subject to the event
observer validity below, not as unconditional physical attribution.
Incremental EP-share statistics include only requests with positive added
step latency; their sample count and both deltas are retained. Near-zero
denominators can make that statistic unstable, and requests with no added
latency are not assigned an invented positive share.

## Same-checkpoint supported TP4/EP1 control

| R | TPF | Step ms | TPS | PCE | Plus pass@1 |
|---|---:|---:|---:|---:|---:|
| 1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 2 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| 4 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Optional TP control validated: `False`. A failed or
incomplete control is NOT used in plots, paired contrasts, headline timing
or quality claims. Its raw results/failure flags remain in the evidence.
In this session native TP4/EP1 stopped on HumanEval/11 at R3: repeated clean
execution changed final tokens, NFE and acceptance despite the frozen
determinism recipe. See `results/gpu4567_det_20261014/tp_control_parity_audit.json`.
No alternative MoE kernel or engine rewrite was attempted to rescue this
optional comparison. The exact internal cause is unresolved. Thus EP-vs-TP
specificity is unavailable; the primary EP4 sweep is judged independently.

## Real block utilization and sparse stage costs

| R | Mean real active blocks | Mean dummy-row fraction | Dispatch ms/forward | Expert ms/forward | Combine ms/forward |
|---|---:|---:|---:|---:|---:|
| 1 | 0.838 | 0.000 | 20.443 | 12.915 | 7.792 |
| 2 | 1.094 | 0.343 | 20.852 | 12.650 | 8.780 |
| 3 | 1.101 | 0.559 | 24.120 | 12.679 | 12.458 |
| 4 | 1.101 | 0.669 | 21.790 | 12.683 | 9.770 |

Costs are request-mean sums across all 19 sparse layers per actual forward.
Block utilization excludes native prefill forwards and separates physical
dummy rows from real active blocks. Capacity R is not equated to utilization.

## Event observer validity

| R | Event-only wall overhead; 95% CI | Status |
|---|---|---|
| 1 | 0.032 [0.005, 0.052]; n=24 | VALIDATED_BOUNDED |
| 2 | 0.046 [0.038, 0.056]; n=24 | VALIDATED_BOUNDED |
| 3 | 0.047 [0.036, 0.060]; n=24 | VALIDATED_BOUNDED |
| 4 | 0.065 [0.037, 0.099]; n=24 | VALIDATED_BOUNDED |

## Request-level paired statistics

Full means and >=2000-resample paired request CIs are in
`results/gpu4567_det_20261014/conversion_summary.json`; repeats/layers/forwards are NOT independent
samples. Event observer validity, TP-only contrasts and quality CIs are in
`results/gpu4567_det_20261014/headline_summary.json`. Stage attribution is UNTRUSTED whenever the
same-session observer overhead/upper uncertainty exceeds 10%. Clean E2E
remains reported in that case. Cross-R token identity is not required;
same-R output/NFE/acceptance identity across observers IS required.
R order was frozen as 3/2/4/1, first EP4 then TP4; observer order, not R order,
is counterbalanced within each request. Thus between-arm thermal/system drift
cannot be eliminated by treating repeats as new problems. Other users on
GPUs 0--3 were not interrupted; those GPUs are outside our execution mask.

## Prior boundary and limitations

See `prior_overlap.md`. MultiBD, MultiTF, TPF/TPS conversion loss, B1 splitting
and generic overlap are not new contributions. Earlier 5.77x live-only bound
is NOT measured Epoch. This is an eager bounded experimental EP bridge, not
the paper's optimized TP2 static runner; absolute TPS is not a paper reproduction.
The 24-problem quality subset cannot establish broad benchmark noninferiority.
No synthetic oracle is substituted for physical EP timing. No adaptive method
is implemented. Calibrated tolerances, request selection and R were not retuned
on held-out results.

## Reproduce / evidence

See `reproduce.md`. Code, full route/weight traces, actual ownership, process
failures, clean outputs, event-only intervals, per-forward/request/layer CSV,
quality test identities, bootstrap JSON and eight plots are retained in this
directory. Model weights and isolated third-party environments remain outside
git; exact revisions and commands are pinned.
The complete layer table is `results/gpu4567_det_20261014/ep_metrics.csv.gz`
(lossless compressed CSV). A preserved plain intermediate table is explicitly
named `ep_metrics.intermediate_before_full_sweep.csv`; do not substitute that
partial snapshot for the completed data.

## Final judgment

`NO-GO`. Completed all-R primary EP4 sweep: `True`;
complete validated EP4/TP4 sweep: `False`;
all held-out EP4 quality cases evaluable: `True`.
Gate details and request-level statistics: `results/gpu4567_det_20261014/final_verdict.json`.
The session-specific interpretation and contract answers follow.

## Evidence-backed interpretation and all 18 contract answers

1. **Checkpoint/runtime:** non-DMax, MultiTF-trained
   `SJTU-DENG-Lab/MBD-Code-LLaDA2-mini-16B@83434493ea9f026cbc139d681d7ae04ffefe052f`,
   official Diffulex `mbd-lms@9d1a04e794351f18039b3234a10e8b07e15bcbbf`.
   The tested backend is the bounded eager experimental EP bridge, not a
   reproduction of the paper's optimized TP2 throughput.
2. **Actual MoE geometry:** 20 layers, 19 sparse layers, hidden 2048,
   256 routed experts, top-8, intermediate 512, BF16, one shared expert.
   A complete layer-specific expert is 6 MiB; each owner has 64 experts,
   384 MiB of routed weights per sparse layer. No weight replication.
3. **TRUE EP4:** PASS on physical **4/5/6/7**, under the latest explicit user
   override, not on the spec's original 2/3/6/7. Real four-rank Dispatch,
   owner-local execution and Combine; loaded owner tensor samples match
   the pinned checkpoint. Every captured held-out receive-row/path check
   passes. All 96 EP4 request/R combinations pass exact same-R final-token,
   NFE and acceptance parity across clean/event/trace observers. Original
   failures are retained and are not credited as valid measurements.
4. **TPF:** request means R1/R2/R3/R4 = **3.500/4.643/4.713/4.715**.
   Paired R>1 gains = **1.326/1.345/1.346×**. The algorithmic gain mostly
   saturates at R2, rather than scaling with nominal capacity to R4.
5. **Step latency:** clean warm-generation means =
   **98.910/98.059/108.503/104.024 ms**. R3 has a moderate added cost;
   this is not a consistently monotonic R-to-latency curve.
6. **TPS:** means = **35.424/47.875/44.005/46.358 tokens/s**.
   Paired gains = **1.341/1.247/1.312×**. These use final returned output,
   not physical/dummy rows, and are warm-prompt generation, not cold TTFT.
7. **PCE:** paired means = **1.012/0.932/0.976** for R2/R3/R4.
   Request-bootstrap 95% CIs = **[0.970,1.044]/[0.880,0.980]/[0.927,1.014]**.
   None meets the poor-conversion gate of <=0.75. Even R3's aggregate lower
   CI is above 0.85. R2 converts its TPF gain efficiently in this regime.
8. **EP cost per final committed token:** event-only routed D+E+C means =
   **13.635/10.911/12.406/11.232 ms**. Paired R>1 ratios =
   **0.796/0.909/0.817×**. Conversely, destination-deduplicated remote
   logical D+C bytes/token = **3.785/5.904/8.752/11.633 MB** (decimal);
   paired ratios = **1.550/2.287/3.039×**. Bytes rise, but useful-token EP
   time does not show the hypothesized >=20% degradation. Byte reduction
   opportunity alone is not measured latency headroom or a method GO.
9. **Expert working set:** mean union/sparse layer =
   **66.50/82.02/83.98/84.61**, while assignments/active expert =
   **4.21/6.53/9.53/12.54**. Larger batches improve per-expert batching while
   the active union mostly saturates. Actual expert-tensor sizes underpin
   the active-weight proxy; it is not actual HBM/DRAM traffic.
10. **Rank graph:** edge density = **0.9951/0.9997/0.9997/0.9995**. It is
    already effectively dense at R1. Destination fanout/token =
    **3.059/3.110/3.143/3.147**, including local destinations. There is no
    new sparse-to-dense rank-graph transition as R increases.
11. **Rank pressure:** mean assignment max/mean =
    **1.623/1.528/1.559/1.630**; rank CV = **0.441/0.388/0.418/0.470**.
    Load imbalance does not monotonically increase from R1. Raw per-layer,
    per-forward rank counts remain available; means are not worst-tail proofs.
12. **Added step attribution:** mean paired Delta_step at R2/R3/R4 =
    **-0.851/+9.594/+5.115 ms**, and Delta_EP = **+1.132/+8.107/+3.091 ms**.
    The primary mean-of-request-ratios is conditioned on positive Delta_step:
    n=**5/13/11**, mean **-1.795/2.155/-6.376**, with very wide CIs and
    near-zero-denominator outliers. Do not interpret those as literal bounded
    percentage shares. As a secondary stability diagnostic, ratios of delta
    sums within the same positive subsets are **0.632/0.772/0.437**. The
    raw metric is retained, not switched to promote a result. R3's extra
    routed-stage cost is consistent with its moderate slowdown, but Gate 2
    and the quality gate fail; no strong EP conversion-collapse claim follows.
13. **TP control:** attempted native TP4/EP1, not validated. Its R3 clean
    repeats on HumanEval/11 changed tokens/NFE/acceptance despite the fixed
    deterministic recipe. Raw failure and the CPU parity audit are preserved.
    No TP contrast or TP quality number is headline evidence. This native
    backend uses complete-expert ownership plus full-batch all-reduce, not
    intermediate-dimension-sharded canonical pure TP. EP-vs-TP specificity
    is therefore **UNAVAILABLE**, not silently presumed.
14. **Quality:** full EvalPlus Base+Plus tests in a resource-limited private
    namespace, 12 HumanEvalPlus + 12 MBPPPlus held-out problems:
    R1 **16/24 (66.7%)**, R2/R3/R4 **13/24 (54.2%)**. Each comparison has
    four correct-to-wrong and one wrong-to-correct transition; paired delta
    **-12.5pp**, 95% CI **[-29.17,+4.17]pp**. This small cohort cannot prove
    a broad accuracy loss, but certainly does not establish <=1–2pp quality
    preservation. No capped completion is counted as valid benchmark quality.
15. **Held-out robustness:** all R arms use the same 24 metadata-frozen
    requests; all primary cases completed, with three clean repeats per
    request and request-level bootstrap. Poor conversion <=0.75 does not
    occur as an aggregate across two R settings. The pool was frozen before
    routing; runtime diagnostics reused one request, without changing
    scientific parameters, selection or tolerance. It is not newly untouched
    data collected after the resume. R order is fixed, not counterbalanced,
    so between-arm thermal/fabric drift remains a limitation.
16. **Dominant interpretation:** useful parallelism saturates after R2;
    additional physical rows are substantially dummy capacity, rather than
    genuinely concurrent fresh blocks. Mean real pre-forward active blocks
    at R2/R3/R4 = **1.094/1.101/1.101**, dummy-row fractions =
    **34.3%/55.9%/66.9%**. At R3 and R4, 2/24 requests reach at most one real
    active block, 21/24 at most two, and only 1/24 reaches three; none reaches
    four. This is the **pre-forward** state, not the earlier post-sampler
    snapshot. Already-dense rank communication and improved expert batching
    are consistent with efficient EP time/commit despite larger logical
    payloads. A pure transport/HBM cause is not isolated by this PoC.
17. **Verdict: NO-GO** for the tested B1/default-threshold/non-DMax eager
    regime. Algorithmic TPF gain and byte growth exist, but the hypothesized
    large physical EP conversion loss is not established, and quality-safe
    scaling is not demonstrated. This is not a universal claim that MultiBD
    or every optimized EP implementation cannot have an admission problem.
18. **Next action:** do not implement an EP-aware running-set controller
    from this evidence. Close this tested motivation. If a later study is
    independently justified, first validate deterministic native TP control
    and a metadata-defined workload with genuinely high real multi-block
    occupancy; that is a different confirmation regime, not a rescue here.

The event observer's mean wall overhead is **3.15/4.62/4.74/6.46%** for
R1/R2/R3/R4; 95% upper bounds are **5.19/5.58/6.04/9.92%**. R4 is close to
the frozen 10% ceiling and does not meet the preferred <=5% target. Event
stage costs remain a bounded observed stratum, not an exact clean-run
decomposition. Heavy-trace latency is never used as headline timing.
No Epoch runtime comparison, expert pruning, replication, DMax, learned
predictor, new CUDA kernel or adaptive controller was implemented.
