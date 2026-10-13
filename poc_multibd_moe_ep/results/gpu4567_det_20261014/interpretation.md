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
