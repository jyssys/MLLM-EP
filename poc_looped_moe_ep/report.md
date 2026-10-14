# Looped MoE × physical EP: bounded kill-test

## Decision

**Verdict: `SMALL_MODEL_ONLY`. Method implementation: NO-GO.**

An authentic pretrained Foil-1 runs correctly with real four-rank sharded
experts on GPUs **4/5/6/7**. Loop-dependent routing and changing assignment-hot
ranks are present. However, improving the global static rank-load proxy does
**not** produce a material physical latency gain. The required validated
best-static → per-loop **latency** oracle is not established, and the tested
~0.554B learned model is grossly unrepresentative of four-H100 deployment.
No scheduler, migration, replication or other optimization method is built.

This is not a universal proof that every larger Looped MoE has no EP problem.
It is a refusal to promote structural observations into an EP method claim.

## Exact model, runtime, sharing and hardware

- Checkpoint: `ShourenWSR/how-to-loop-moe`, `Foil-1`,
  `c1c857bab4665f00302f191fde5feb274ea97936`, Apache-2.0.
- Actual file: 553,730,048 BF16 learned-weight elements plus 9,437,184 FP32
  RoPE-buffer elements; 1,145,221,096 bytes. The ~563.2M model-card accounting
  includes buffers; learned parameters are ~553.7M. Full file/code hashes are
  in `results/model_manifest.json`.
- One physical core sparse pool: 64 experts, top-2, hidden1024, expert FFN1536,
  three matrices/expert, **9 MiB/expert**, 576 MiB/core pool. The same three
  registered tensors repeat at native **H=16**; core router and gain-free
  norms are shared, while 16 attention parameter sets are untied.
- One independent 8-expert head pool and one 8-expert tail pool run once each.
  They are included in whole-model timing, excluded from core divergence.
- Official HF native full-prefix, cache-free execution. No cached-serving
  implementation was invented. H is never altered; H1/2/4/8 are not trained
  loop-count controls for these weights.
- Runtime: existing Torch2.8.0/CUDA12.8 + transformers4.57.0 +
  DeepEP1.2.1+73b6ea4. Local `F.grouped_mm`→`torch._grouped_mm` compatibility
  alias; native forward/router/attention semantics otherwise unchanged.
  Original FP32 RoPE buffers are restored after HF dtype conversion.
- Validated normal DeepEP dispatcher from pinned Diffulex
  `9d1a04e794351f18039b3234a10e8b07e15bcbbf`, with the previous PoC's owner-local
  ID correction. No changes to that external checkout or shared environment.
- Topology: **TP1 replicated dense / source-SP4 / physical EP4 / DP1 / PP1**.
  All ranks cooperate on the same B1 request, not four independent requests.
  Sparse weights have unique owners: 16 core and 2+2 outer experts/rank,
  **180 MiB routed weights/rank**. Source rows are strided by token index %4;
  exact full-source restoration all-reduce is included and separately timed.
- Four H10080GB GPUs, all selected pairs NV18/NVSwitch. Logical 0/1/2/3 map to
  physical 4/5/6/7. UUIDs, topology, other-user occupancy and versions are in
  `results/environment.json`. GPUs0–3 and their users were not disturbed.

## Feasibility and representativeness

`model_runtime_audit.md` was written before substantial integration.
Foil fits comfortably on ONE H100; forcing its sparse path onto four H100s
creates a small, communication/API-overhead-heavy mechanism test.

The larger-model audit is not an unsupported absence claim. Loopie preview
links returned HTTP401 and the linked code subpaths HTTP404. LOOM/Tying-the-
Loop provide useful code but no directly usable larger download was identified
in the inspected READMEs. **IFM ~8.78B pretrained MoE artifact files are public**;
they were not loaded here. Their native xLLM code includes A2A, but the actual
artifact has **25 experts**, and native ownership requires expert count exactly
divisible by world size. Native EP4 therefore is not a valid drop-in. Unequal/
padded ownership, strict artifact loading and another inference runtime were
not engineered to rescue this bounded experiment. See exact source/config
revisions and availability evidence; do not claim no larger weights exist.

## Frozen workload and measurement protocol

WikiText-2 raw test revision `b08601e04326c79dfdd32d625aee71d232d685c3`:
**8 calibration +24 held-out** paragraph episodes, metadata-only seed3407,
no routing/output selection. Prefix lengths64/128; next-token probes at
teacher-forced prefix offsets0/4/8. These three native full-prefix forwards
form ONE timing episode. Overlapping positions are not new independent samples.
The pool is not claimed Wikipedia-article-disjoint or independent of pretraining.

Each arm/request has two warmup episodes, three clean repeats, three light
event repeats, one heavy trace. S0/S1 order alternates across requests; event/
clean order is also counterbalanced. Placement setup occurs before warmup,
never within a native loop. Both arms are static for the complete episode.
All reference forwards execute before sharding; no full reference expert
weights remain on GPUs during measured EP execution.

Clean wall uses the maximum rank-local elapsed wall after a pre-episode barrier
and final GPU completion. Stage accounting uses **one cumulative-critical
rank per request/static arm**, selected from all light-event repeats, for
D+E+C+source restoration. We do not sum unrelated rank/component maxima.
Per-loop longest-local-interval ranks are diagnostic, not cross-GPU aligned
absolute ready/completion timestamps or invented CWT. Critical-path causality
at production scale is not established.

CUDA events resolve only after an episode; no observer-added per-loop global
host sync. Heavy trace never supplies headline timing. Dispatch/Combine API
intervals include count/layout/control and host-launch gaps; these are not
pure payload movement or effective wire bandwidth. Sparse expert execution
includes reading expert weights. Logical byte accounting is destination-
deduplicated BF16 rows, verified against actual DeepEP receive rows/paths.
Logical bytes are not measured wire bytes or HBM traffic.

Statistics: 2,000 paired bootstrap resamples of **24 request episodes** after
averaging repeats. Ranks, loops, layers, tokens and repeats are not confidence
samples. Paragraph/topic dependence and small-model scope limit generality.

## Correctness and physical EP4

PASS: all four ranks have authentic unique-owner expert shards. Recorded
owner samples match the pinned checkpoint (72 pool/projection/rank/arm checks,
covering every owned expert). No routing, weight scaling, precision or expert
function approximation occurs. A static permutation changes transport slots
and physically loaded owner tensors, not the model's global expert IDs.

All **4,032 last-token logit-vector comparisons** in the held-out experiment
match the local native reference **bitwise**. Native router IDs/weights and
representative block residual samples match exactly; natural greedy4 token
continuation matches reference under both placements, including EOS/NFE length.
The 4,032 comparisons are correctness checks, not 4,032 independent requests.
Full branch-vector coverage is not claimed from sampled residual values.

The original failed publication-layout probe is retained; the contiguous
allocation seam is regression-tested and the identical tiny physical probe
then passes with zero logit error. CPU tests also cover destination dedup,
arbitrary owner maps, native H/capacity, frozen calibration provenance and
honest structural-only oracle labels. No reasoning/coding accuracy benchmark
or variable-H quality experiment is claimed; fixed-H placement math is exact.

## Stage A: real loop differences, but hot ranks are not a method signal

For the shared core pool on the initial prefix probe:

- Mean adjacent-loop expert-histogram JSD: **0.14695 bits**,
  request CI **[0.13390,0.16153]**.
- Matched half-sample between-loop minus within-loop JSD: **+0.03973 bits**,
  CI **[0.03599,0.04393]**. This is the sample-size-matched noise diagnostic.
  The retained full-sample-versus-half-sample statistic is NOT a valid
  matched-noise test and is not used to claim significance.
- Same-token adjacent-loop top-2 overlap: **35.3%**. Routing is genuinely
  different, which is already known in the model/prior literature.
- Assignment-hot rank switches between adjacent loops: **41.1% S0** and
  **58.9% S1**. MORE hot-rank switching under the better-balanced S1 does not
  imply worse execution; a changing argmax is not a critical-path diagnosis.
- Mean rank max/mean: **1.338 S0 →1.211 S1**; rank CV **0.246→0.165**.
  Rank edge density is already **0.99935→0.99995**, essentially saturated.

See `expert_routing_by_loop.png`, `rank_load_by_loop.png` and
`physical_loop_cost.png`. Full per-loop actual timing, assignment-hot ranks,
expert-time-hot ranks, fixed cumulative-critical ranks, active experts,
per-rank logical traffic and source/destination edges are in the compressed
CSVs and original four-rank route/weight traces.

## Physical S0 vs frozen global S1

All numbers below are **ms per three-prefix-forward episode**, NOT cached
decode TPOT, full generation E2E or deployment throughput.

| Arm | Clean wall | Dispatch | Owner expert | Combine | Routed D+E+C | Source restore |
|---|---:|---:|---:|---:|---:|---:|
| S0 contiguous | 205.539 | 47.503 | 9.428 | 16.181 | 73.112 | 6.164 |
| S1 frozen global proxy | 204.476 | 47.811 | 9.480 | 16.266 | 73.558 | 6.167 |
| S2 per-loop latency oracle | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |

S1 uses all calibration loop invocations jointly, with 16 experts/rank,
14 starts ×8000 bounded swaps. Its objective is sum of invocation-wise rank
assignment maxima, NOT just global average balance; remote deduplicated rows
break ties. A global-frequency heuristic is separately retained. Calibration
proxy objective S0 **25,170**, frequency-static **23,337**, best found **21,575**.
No held-out request is used to tune it. **This is not a certified best physical-
latency static placement**; the distinction remains explicit throughout.

Held-out S1 vs S0:

- Sum-max assignment reduction: **8.61%**, CI **[4.83,12.25]%**.
- Clean episode reduction: **0.42%**, CI **[-0.95,+1.86]%**.
- Actual routed EP reduction: **-0.81%**, CI **[-3.22,+1.70]%**.
- EP+restoration reduction: **-0.75%**, CI **[-3.00,+1.68]%**.

Thus the structural improvement does NOT translate into a material measured
latency gain. Sparse D+E+C is ~34.6–34.7% of event-lane wall, but expert time
is only ~4.6% of clean wall. That small-model API-heavy stage share is not a
large-model bottleneck claim. Dispatch being present/large is not evidence
that loop-conditioned assignment imbalance explains its cost.

By frozen initial prefix length (12 independent requests in each stratum),
S1 clean reductions are **+1.22% [-0.79,+3.37]%** for64 tokens and
**-0.38% [-2.02,+1.25]%** for128. Corresponding actual EP reductions are
**+0.25% [-2.88,+4.28]%** and **-1.87% [-5.23,+0.46]%**. Neither stratum
establishes a physical gain; source-rank/forward-phase details remain in the
CSV rather than being promoted to independent confidence samples.

Observer overhead: **2.79% S0 [1.72,3.72]%** and
**3.66% S1 [2.30,5.03]%**. Both remain below the frozen 10% invalidity ceiling;
S1's upper CI slightly exceeds the preferred5% target. The 0.4% point gain
must not be sold as speedup within this noise/observer scale.

## Stage B: what the oracle does and does not show

Future-truth per-invocation greedy capacity-balanced load candidates retain
16 experts/rank and fall back to S1 if worse. A separate perfect-balance
`ceil(paths/4)` value is an optimistic load lower bound, not guaranteed legal
attainable placement. Against S1:

- S2 candidate **structural assignment-max headroom: 17.28%**,
  CI **[16.27,18.35]%**.
- Perfect-balance **structural upper headroom: 17.35%**,
  CI **[16.33,18.41]%**.

Neither is EP latency, projected E2E, or executable speedup. The candidate
uses future truth and may change mapping per request/prefix/loop, so it also
contains generic request/time adaptation, not a clean proof of loop-specific
placement conflict. The heuristic is not a globally optimal timing oracle.
No unvalidated assignment→time model is used to fill the missing cell.
**Required S1→S2 EP timing headroom is UNVALIDATED; G2 does not pass.**

A nonlooped/equal-work physical control is NOT_RUN: representative-scale and
validated timing-oracle gates failed before Stage C causal promotion. We do
not assert generic-vs-recurrent uniqueness without that control. S1's measured
flat latency is useful negative evidence on THIS model, not proof of the
unknown true latency-optimal oracle on larger models.

## Movement feasibility, with favorable assumptions exposed

Current-GPU physical P2P probes cover all12 directed selected-GPU edges:
native three 3-MiB tensors (9 MiB/expert) plus96/384-MiB bulk buffers,
3 warmups and10 repeats. Single-expert three-copy service is about
**0.072–0.076 ms** per edge. Buffers are warm; this is not wire/DeepEP bandwidth.

For the future-truth sequence of ownership-changing candidate maps across
48 core invocations/episode, copy estimates average:

| Copy assumption | ms/episode |
|---|---:|
| Serial isolated expert-copy services | 158.12 |
| Perfect concurrent edges, expert-sized services | 22.32 |
| Zero-launch, fastest measured bulk bandwidth, perfect edges | 7.28 |
| Generously erase ALL core expert time, hold other stages fixed | 8.48 |

Zero-expert-envelope/copy ratios are **0.054/0.380/1.165×**, respectively.
The last is deliberately extremely favorable; it excludes all launch,
packing, allocation, synchronization and ownership-coordination cost, and
assumes the physically impossible benefit of zero expert computation.
It does NOT certify ROI<1 for every conceivable coalesced implementation,
nor does it show a useful deployable ROI. Actual load-balancing savings
would leave expert computation and must be measured, not inferred from17%.
No memory-moving runtime method or measured net gain exists here.

The reuse-count plot is an assumption sweep for compatible future work;
native H16 does not imply16 compatible reuse opportunities after a mapping
change. Earlier dLLM copy-ROI failures were considered, not substituted for
these measurements. Stage C and full serving E2E remain NOT_RUN.

## All final contract questions / stop decision

1. **Authentic checkpoint/sharing/H:** exact pinned Foil-1 above, H16, one
   shared core pool plus independent head/tail; actual loaded tensor geometry.
2. **Independent sharing:** core experts/router/norms shared, attention untied;
   no arbitrary weight tying or repeated random Qwen layers.
3. **TRUE EP4:** PASS, unique owner map, real D/C, native reference parity.
4. **Representative scale:** FAIL; <1.1GiB file on H100×4 is artificial.
5. **Loop differences:** yes, exact expert changes and matched JSD evidence;
   prior-known structural fact, not a method contribution.
6. **Physical straggler penalty:** not established as a material placement-
   sensitive cost; static load improvement has no significant latency gain.
7. **Hot ranks/graph:** actual per-loop ranks retained; graph already dense;
   assignment-hot ranks must not be equated to GPU completion stragglers.
8. **S1 removes S0 issues?:** improves load proxy8.6%, not time. Physical
   best-static optimality is not certified, so no overclaim of full resolution.
9. **Held-out S1→S2:** ~17.3% in assignment-load proxy only; timing UNVALIDATED.
10. **Clean decode/E2E attribution:** prefix episodes measured; source restore
    retained; cached TPOT/full-generation E2E NOT_RUN, no extrapolated speedup.
11. **Nonlooped control:** NOT_RUN before Stage C; no recurrent-only claim.
12. **Practical net method:** not demonstrated after movement/coordination;
    no-copy alternative also has no supported material timing target.
13. **Correctness/observer:** exact native/placement probes and all accounting
    pass; bounded observer overhead, no accuracy benchmark claim.
14. **Generality:** 24 metadata-frozen held-out paragraphs, both lengths;
    not article-disjoint, benchmark-wide or large-checkpoint evidence.
15. **Verdict/next step:** SMALL_MODEL_ONLY; stop method engineering. Reopen
    only if a representative authentic Looped-MoE native EP runtime becomes
    available and can demonstrate the missing best-static/physical latency
    premise. Do not invent a new placement, replication or routing trick here.

Code/trace/CSV/JSON/seven figures are under this PoC. See `reproduce.md` for
pins and exact commands, `prior_overlap.md` for verified boundaries, and
`results/verdict.json` for machine-readable gates. The surgical-change and
simplicity-first discipline kept the bridge to the native expert expression
and prevented an unsupported new serving-engine or larger-model rescue.

Branch: `flashvep/looped-moe-ep-poc`. The final handoff reports the containing
commit SHA; all unrelated original dirty/staged work remains preserved.
