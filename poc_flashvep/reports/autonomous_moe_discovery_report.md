# Autonomous MoE-EP discovery sprint

## Executive research judgement

This bounded discovery loop found no validated new optimization method.  The
most reproducible new observation is a *shape-regime interaction*: with the
same hidden activation, token count, total top-k assignments, and balanced
EP-rank work, DeepEP/TritonExperts latency changes sharply with the joint
`M × active experts × destination fanout × layer` geometry.  It is not a
monotone fragmentation tax.  A second, even larger observation is that the
first measured shape in a persistent worker can flip the sign of a comparison
(A32-first versus A2-first), so per-shape global warmup is mandatory before
any shape claim.

These are empirical diagnostics, not production opportunities yet.  The
routes in the fragmentation/fanout controls are synthetic route IDs replayed
through real Qwen3-VL weights and the real DeepEP high-throughput/TritonExperts
kernel path.  They must not be described as natural model-routing evidence.
No scheduler, routing, placement, dropping, merging, RL, or custom kernel
method was implemented.

## Scope and accounting

| Item | Value |
|---|---|
| Repository | `jyssys/MLLM-EP` |
| Branch | `flashvep/overnight-moe-ep-discovery` |
| Base commit | `dac63be` (previous sprint commit `8050042`) |
| Primary model | Qwen3-VL-30B-A3B-Instruct, snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Cross-model control | Qwen3-30B-A3B, snapshot `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Topology | TP2 / DP2 / EP4 / PP1 |
| Precision/backend | BF16, DeepEP high-throughput, TritonExperts, eager, DBO off, prefix cache off |
| Placement | linear, logical expert `e // 32` |
| GPUs | physical 1,2,3,4 only (`CUDA_VISIBLE_DEVICES=1,2,3,4`) |
| Successful live sessions | 22 bounded persistent-worker sessions |
| Live session wall time | ~28.7 minutes total (model load + replay) |
| Aggregate GPU time | ~1.91 four-GPU hours |
| New-route kernel measurements | 4-GPU CUDA-event replay; correctness and token partition checks passed |

The local `vllm_backend_matrix.py` received one compatibility-only fix so the
same controlled driver can tokenize both Qwen-VL processors and the generic
Qwen3 tokenizer.  It does not change model math or routing.

## Measurements completed

### H1/H1b/H1c — inverse fragmentation and its boundary

The original expanded Qwen3-VL layer-24 grid used M = 64, 128, 256, 512,
1024, 2048, 4096, 8192 and active experts per rank A = 1, 2, 4, 8, 16 (five
timed iterations per point).  The fair unique-ID baseline is A2; A1 duplicates
top-k IDs and is diagnostic only.  In that run, A2→A16 expert change was:

| M | Expert critical change |
|---:|---:|
| 64 | −7.5% |
| 128 | −46.5% |
| 256 | −47.3% |
| 512 | −42.3% |
| 1024 | −23.2% |
| 2048 | −5.0% |
| 4096 | −1.7% |
| 8192 | −6.2% |

An A32-equivalent diagnostic extended the active-expert axis.  The first
run appeared to show A2→A32 reductions of −54.7%, −35.7%, −22.7%, and
−16.6% at M=128…2048, but a 100-iteration repeat on the same route grid
reduced this to +3.7%, +9.3%, +3.6%, +1.5% (M=128…2048) and approximately
0% at M=4096/8192.  This is a direct falsification of treating the first
large effect as a stable law.

Layer controls still show regime dependence: the A2→A16 M=512 effect was
approximately null at layer 4 and about −1% at layer 44, whereas the A32
M=512 diagnostic can be fast after a different worker state.  The correct
conclusion is **H1 general law: NO_GO; H1b/H1c: HOLD diagnostic**.

### H16/H23 — worker-state/order control

Two runs contained exactly the same A2 and A32 M=512 cases, same activation,
routes, placement, and total/rank assignment counts.  Only first-case order
changed; each case had 50 measured repetitions after the local warmup.

| First case | A2→A32 expert | A2→A32 critical wall |
|---|---:|---:|
| A32 first | **+60.8%** | **+134.6%** |
| A2 first | **−37.3%** | **−54.0%** |

The sign flip is too large to be routing causality.  It is evidence that
first-use kernel compilation/allocator/workspace/stream state can dominate a
shape comparison.  This is a **STRONG diagnostic**, not an optimization
finding.  It explains why the earlier A32 positive result was not promoted.

### H5b/H5c — randomized temporal-state control

The same target route was measured 30 times after each of steady,
alternating, similar-vision, and disjoint-text primes, with all 120 target
cases interleaved by a fixed random schedule in one persistent worker.

| Prime condition | Target-B median (ms) | Change vs steady |
|---|---:|---:|
| steady | 0.8977 | 0.00% |
| alternating | 0.8880 | −1.08% |
| similar | 0.8853 | −1.38% |
| disjoint | 0.8976 | −0.01% |

The earlier similar-prime +27.8% signal disappears.  Position duration still
has Pearson −0.443 and up to 47.5% condition CV, so replay state remains a
measurement concern.  H5 route-history warmth is **REJECTED**; H5c position/
allocator state is **PROMISING diagnostic**.

### H6/H19/H22 — destination fanout geometry

For fanout F=1,2,4, the route controls keep M, top-k, total assignments, and
aggregate per-rank assignments identical.  They were replayed at layers 4, 24,
44 for M=128 and 512, plus M=1024 at layer24.  A separate M=512 layer24 run
used 20 measured repetitions.

At M=512, F1→F4 expert critical changes were −16.3% (layer4), −17.1%
(layer24), and −12.4% (layer44); the 20-repetition layer24 replicate was
−12.8%.  At M=128, the same F1→F4 change was +113.0%, +114.4%, and +104.2%.
The direction is therefore a reproducible M/fanout regime effect, not a
monotone “fanout tax.”  Phase aggregation across layers gives:

| M | F1→F4 dispatch | F1→F4 expert | F1→F4 combine | F1→F4 critical wall |
|---:|---:|---:|---:|---:|
| 128 | +3.3% | +104.8% | +10.3% | +29.9% |
| 512 | +12.1% | −14.3% | +281.1%* | −13.4% |
| 1024 | +36.2% | −5.8% | −11.7% | +2.4% |

`*` combine is unstable because the F1 baseline is very small; full critical
wall and expert phase are the reliable primary signals.  H6 original
“fanout tax” is **NO_GO**, while H19/H22 phase-specific fanout×M geometry is
**PROMISING diagnostic** and requires kernel-name/launch-state attribution.

### Generic Qwen3 control

The same controlled H1 path was run with Qwen3-30B-A3B.  At M=512, A2→A16
expert change was +5.9% while critical wall was −0.6%; at M=128, A2→A16
expert change was −55.6% with a small-shape tail.  A dedicated 20-repetition
A2/A16/A32 M=512 run gave A2→A32 expert −48.2% and wall −58.0% in that
worker state.  This confirms that the shape sensitivity is **not MLLM-only**;
it is a generic MoE execution diagnostic whose sign is state-sensitive.

### H15 expanded residual mining

The new recursive mining pass contains 3,840 phase rows and 900 unique
case/rank pairs (over 1,000 measurements, counting phase observations).  The
simple feature model gives:

| Phase | R² | RMSE (ms) | |residual| ≥10% |
|---|---:|---:|---:|
| dispatch | .092 | .0498 | 393 |
| combine | .197 | .1310 | 810 |
| expert | .529 | .1066 | 193 |
| wall | .529 | .2940 | 574 |

The data are repeated persistent-worker medians rather than independent
launches.  Expert/work volume explains about half of the variance; dispatch
and combine retain large regime residuals.  This supports H15/H19 as a
piecewise-kernel diagnostic, not a sufficient-statistics controller.

## Prior controls reused (not rediscovered)

The existing branch artifacts were used for H2/H4/H7/H8/H10/H11/H12/H13/H14
controls.  They show: shape features do not improve held-out error beyond
assignment count; route/order/spatial/partition oracle effects are ≤2.3%;
text/vision M curves both prefer M=512; and category/layer spreads are not
stable control-plane effects.  These controls are linked by provenance rather
than rerun in this loop.

## Hypothesis scoreboard

The machine-readable scoreboard is
`poc_flashvep/reports/autonomous_moe_discovery_scoreboard.csv`.
Status is deliberately conservative; a “PROMISING” diagnostic is not a GO
for a method.

| ID | Verdict | Effect / evidence | Causal clarity | Novelty | Next decisive test |
|---|---|---|---:|---:|---|
| H1/H1b | NO_GO / HOLD | inverse fragmentation not stable across repeat | 4 | 2 | global per-shape warmup + natural-route replay |
| H1c/H1d | HOLD / BLOCKED | apparent M/A boundaries; kernel names absent | 3 | 3 | kernel identity/launch capture |
| H4 | NO_GO | <5% route-transition effect | 4 | 2 | none unless runtime state control changes |
| H5b | REJECTED | randomized condition effects ≤1.4% | 5 | 2 | closed |
| H5c/H16 | PROMISING / STRONG_DIAGNOSTIC | position/order sign flip; CV up to 47.5% | 5 | 3 | controlled first-use state tracing |
| H6 | NO_GO | no monotone fanout tax | 4 | 2 | closed as stated |
| H19/H22 | PROMISING_DIAGNOSTIC | M512 expert −12…−17%, M128 +104…+114% | 4 | 4 | phase/kernel launch attribution |
| H2/H15 | NO_GO / HOLD | assignment model; R² .529 expert, residual clusters | 3 | 3 | pre-registered M×layer bins |
| H10 | HOLD | 45% layer spread, no universal rule | 3 | 3 | independent layer-normalized matrix |
| H13 | HOLD | 30% category spread, unstable | 3 | 3 | matched category remeasurement |
| H7/H8/H14 | NO_GO | prior order/partition/spatial ≤2.3% | 4 | 2 | do not pursue |
| H16/H23 | STRONG_DIAGNOSTIC | same pair, order flips sign by >100 pp | 5 | 3 | kernel warmup-state instrumentation |

## Top-10 findings and research ranking

Scores below are 1–5 for effect strength, robustness, causal clarity,
novelty potential, systems importance, and implementability.  They are also in
the scoreboard CSV.

1. **Shape-regime inversion is real but not monotone (H19/H22).**  Effect
   12–17% at M512 and >100% reversal at M128 under equal work; repeated across
   three layers.  Control fixes total assignments/rank load.  Mechanism is
   likely launch/packing geometry.  This is MoE/EP-specific operator evidence,
   not MLLM-specific, and prior-art risk is medium because grouped-GEMM tuning
   is crowded.  Cheapest decisive experiment: capture kernel names and launch
   state after global warmup, then replay natural routes near the same shapes.

2. **First-shape worker state can invert a result (H16).**  A2/A32 order
   flips expert effect from +60.8% to −37.3%.  This is surprising and has high
   causal clarity as a measurement diagnostic, but low novelty as a production
   phenomenon until the underlying state is identified.  It applies to generic
   MoE as well as Qwen3-VL.

3. **A32 positive result is not robust (H1b).**  One grid suggested −35.7%
   at M512; a 100-repetition repeat gave +9.3% and an order-controlled run
   gave opposite signs.  The correct research outcome is falsification of a
   naive fragmentation policy.

4. **Randomized route history does not produce warmth (H5b).**  The +27.8%
   similar-prime result vanished at n=30 per condition.  The common tail and
   position correlation point to allocator/first-use state instead.

5. **M512 fanout benefit is phase-selective (H19).**  Expert critical time
   improves while dispatch can rise; combine is too small/noisy at some points.
   Therefore “communication fanout” is not a single scalar cost.

6. **M128 is a distinct small-shape regime.**  Four-rank fanout and active
   experts can be dramatically slower than F1 at M128 although M512 reverses.
   This is likely more useful for a kernel regime taxonomy than a scheduler.

7. **Generic Qwen3 reproduces shape sensitivity.**  It weakens any MLLM-only
   novelty claim and is an essential control against modality storytelling.

8. **Expert phase is more predictable than communication phases.**  Expanded
   residual mining gives R²=.529 for expert/wall but .092 dispatch and .197
   combine.  Communication residuals are the strongest unexplained component.

9. **Layer variation is substantial but not a policy.**  Layer24 can show
   strong shape response while layer4/44 may be near null; this argues for
   per-layer kernel regime attribution rather than a global expert rule.

10. **Prior control-plane effects remain weak.**  Order, spatial, partition,
    and modality-granularity controls are ≤2–5% or volume-dominated, so they
    should not be revived without a new causal variable.

## Novelty filter and next queue

| Finding | Dense LLM? | MoE without EP? | EP required? | MLLM-specific? | Interpretation |
|---|---|---|---|---|---|
| H19 shape/fanout regime | likely yes | likely yes | not yet proven | no (generic Qwen control) | generic grouped-MoE execution geometry |
| H16 worker first-use state | likely yes | unknown | not established | no evidence | measurement/runtime-state issue |
| H5 route warmth | rejected | — | — | — | do not pursue |

The live queue remains in `ACTIVE_QUEUE.md` with H19, H20, H21, H22, H23 and
other pending controls; it deliberately has more than eight rows.  Each
completed experiment generated follow-ups instead of closing the loop.

## Checkpoint record

The sprint was started at 01:26 KST and the last bounded run completed at
02:49 KST.  The checkpoint files are retained even though this interactive
execution window did not reach literal T+2/T+4/T+6/T+8/T+10 wall-clock
milestones; they record the live evidence available at each analysis pass.
No result is represented as a 10–12-hour sweep.  Extending the run should
only be done for the decisive kernel-state and natural-route tests above, not
by repeating rejected hypotheses.

## Recommendation

**BEST NEW RESEARCH DIRECTION:** a kernel-regime atlas for MoE execution shape:
jointly model M, active experts, per-token EP fanout, and layer after a
strict, global per-shape warmup protocol.  The scientific contribution would
need to be a new causal interaction (especially a communication-phase
boundary), not “fragmentation-aware scheduling.”

**SECOND BEST:** instrument first-use/allocator/workspace/kernel selection
state and make all replay comparisons order-invariant.  This is necessary
before trusting any future positive result.

**MOST SURPRISING NEGATIVE:** the apparent +27.8% similar-route warmth
vanished under randomized n=30 interleaving.

**MOST SURPRISING POSITIVE:** F1→F4 fanout changed M512 expert time by
−12…−17% but changed M128 by +104…+114% under equal work.

**INTERESTING BUT PRIOR-ART-CROWDED:** generic shape tuning, routing-order
batching, expert placement, capacity/rerouting, chunked prefill, and TP/EP
switching.

**DO NOT PURSUE:** a fragmentation/fanout tax policy, route-history warmth
policy, or any method based on the first A32 positive run.

**NEW HYPOTHESES FROM RESIDUALS:** H19 phase-specific fanout regime, H20
kernel-launch boundary, H21 natural-route proximity to a fast band, H22
expert-versus-combine phase separation, and H23 global-warmup/order
confounding.  These are queued as bounded follow-ups, not claims.

