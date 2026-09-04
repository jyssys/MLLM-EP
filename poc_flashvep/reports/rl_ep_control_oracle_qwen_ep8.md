# Qwen3-30B-A3B EP8 control-action oracle PoC

## Executive result

**FINAL STATUS: NO_GO (for RL-policy headroom in this bounded control).**

The previous natural EP8 capture is a strong straggler testbed, and the
KEEP rerun reproduces it.  However, the only real temporary action measured
here is not quality/route preserving: it drops a median 12.16% of route slots
and changes the first greedy output in 15/24 measured driver records.  Its
lower expert stage sum is therefore not an admissible action gain; full driver
wall time is 4.74% slower.  The EPLB plan has a large *count-only* upper-bound
but a real one-expert migration broadcast costs about 37.26 ms, and the plan
was not installed in vLLM.  A conservative safe action oracle selects KEEP in
all 288 paired invocation/layer rows, so there is no evidence to justify RL
training in this branch.

This is a negative result for the controller gate, not a claim that the
underlying EP8 expert straggler does not exist.

## Configuration and provenance

| item | value |
|---|---|
| model | Qwen3-30B-A3B, checkpoint `/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| topology | TP2 / DP4 / EP8 / PP1 |
| GPUs | 8×H100, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| config | BF16, DeepEP high-throughput, TritonExperts, DBO off, prefix cache off, eager, linear placement |
| model facts | 48 routed MoE layers, 128 routed experts, top-8, 16 experts/GPU |
| baseline schedule | 6 real text conditions, 4 synchronized DP drivers, 2 repetitions (rep0 warmup, rep1 measured), one output token |
| route/timing source | KEEP and TEMP are actual Qwen worker traces; rank durations are CUDA-event durations, never cross-device timestamp subtraction |

The KEEP source was the validated EP8 result from
`rl_ep_straggler_qwen_ep8_poc_20260904_stage0_attempt7`; the new paired runs
are `rl_ep_control_oracle_qwen_ep8_20260904_keep` and `..._temp3`.
The exact reproducibility manifest and commands are in the result directory.

Reference source audits:

- Capacity-Aware-MoE: commit `9c73c8eee6ca64836eb873e77aa096fb4955e658`.
  Its score-ranked capacity/overflow semantics were followed by an
  experiment-only wrapper; the package was not integrated into vLLM.
- EPLB: commit `d52c72d5b2f2fb4c41afbf8eb21366820239913d`.
  `rebalance_experts` was run on the captured 48×128 logical-expert load.

## Stage 0: natural straggler and KEEP reproduction

The earlier full capture contained 576 invocation/layer views:

| expert CUDA max/mean | value |
|---|---:|
| median | 1.2867 |
| p90 | 1.4578 |
| max | 1.8779 |
| fraction ≥1.15 / ≥1.25 / ≥1.50 | 81.4% / 60.9% / 4.7% |
| rank assignment max/mean median | 1.9307 |
| rank ratio ↔ expert CUDA Pearson correlation | 0.577 |

The new KEEP run (288 measured invocation/layer views) remains strong:

| expert CUDA max/mean | value |
|---|---:|
| median / p75 / p90 / max | 1.3140 / 1.3892 / 1.4683 / 1.7634 |
| fraction ≥1.15 / ≥1.25 / ≥1.50 | 97.2% / 80.6% / 6.3% |
| median dispatch / expert / combine max time | 0.964 / 0.915 / 0.693 ms |
| median conservative stage sum (dispatch+expert+combine) | 2.579 ms |

Thus the natural EP8 testbed claim is reproduced before applying any action.

## Stage A: real TEMP_BALANCE

The hook is a bounded wrapper around the router's real logits.  It performs
over-selection, per-expert score-ranked clipping, and top-k among survivors;
KEEP leaves the stock route unchanged.  DeepEP and TritonExperts execute the
resulting route on all eight GPUs.

### Paired operator evidence

| metric (TEMP vs KEEP) | median change |
|---|---:|
| expert CUDA max time | **−22.44%** |
| combine max time | **−60.59%** |
| dispatch max time | **+42.36%** |
| conservative routed-MoE stage sum | **−12.64%** |
| rank assignment max/mean | **−39.80%** |
| TEMP expert CUDA max/mean | 1.0584 (vs KEEP 1.3140) |

The apparent stage reduction is not a usable balancing gain.  Across 4,608
router calls, TEMP changed 95.08% of assignment slots, invalidated/dropped a
median 12.16% of original slots (25% at the worst call), and the resulting
`assignments_after` was smaller than `assignments_before`.  The measured
driver-wall median was:

| action | driver wall median |
|---|---:|
| KEEP | 3,415.91 ms |
| TEMP_BALANCE | 3,577.84 ms (**+4.74%**) |

The first greedy output differed for 15/24 measured `(wave, DP rank)` records
(62.5%).  Therefore the lower routed stage is a route/quality-cost
diagnostic, not a positive control.  This is why the safe oracle marks TEMP
ineligible rather than converting dropped work into a false speedup.

Per-condition conservative stage-sum changes were heterogeneous: balanced
2K −22.66%, long-balanced −17.19%, long-math −26.31%, vision-proxy-long
−6.40%, but the heterogeneous and short conditions were +2.80% and +6.87%.
Dispatch expansion is the consistent counter-cost.

## Stage B: EPLB PERSIST_BALANCE plan and migration cost

The exact KEEP histograms were aggregated by layer and passed to official
EPLB.  Plans were generated for the preregistered small/large diagnostic
choices of 136/144 physical experts (17/18 slots per GPU).

| plan | baseline rank max/mean | predicted rank max/mean | count-proxy reduction |
|---|---:|---:|---:|
| EPLB_SMALL (136 physical) | 1.9290 | 1.0000 | 48.12% |
| EPLB_LARGE (144 physical) | 1.9290 | 1.0000 | 48.12% |

These are logical-load packing projections, not GPU latency.  A single
actual Qwen local expert's BF16 `w1` and `w2` tensors (`[1536,2048]` and
`[2048,768]`, 9,437,184 bytes total) were cloned and broadcast over the real
EP group:

| migration timing | ms |
|---|---:|
| median across 8 ranks | 37.26 |
| max across 8 ranks | 37.47 |

The benchmark is a one-expert two-tensor broadcast, not an EPLB placement
installation.  Charging one measured migration over four future windows adds
about 9.37 ms/window, while a measured routed-MoE stage is only about 2.4–3.1
ms.  Consequently the count-only PERSIST headroom is not amortized in this
bounded episode.  No end-to-end PERSIST latency or quality claim is made.

## Temporal structure and action oracle

The source schedule repeats deterministic domain prompts.  Adjacent-wave hot
expert recurrence is 1.0 with one unique hot expert per condition/layer, but
this is a repeatability check, not evidence of general persistence under
domain switches.  No new token-level route trace for an installed placement
was available, so a future-aware oracle cannot honestly be called GPU
validated here.

For transparency, the result directory contains both:

1. **Raw timing winner:** if route dropping is ignored, TEMP has the lower
   measured stage sum in 71.5% of rows.
2. **Safe action oracle:** any invalid route slot incurs a quality/route
   rejection; PERSIST includes the measured migration charge.  This oracle
   selects `KEEP` in 288/288 rows (100%).

The per-condition proxy summary is:

| condition proxy | safe best action | KEEP stage ms | TEMP raw stage ms | PERSIST predicted cost ms |
|---|---|---:|---:|---:|
| balanced_2k | KEEP | 2.423 | 1.874 | 11.384 |
| hetero_512_1k_2k_4k | KEEP | 2.397 | 2.517 | 11.375 |
| vision_proxy_long | KEEP | 2.976 | 2.730 | 11.844 |
| long_balanced | KEEP | 2.850 | 2.320 | 11.695 |
| short_mixed | KEEP | 1.908 | 2.020 | 11.006 |
| long_math | KEEP | 3.082 | 2.275 | 11.780 |

`PERSIST predicted cost` is explicitly a conservative counterfactual, not a
measured PERSIST action.  The large value is the one-expert migration charge
plus the EPLB count-proxy-adjusted expert term; it is intentionally not
presented as a production latency prediction.

### Gate

The fixed gate was applied without changing thresholds:

- dynamic safe oracle gain versus KEEP: 0%;
- safe action distribution: KEEP 100%, TEMP 0%, PERSIST 0%;
- raw TEMP timing does not pass route/quality validity;
- PERSIST migration was measured but no installed action outcome exists.

**RL_POLICY: NOT_RUN.**  The controller gate is `NO_GO_FOR_RL_HEADROOM` in
this bounded experiment.  Training a policy on the invalid TEMP route or on a
count-only PERSIST proxy would be methodologically unsound.

## Required final answers

| requested item | result |
|---|---|
| TEMP_BALANCE_GAIN | Expert stage −22.44%, but conservative full driver wall **−4.74%** (slower); route-invalid |
| PERSIST_BALANCE_GAIN_BEFORE_MIGRATION | 48.12% rank-load count proxy; no GPU latency claim |
| EXPERT_MIGRATION_COST | 37.26 ms median for one 9.44 MB expert broadcast (max 37.47 ms) |
| PERSIST_BALANCE_AMORTIZED_GAIN | not established; conservative charge removes projected benefit |
| HOTSPOT_PERSISTENCE | 1.0 only for repeated deterministic prompts; general sequential effect unsupported |
| BEST_ACTION_BALANCED | KEEP under safe oracle |
| BEST_ACTION_TRANSIENT | KEEP under safe oracle |
| BEST_ACTION_PERSISTENT | KEEP under safe oracle; PERSIST not installed |
| MYOPIC_VS_DYNAMIC_GAP | 0% safe-oracle gain; no meaningful gap |
| DYNAMIC_ORACLE_GAIN | 0% safe, with TEMP/PERSIST validity costs |
| ORACLE_ACTION_DISTRIBUTION | KEEP 100%, TEMP 0%, PERSIST 0% |
| RL_POLICY | NOT_RUN |
| SEQUENTIAL_EFFECT | REJECTED for generalization (repeat-only evidence) |

### Interpretation

1. A strong EP8 natural straggler exists, but a safe high-level action has not
   been demonstrated.
2. TEMP can flatten rank/expert counts, yet its current implementation drops
   work and expands dispatch enough to make serving slower.
3. EPLB can make the aggregate logical-count projection nearly perfectly
   balanced, but migration is not free and the actual placement path was not
   installed.
4. The action diversity needed for future-aware RL is absent after validity
   and migration costs are included.

**Next single action:** add a route-preserving Capacity-Aware intervention
that never emits invalid slots (capture alternate candidates and reroute only
within available top-k alternatives), then validate one highest-skew layer on
the same eight-GPU EP8 workload. Do not train RL until that bounded action has
a quality-preserving GPU result and an installed EPLB/PERSIST timing.

## Artifacts

- Analysis/result directory: `poc_flashvep/deepep_revalidation/results/rl_ep_control_oracle_qwen_ep8_20260904_final/`
- Key files: `action_pair_metrics.csv`, `driver_latency_paired.csv`,
  `temp_action_raw_summary.csv`, `migration_timings.csv`,
  `placement_plan.json`, `eplb_plan_summary.csv`, `action_oracle.csv`,
  `dynamic_episode_action_summary.csv`, `gate_summary.json`, and `figures/`.
- Original raw KEEP/TEMP/migration captures are preserved in their separate
  source directories and are referenced by `experiment_manifest.json`.
