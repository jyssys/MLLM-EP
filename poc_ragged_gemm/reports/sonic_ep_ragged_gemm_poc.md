# Routing-Preserving Ragged Grouped GEMM for EP Inference — bounded PoC

## 1. Research question and final answer

Can H100/BF16 MoE expert compute be reduced by specializing ragged tails while
preserving pretrained routing, dispatch, placement, scores, weights, and every
token/expert assignment? In the measured Qwen3 regime, **no**. Tails are
structurally ubiquitous, but the presumed full-tile cost is not exposed as a
material causal latency staircase. The measured-tail exact-routing oracle is
only 1.007x median and 1.018x p95 at layer makespan.

`FINAL METHOD STATUS: NO-GO`

Inference-only evaluation is valid for this question because expert M sizes are
fully determined by inference-time top-k routing, while weights and routing
semantics stay immutable. No training claim is made.

## 2. Difference from prior methods

[SonicMoE](https://github.com/Dao-AILab/sonic-moe) changes token/expert counts
to align them to a tile. This PoC instead asks whether the *original* counts and
assignments can execute faster. TEMPO is a makespan-aware dispatch/placement
method; this PoC does not change dispatch or placement. DA-MoE adaptively
selects kernels from routing shape; it establishes that shape matters but is
not an exact-routing tail implementation.

The novelty risk is high. [TMA-Adaptive FP8 Grouped
GEMM](https://arxiv.org/abs/2508.16584) already handles arbitrary residual rows
with an adaptive descriptor scheme on Hopper, preserves valid outputs, and
reports 1.7–20.4% gains in FP8. NVIDIA also documents an exact-offset grouped
GEMM interface for [ragged contiguous M
groups](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/005_grouped_gemm_contiguous_offset.html),
currently demonstrated on Blackwell. A future contribution would have to be a
specific H100/BF16 MoE inference advance over these mechanisms, not the generic
idea of ragged grouped GEMM.

## 3. Environment and artifacts

* Sonic source: `7396f3e604827d8186c2e16e64b28ee33d3defd0`, clean external checkout at
  `/home/esjung/external/sonic-moe`; it is not vendored.
* GPU: four H100 80GB; only physical GPUs 4,5,6,7 were exposed. Synthetic runs
  used physical GPU 4.
* Main stack: Python 3.12.13, PyTorch 2.11.0+cu129, CUDA 12.9, Triton 3.6.0,
  vLLM 0.20.0, FlashInfer 0.6.8.post1.
* Sonic compatibility run: QuACK 0.5.0 / CUTLASS DSL 4.5.3. Sonic main's
  declared QuACK 0.6.4 / DSL 4.6.2 was tested only in an isolated environment
  and hit a CUTLASS DSL kwargs-wrapper ABI error during first compilation. The
  vLLM environment was not modified.
* Synthetic shapes: BF16, E=128, K=8, G=32, N=4096, M tile 128;
  Sonic-like H/I=4096/1024 and Qwen3 H/I=2048/768. Timings use 20 warmups and
  100 measurements.
* Qwen checkpoint revision:
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.

## 4. Qwen3 model and actual EP topology

The downloaded configuration is `Qwen3MoeForCausalLM`, BF16, 48 layers,
H=2048, I=768, 128 routed experts, top-8, and normalized top-k probabilities.
The simplest validated EP4 configuration in this vLLM build is
TP4/DP1/EP4/PP1. Runtime logs and expert maps verify linear, disjoint placement
with 32 local experts per rank (rank 0 owns 0–31, and so on). DBO and prefix
caching were off.

vLLM auto-selected `FlashInferExperts` with the generated H100 CUTLASS
unquantized M128 family rather than TritonExperts. The live measurement is still
the requested real local expert path: CUDA events surround only expert compute,
after dispatch and before combine; top-k IDs and weights are read but never
changed. Dispatch/combine were not mixed into the primary timing.

## 5. POC0 source audit

Sonic passes cumulative variable-M offsets to QuACK grouped GEMMs. The checked
SM90 launch uses one launch-wide M tile (128 by default; 256 is an available
configuration), not a heterogeneous tile per expert. QuACK schedules each group
using ceiling division and predicates invalid addressing. Sonic's actual
token-rounding benchmark fixes M tile 128 and rewrites counts using nearest,
up, or down rounding.

This source evidence proves quantized scheduling and masks, but does not by
itself establish that all invalid rows expose a full tensor-core cost. The
causal timing below answers that question: 50% more arithmetic tiles did not
increase the end-to-end expert kernel latency, so “one partial tile equals one
fully exposed tile of hardware work” is false for the tested persistent kernel
regime.

See `source_audit.md` in the result directory for detailed overlap notes.

## 6. POC1 — iso-N/iso-G tile causality

`POC1_TILE_CAUSALITY: NO-GO`

| Shape | Aligned, Q=32 | Boundary, Q=48 | Boundary penalty | Largest same-N/G spread |
|---|---:|---:|---:|---:|
| Sonic-like 4096/1024 | 0.469632 ms | 0.467952 ms | -0.358% | 9.381% |
| Qwen3 2048/768 | 0.264608 ms | 0.263456 ms | -0.435% | 5.400% |

N, G, weights, dtype, GPU, and kernels are identical inside each pair. Only
the per-expert histogram changes. Despite Q rising from 32 to 48, the
boundary-heavy case is slightly faster. At the 1x/2x/3x M-tile boundaries, the
neighbor-to-boundary changes are respectively +0.55%/-1.41%/-5.28% for the
Sonic-like shape and +0.35%/+0.05%/+1.04% for Qwen3. No consistent staircase
appears. The isolated QuACK 0.6.4 rerun was blocked by the ABI issue above.

## 7. POC2 — tile-cost explanation

`POC2_TILE_EXPLANATION: HOLD`

| Shape | N-only CV R² | N+G CV R² | N+G+Q CV R² | Full/tail CV R² |
|---|---:|---:|---:|---:|
| Sonic-like | 0.8028 | 0.8028 | 0.8140 | 0.8112 |
| Qwen3 | 0.8263 | 0.8263 | 0.8371 | 0.8346 |

Adding Q improves CV R² by only +0.0112 and +0.0109, reducing RMSE by roughly
3%. This is repeatable explanatory signal, hence HOLD, but it is too small and
inconsistent with a causal 10% staircase. These regressions are descriptive,
not a novelty claim; TEMPO/DA-MoE already cover generic shape-aware costs.

## 8. POC3 — real Qwen3 EP trace and stragglers

`POC3_REAL_TRACE_HEADROOM: NO-GO`

The live suite covers short/medium/long prefills, low/medium/high request
counts, three measured repetitions, 48 layers, and all four ranks: 5,184
rank-layer observations and 1,296 layer groups.

* Active experts with a partial tail: 99.815%.
* Effective tiles that are tails: 81.600%.
* Padded rows / arithmetic effective rows: 61.836%.
* Padding amplification: median 3.384x, p95 14.441x.
* Assignment/latency Spearman: 0.8169; Q/latency Spearman: 0.9198.
* Token-argmax actual critical-rank match: 44.37%; Q-argmax: 50.62%.
* Token and Q critical ranks differ in 34.03% of layers; Q corrects a wrong
  token prediction in 14.20% of all layers.

The structural tail prevalence and stronger Q correlation are positive. They
do not establish removable compute: the measured-tail oracle is only 1.008x
median/1.017x p95 per rank and 1.007x median/1.018x p95 at layer makespan.
Therefore the practical headroom criterion fails despite large arithmetic
padding statistics.

## 9. POC4 — exact-routing tail oracle and Sonic counterfactual

`POC4_EXACT_ROUTING_ORACLE: NO-GO`

The oracle uses measured single-group small-M costs for rows
1,2,4,8,16,32,48,64,96,127,128 and interpolates between them; it does not
assume linear FLOPs. On live Qwen histograms, layer-makespan speedup is 1.007x
median and 1.018x p95. Synthetic Qwen histograms show 1.004x median and 1.012x
p95. Both are below even the 5% HOLD region.

The actual Sonic nearest-rounding implementation at T=2048 changes 3.815% of
original token/expert assignments and reduces arithmetic tiles by 32.275%
(189→128), but latency changes from 0.671930 to 0.678694 ms: 0.990x, a
regression. Up/down variants change 23.83% of assignments and are not comparable
pretrained-inference semantics. T=512 nearest/down collapses counts to zero and
is explicitly excluded as a meaningful baseline. Since the meaningful Sonic
nearest case has no speedup, “fraction of Sonic speedup recovered” is N/A; the
zero-edit oracle is only ~0.7% at live makespan.

## 10. POC5 — conditional dual path

`POC5_DUAL_PATH: NOT RUN`

POC1, POC3, and POC4 did not all reach HOLD. Per the preregistered stop rule,
no custom kernel, dual launch, three-way performance claim, correctness claim,
or EP prototype replay was implemented. Routing edit is zero for every main
measurement because routing was never mutated.

## 11. Figures and interpretation

* `plot1_iso_ng_tile_causality.png`: same-N/G histograms do not show an aligned
  advantage.
* `plot2_boundary_staircase.png`: boundary crossings are noisy/non-monotonic.
* `plot3_tile_features_vs_latency.png`: Q has modest descriptive value.
* `plot4_real_ep_tail_distribution.png`: arithmetic tails are extremely common.
* `plot5_real_ep_tile_vs_straggler.png`: Q improves rank identity modestly.
* `plot6_exact_routing_oracle_headroom.png`: live measured-tail oracle remains
  below 2% through p95.
* `plot7_sonic_routing_edit_tradeoff.png`: tile reduction is not equivalent to
  kernel speedup in this regime.

## 12. Numerical correctness and semantic scope

The synthetic main path feeds the exact same row offsets, expert weights, and
expert counts to the stock Sonic/QuACK kernel. The live hook is read-only and
does not change expert IDs, weights, scores, dispatch order, or outputs. Because
POC5 was gated off, no alternate numerical output exists to compare; it would
be misleading to claim prototype equivalence metrics.

## 13. Evidence, limitations, and decision

The strongest positive evidence is structural: 99.8% of active experts have a
tail, Q correlates with expert latency at 0.920, and Q improves actual
critical-rank match by 6.25 percentage points. The strongest counter-evidence
is causal: raising Q by 50% at fixed N/G produces -0.4%, not +10%, latency, and
the live measured-tail makespan oracle is only 0.7% median.

Limitations are the QuACK 0.5 compatibility measurement versus Sonic main's
declared 0.6.4 stack, automatic use of FlashInfer rather than TritonExperts in
live vLLM, bounded text workloads, and no Nsight Compute hardware-counter run.
These limitations cannot turn a measured <2% oracle into evidence for a 10%
prototype gate; they do mean the conclusion is scoped to the tested H100/BF16
kernel families rather than every grouped GEMM implementation.

`NOVELTY_RISK: HIGH`. Exact residual-M methods already exist in FP8 Hopper work
and current CUTLASS APIs. Combined with the failed headroom gate, a custom
RaggedGEMM kernel is not justified now.

## 14. Next single recommended action

Do **not** implement the dual-path kernel. First reproduce the iso-N/G boundary
sweep on Sonic's officially supported QuACK 0.6.4 environment/container; only
reopen this direction if it shows a reproducible ≥5% staircase for the real
Qwen3 H=2048/I=768 BF16 shape.
