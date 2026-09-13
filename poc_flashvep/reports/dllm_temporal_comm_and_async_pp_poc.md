# LLaDA2.0-Flash temporal communication and asynchronous PP PoC

## Final status

- **Track A — `NO-GO`**
- **Track B — `CHARACTERIZATION-SIGNAL`**
- **Overall — stop both as primary method directions; do not combine them.**

Track A reveals strong temporal destination locality but less than 1% gross
request-level payload headroom.  Track B reveals a large theoretical pipeline
ceiling and real boundary/phase asymmetry, but no quality-safe common policy
reaches 1.2x on both bounded tasks after NFE adjustment, and no live PP speedup
is claimed.

## Scope and evidence discipline

This study used only physical H100 GPUs 4/5/6/7.  It started from project
commit `fee91a74d353c4f3a9efc17f1be9d045d777bdd8`, dInfer substrate commit
`d5b8e66074ce6367ab8a66b3c2b85bc8c3e272e9`; the diagnostics are local dInfer
commit `6cce007` and are exported with the project.  The local
`LLaDA2.0-flash-744c3f8` checkpoint.  The checkpoint config is BF16, 32 layers,
H=4096, 256 routed experts, top-8, one shared expert, and MoE intermediate
1024.

Three evidence classes are kept separate:

1. **Measured live:** true-EP4 clean BCT/NFE, actual dispatch hidden traces,
   DeepEP payload curves, codec cost, stage-boundary transport, stage-local
   topology/HBM.
2. **Measured sequential approximation:** full trajectories and task outputs
   after stale-boundary intervention, still executed on TP4+EP4 rather than PP.
3. **Analytical:** exact flow-shop pipeline ceiling and NFE/stale-fraction
   adjusted policy speed upper.  These are not observed speedups.

Observer-heavy traces are excluded from clean request timing.

## Baseline

The strongest prior static setting was revalidated: dense TP4 + routed EP4,
DeepEP normal path, owner-rank fused expert execution and reverse combine,
batch=mini=32, gen=32, block=32, threshold=0.9.

| Task | Clean restart times | Median BCT | NFE | Bounded score |
|---|---:|---:|---:|---:|
| GSM8K-32 | 5.932 / 6.711 / 5.932 s | **5.932 s** | 66 | 5/32 |
| HumanEval-32 | 11.481 / 7.214 / 7.361 s | **7.361 s** | 86 | 6/32 |

The first HumanEval value is a cold outlier; median is used.  The low bounded
scores make relative-score percentages coarse, so sequence identity, pass
agreement, and NFE are reported as additional quality signals.

## Track A — temporal EP communication compression

### Phenomenon

The same-destination premise is verified.  Lag-1 remote cache-hit rate is
84.65% on GSM8K and 81.03% on HumanEval; late-phase hits rise to 93.20% and
91.70%.  Same-expert rates are 78.80% and 72.80%.  Hidden cosine p50 is
0.9627/0.9493, while hidden rel-L2 p50 is 0.2775/0.3221.  The delta is dense
rather than sparse.

FP8 delta reconstruction rel-L2 is about 0.27%, and row-scaled INT8 about
0.062%.  With K=16 refresh, only 39.68%/37.98% of **dispatch payload bytes**
can be removed because misses and full refreshes remain.

### Economic gate

Fresh DeepEP calibration shows p50 dispatch of roughly 0.10--0.26 ms and
combine of 0.15--0.26 ms from 4 to 4096 global rows; startup and shape costs
dominate much of this range.  The dispatch path is only about 4.72%/4.55% of
clean BCT, and compression touches only its payload-sensitive part.

| Task | Best zero-codec-cost gross request oracle | Feasible with measured unfused codec |
|---|---:|---:|
| GSM8K | **0.72%** | 0% |
| HumanEval | **0.37%** | 0% |

This is below the 5% kill gate.  No live cache protocol or trajectory-quality
test was built.  This obeys, rather than bypasses, the oracle-first contract.

### Novelty attack

[CompactFusion](https://arxiv.org/abs/2507.17511) already proposes residual
compression of step-wise diffusion activations, and
[DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html)
already exploits temporal activation/routing consistency and conditional
communication in diffusion MoE.  Dynamic destination-cache coherence would be
a narrower gap, but it has no economic mass here.

**Track A decision: `NO-GO`.**

## Track B — AsyncDiff-like approximate PP

### Topology truth

SGLang constructs valid PP2xEP2 groups and PP4 groups.  After a minimal loader
fix for non-local `PPMissingLayer`s, the model's stage-local parameters fit:

- PP2xEP2: 16 layers/stage, 128 routed experts/rank, 51.1--52.9 GiB/rank;
- PP4: 8 layers/stage, all 256 stage experts/rank, 47.7--52.4 GiB/rank.

Exact BF16 adjacent-stage transfer is 0.040--0.079 ms p50 for 0.25--8 MiB.
However, dInfer lacks a PP-aware diffusion return/KV contract, so these are
capacity/topology results, not a request-level PP baseline.  Current upstream
[vLLM LLaDA2 documentation](https://github.com/vllm-project/dllm-plugin/blob/main/docs/OPERATOR_LLaDA2.md)
also explicitly marks PP>1 unsupported.

### Pipeline ceiling and boundary stability

Measured layer times yield an ideal, zero-dependency flow-shop request ceiling
of 1.52x (PP2) and 2.13x (PP4) after fill/drain and Amdahl adjustment.  This is
large enough to test approximation.

Lag-1 boundary state is most stable after layer 8 and least stable after layer
24:

| Boundary | GSM cosine / rel-L2 | Human cosine / rel-L2 |
|---:|---:|---:|
| 8 | 0.99695 / 0.0824 | 0.99673 / 0.0867 |
| 16 | 0.98632 / 0.1715 | 0.98029 / 0.2089 |
| 24 | 0.96606 / 0.2659 | 0.94891 / 0.3276 |
| 32 | 0.98615 / 0.1783 | 0.97974 / 0.2164 |

Cosine is therefore insufficient as a safety metric.

### Full-trajectory policy sweep

Eighteen warm-up, refresh, phase, and boundary policies were executed as
sequential stale-boundary interventions.  With no refresh, PP4 drops GSM8K
5→3 and HumanEval 6→1, while increasing NFE by +52/+39.  Exact periodic
refresh restores coarse task scores but not trajectories.

| Policy | GSM score, NFE delta, upper | Human score, NFE delta, upper | Worst-task upper |
|---|---:|---:|---:|
| PP4 W4/K2 | 5/5, +0, 1.212x | 6/6, +14, 1.060x | 1.060x |
| PP4 W4/K8 | 6/5, +17, 1.291x | 6/6, +30, 1.184x | **1.184x** |
| PP4 late-only W4/K4 | 7/5, +5, 1.051x | 6/6, -2, 1.063x | 1.051x |

PP4 W4/K8 is the best cross-task score-preserving analytical point, but only
6.25%/31.25% of sequences are exact.  Its 1.184x worst-task upper is below the
1.2x characterization threshold and well below the requested 1.3x HOLD gate.
Real PP overhead can only lower it.

### Error structure and prior art

Periodic exact refresh does reset some benchmark error, but stale boundaries
change token acceptance and hence future NFE.  Late-only operation is safest
yet has too little mass; middle states offer more mass and more error.  This
phase/boundary interaction is a useful dLLM characterization, not a method.

[AsyncDiff](https://arxiv.org/abs/2406.06911),
[PipeFusion](https://arxiv.org/abs/2405.14430),
[DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html),
and [ParaStep](https://arxiv.org/abs/2505.14741) already occupy stale-state
diffusion pipeline/reuse space.  Discrete unmask trajectory and hybrid PPxEP
are a possible gap, but this PoC does not demonstrate the necessary Pareto.

**Track B decision: `CHARACTERIZATION-SIGNAL`.**

## Final recommendation

Do not implement either candidate further on the current substrate and do not
combine them.

- Track A is conclusively economics-limited, despite a real temporal signal.
- Track B should be retained only as a characterization and possible future
  revisit if a trustworthy exact LLaDA2 PP runtime becomes available.  A
  larger benchmark would also be required because 32-example score changes are
  too coarse.

The central negative result is informative: high activation similarity and a
large theoretical pipeline ceiling do not automatically produce a useful
system method.  In Track A the bottleneck is fixed communication cost; in Track
B the bottleneck is discrete trajectory error and added NFEs.

## Artifacts

- Detailed reports: `poc_dllm_temporal_comm_async_pp/reports/`
- Derived tables and figure:
  `poc_dllm_temporal_comm_async_pp/results/temporal_comm_async_pp_20260913_203553/analysis/`
- Raw result root:
  `poc_dllm_temporal_comm_async_pp/results/temporal_comm_async_pp_20260913_203553/`
- GPU accounting: `poc_dllm_temporal_comm_async_pp/GPU_TIME_LOG.csv`
- Logged task GPU wall: 1,511 s; 1.676 four-GPU-hours including bounded
  failed topology bring-up.
