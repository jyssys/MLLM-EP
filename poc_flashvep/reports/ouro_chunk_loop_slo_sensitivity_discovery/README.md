# Ouro C/G characterization and SLO sensitivity

This folder contains the follow-up to the frozen fixed-quantum study.
The primary engine is pinned vLLM-RLT, not optimized native-vLLM.
The model is Base Ouro-1.4B BF16 with static R4.
Four independent single-GPU workers use physical GPUs 4–7.
This is not TP, PP, or shared DP serving-capacity scaling.

## Results

- [Analysis and decision](analysis.md)
- [Benchmark summary](benchmark_summary.md)
- [C/G characterization](characterization.md)
- [SLO sensitivity](slo_sensitivity.md)
- [Natural serving tables](serving_tables.md)
- [Pareto interpretation](pareto_analysis.md)
- [Separate profiling](profiling.md)
- [Heavy quantum tables](profiling_tables.md)
- [Correctness and quality scope](correctness.md)
- [Independent decision gates](decision_gates.md)

## Contracts and reproduction

- [Frozen specification](inputs/authoritative_spec.md)
- [GPU-worker amendment and fairness](gpu_parallelism.md)
- [Timing contract](timing_contract.md)
- [Source-backed SLO definitions](literature_slo_contract.md)
- [Reproduction commands](reproduce.md)
- [Failures and limits](failures.md)
- [Evidence coverage](missing_data.md)
- [Implementation scope](implementation_scope.md)

Raw server files remain under the local `raw/` and `logs/` folders.
Lossless published archives and checksums preserve portable evidence.
The archive tool refuses to overwrite different existing evidence during extraction.
Full local Nsight captures remain separate from portable CSV exports.

The old `CONDITIONAL_FIXED_QUANTUM_ONLY` verdict remains unchanged.
The native-vLLM Ouro Loop PP program remains paused.
