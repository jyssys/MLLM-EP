# Atomic Prefill PD: Runtime and Scope Audit

Status: implementation and qualification are in progress.
This report does not establish a Prefill-heavy regime.

## Source boundary

The experiment branch is `flashvep/looped-pd-atomic-prefill-qps-crossover`.
Its base is `f9bdad32855e4af620cace1111b366c366039c40`.
The private RLT HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
The private working-diff SHA256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
No private engine file or previous experiment source is edited.
Process-local wrappers reuse the qualified numerical, KV, and NIXL paths.
Each new run stores its configuration, command, source archive, and terminal status.

The preservation contract protects 126,069 unrelated index entries.
It also protects previous source hashes, four previous reports, and the private runtime diff.
Its file is `looped_pd_atomic_prefill_qps_crossover/configs/preservation_contract.json`.

## Model and backend

| Model | Revision | Semantics | Backend |
|---|---|---|---|
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | Native AR, BF16 | Installed vLLM 0.26, native NIXL |
| Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | Fixed R4, BF16 | Qualified RLT, FA4, native NIXL |
| Huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | Fixed R32, BF16 | Qualified RLT, FA4, native NIXL |

The Python environment is `/home/esjung/.venvs/looped-pd-abc`.
The installed stack uses Python 3.14.6 and PyTorch 2.11 with CUDA 12.9.
The qualified Looped numerical variants remain `down_norm` and `down_norm_adapter`.
Their selected projections use fixed 256-row GEMM tiles.
A full logical Prompt therefore does not imply one large physical Down GEMM.

The new wrapper removes the old harness's 4,096-token assertion for Ouro only.
It does not extend Huginn's 4,096-token context or alter a checkpoint.
Ouro's checkpoint limit is 65,536 tokens.
That configuration value is not a successful runtime qualification.

## Atomic execution

Primary Looped runs use native Atomic Prefill for each token chunk.
Full-Prompt runs must submit exactly one `[0,L)` Prompt span per request.
The observed `ExecutionBatch.loop_count` must equal R4 or R32.
Small-C runs keep the same Atomic-per-chunk recurrent policy.
The older staged-G1 path is an explicit correctness control.
It is not relabeled as native Atomic execution.

The internal `arm=PD_G1` selector retains native transport.
Its string is not evidence that the new P execution uses loop group G1.
Human-facing results must use the actual span and loop-count evidence.

Full-Prompt PD can still use multiple NIXL fragments.
It can also use many Attention, Linear, Norm, and sampling kernels.
Atomicity refers to logical Prompt chunking, not kernel or wire-fragment count.

The installed native AR scheduler mutates computed-token counters before returning.
The span hook now snapshots those counters before scheduling.
The original empty-span diagnostic remains preserved.
A new GPU namespace verifies the corrected observer.

## Memory compatibility controls

The first Huginn 3,900-token qualification failed with CUDA OOM.
The 26,624-block pool competed with large Prefill activation buffers.
The retry uses 24,576 blocks and a 7,800-token aggregate budget.
All later Huginn methods use the same memory control.
Weights, BF16, R32, context, and Prompt length remain unchanged.
The failed runs remain in `raw/` with their original logs.

Native Qwen PD and DP2 initially matched 7/8 sequences at 8K.
A repeated default PD matched the first default PD in 8/8 sequences.
These two results do not identify the divergent operation.
The third default native-PD start also matched both earlier starts in 8/8 sequences.
The official `VLLM_BATCH_INVARIANT=1` control failed its native-PD repeat at 7/8.
That optional arithmetic mode is not used for capacity or Serving.
The default native colocated repeat also failed at 7/8.
The default native-PD family and the failed colocated family remain separate.

All current native AR runs explicitly use `async_scheduling=False`.
Here, "default arithmetic" means that batch-invariant arithmetic is disabled.
It does not mean that every option equals the vLLM production default.
No synchronous-versus-asynchronous AR performance result is currently available.

## Data and generation

The frozen pool contains 184 distinct public sources.
It contains 28 ShareGPT sources and 156 LongBench sources.
The LongBench task counts are hotpotqa 64, multifieldqa_en 52, narrativeqa 20,
qasper 14, and 2wikimqa 6.
The common pool is shuffled before the 24/160 calibration/test split.
Actual tokenizer-prefix overlap across these two splits is checked.

Only 29 ShareGPT records passed the initial common 8K availability rule.
LongBench supplementation was declared before held-out Serving results.
No undisclosed filler or repetition was used to construct that source pool.
Exact token-prefix matching is a controlled workload.
It is not a complete natural conversation or an Azure text trace.
All measured main outputs use `ignore_eos=True` and the stated cap.

Larger offered cohorts can use an explicitly labeled cyclic test pool.
Such requests are not independent document samples.
Generation quality, original-HF fidelity, and downstream accuracy are not established here.

## Isolation and resource safety

Eight H100 80GB devices are available.
The initial UUID, physical-ID, and NVLink topology audit is preserved in `state/initial_hardware.json`.
Observed GPU pairs have NV18 connectivity.
Burner ownership is checked by UID, session, command, and approved output log.
No unrelated GPU job is signaled.

Disjoint-pair correctness runs can share the host.
They are not clean latency measurements.
Independent service, SLO calibration, QPS, and final timing runs take an exclusive quiet-lane lease.
Approved burners remain off during those measurements.
An idle watcher restores approved burners after managed work ends.
The restore path explicitly selects all eight GPUs.
The shell script's current default device subset is not assumed to mean all eight.

## DuetServe comparison boundary

[DuetServe](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)
compares Qwen3-8B on two H100 GPUs with 8000 input and 200 output tokens.
Its aggregated control uses an 8,192-token budget.
Its disaggregated control uses Dynamo with 1P+1D.
This study uses native vLLM/NIXL and also tests larger tuned aggregate budgets.
Reproduction requires the same bottleneck mechanism, not identical absolute latency.
An engine or numerical-mode difference is not a causal recurrence effect.

## Current limitations

The QPS search and final independent-start measurements are not complete.
No measured sustainable capacity or Dynamic A benefit is claimed.
No Dynamic A, Loop Split, Token Split, new TP/PP, or training is implemented.
RLT has separate-phase colocated controls and an existing Native Mixed control.
The new Native Mixed worker applies the same numerical fixes without the old Dynamic-A Mixed prohibition.
It uses the original RLT Mixed scheduler and recurrent core.
Ouro and Huginn each pass eight-request output and full-depth gates at C256 and C1024.
Actual mixed-core invocation counts are 53/30 for Ouro and 8/4 for Huginn, respectively.
These are executed qualification counts, not clean efficiency measurements.
Native Mixed is an existing feature. It is not a new method in this study.
The future final colocated setting requires another independent 32-request gate.
Native AR DP2 uses its supported native mixed/chunked scheduler, but its repeat mismatch blocks automatic performance expansion.
