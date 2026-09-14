# Environment and substrate truth

## Isolation and hardware

Every task-owned CUDA launch used exactly `CUDA_VISIBLE_DEVICES=4,5,6,7`.
Logical ranks 0--3 therefore mapped to physical GPUs 4--7 in order. GPU 0--3
were never used. Before every campaign the task-owned utilization workers were
identified by owner/PID/command and stopped; no unrelated process was killed.

| physical GPU | UUID | peer links |
|---:|---|---|
| 4 | `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139` | NV18 to 5/6/7 |
| 5 | `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730` | NV18 to 4/6/7 |
| 6 | `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` | NV18 to 4/5/7 |
| 7 | `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` | NV18 to 4/5/6 |

All devices are 80 GiB H100s. The measurement environment is PyTorch
`2.8.0+cu128`, CUDA runtime 12.8, and NCCL 2.27.3. `nvcc` is not installed in
the active shell; this did not affect execution of the already-built DeepEP
extension.

## Revisions

- Checkpoint: `/home/esjung/models/LLaDA2.0-flash-744c3f8`, revision
  `744c3f8c6c8317d2377d6d16d8a3d4be2caef563`.
- Model config: BF16, 32 transformer layers, hidden 4096, 256 routed experts,
  top-k 8, one shared expert, MoE intermediate 1024, 8 routing groups/top-4
  groups, context 32768.
- dInfer base: `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`.
- Instrumented dInfer lineage: `9132ce9b2580ac2e64bcaaf975b863a5005c7739`.
- DeepEP legacy v0.2 source: `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`.

## True-EP path

The baseline is dense TP4 + routed EP4 + DP1. Each rank owns a contiguous set
of 64 complete routed experts; the shared expert is replicated. The observed
path is DeepEP dispatch, owner-rank fused routed-expert execution, reverse
DeepEP combine, then exact TP gathering. Remote assignments are nonzero on all
ranks. This is not TP-only execution presented as EP.

The valid comparison paths were DeepEP normal and legacy low-latency BF16. The
low-latency buffer printed IBGDA initialization warnings because this is a
single-node run; it then used the explicitly enabled intranode NVLink/P2P
fallback. Current DeepEP V2 was not benchmarked because it requires a newer
software contract than the validated PyTorch 2.8 environment. This is recorded
as an environment boundary, not a method failure.

## Strongest request baseline

The inherited best-static configuration is submitted batch 32,
`mini_batch_size=32`, generation/block length 32, threshold 0.9, deterministic
decoding. It was selected by an earlier static sweep, not chosen as a weak
default.

| task | independent clean restarts (s) | median | NFE | bounded quality anchor |
|---|---|---:|---:|---:|
| GSM8K-32 | 8.839, 5.833, 5.925 | 5.925 s | 66 | 5/32 exact answer |
| HumanEval-32 | 7.261, 7.193, 7.367 | 7.261 s | 86 | 6/32 pass@1 |

The first GSM8K run is a retained cold outlier. The median is the request
anchor. The small bounded scores are substrate consistency checks, not claims
about full-benchmark model quality. Observer-heavy runs are excluded from E2E
claims. Earlier identical-substrate tracing attributed 47.94%/49.54% of clean
GSM8K/HumanEval time to dispatch + routed expert + combine, which defines the
maximum relevant Amdahl mass.
