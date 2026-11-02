# Atomic Prefill PD: Final Runtime Audit

Compiled at 2026-11-02T01:28:43.086700+00:00. Elapsed time: 21.426 hours.

Branch: `flashvep/looped-pd-atomic-prefill-qps-crossover`.

Compilation source commit: `147c4dc48215dd82cc85f77b044ff27325ea84fc`.

Each measured run records its own source archive, source commit, and configuration hash.
The final publication commit can differ from the measurement commit.
Use `git log -1 --format=%H` on this branch to obtain the publication SHA.

Actual analysis-tree SHA256: `a392ed79e603c437d00c3575d39166f104f681e2144fe757f6ef0524e7bc5c18`.

The compilation HEAD can precede uncommitted analysis changes.
See `state/analysis_source_manifest.json` for the exact file hashes.


| Model | Revision | Depth | Precision |
| --- | --- | --- | --- |
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | 4 | BF16 weights and KV |
| tomg-group-umd/huginn-0125 | bb6621b65e90b6a4b9b29ef88dc83866d450470c | 32 | BF16 weights and KV |
| Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 | 1 | BF16 weights and KV |


## Preserved sources

Private runtime HEAD: `6b164d3844d35ab8e753df1864f0c87dc180401a`.

Private working-diff SHA256: `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.

The preservation check protects 126069 unrelated index entries.
Previous source, reports, raw evidence, and checkpoints remain unchanged.
Process-local wrappers add observations and supported configuration controls.

## Execution boundary

Ouro and Huginn use the qualified RLT/FA4 native NIXL path.
Qwen uses installed native vLLM 0.26/FA3 and NIXL.
The Python environment uses Python 3.14.6 and PyTorch 2.11 with CUDA 12.9.

The numerical variants remain `down_norm` and `down_norm_adapter`.
Selected projections retain fixed 256-row GEMM tiles.
A logical Full-Prompt span therefore does not imply one large Down GEMM.

The internal arm string `PD_G1` selects native transport.
Actual Looped Full-Prompt P submissions use `loop_count=4` or `32`.
They are not staged-G1 Prefill solely because the arm string contains G1.

Native Mixed is an existing RLT baseline, not a new method.
Separate-phase and Mixed colocated controls remain distinct.
Their actual scheduler, wavefront flag, token budget, KV pool, and batch histogram remain recorded.

## Supported generic controls

The inherited harness allocated P-sized row workspaces on a Decode-only worker.
The D128 control uses the supported Decode token budget of 128.
Larger legal KV pools are separate settings, not free memory.

Wire-credit controls change `PDConfig.max_inflight_bytes`, not the NIXL protocol or 64-MiB fragment limit.
No transport or memory control is a Loop-specific scheduling contribution.

## GPU and data provenance

```text
index, name, uuid, memory.total [MiB], memory.used [MiB], utilization.gpu [%], driver_version
0, NVIDIA H100 80GB HBM3, GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d, 81559 MiB, 71551 MiB, 100 %, 570.211.01
1, NVIDIA H100 80GB HBM3, GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb, 81559 MiB, 71551 MiB, 100 %, 570.211.01
2, NVIDIA H100 80GB HBM3, GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e, 81559 MiB, 71551 MiB, 100 %, 570.211.01
3, NVIDIA H100 80GB HBM3, GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8, 81559 MiB, 71551 MiB, 100 %, 570.211.01
4, NVIDIA H100 80GB HBM3, GPU-6076e2f2-5b63-3761-5586-56ceb7df8139, 81559 MiB, 71551 MiB, 100 %, 570.211.01
5, NVIDIA H100 80GB HBM3, GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730, 81559 MiB, 71551 MiB, 100 %, 570.211.01
6, NVIDIA H100 80GB HBM3, GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28, 81559 MiB, 71551 MiB, 100 %, 570.211.01
7, NVIDIA H100 80GB HBM3, GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b, 81559 MiB, 71551 MiB, 100 %, 570.211.01
```

```text
[4mGPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	NIC6	NIC7	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NV18	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	0-191	0		N/A
GPU1	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NV18	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	0-191	0		N/A
GPU2	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NV18	PHB	PHB	PIX	PHB	PHB	PHB	PHB	PHB	0-191	0		N/A
GPU3	NV18	NV18	NV18	 X 	NV18	NV18	NV18	NV18	PHB	PHB	PHB	PIX	PHB	PHB	PHB	PHB	0-191	0		N/A
GPU4	NV18	NV18	NV18	NV18	 X 	NV18	NV18	NV18	PHB	PHB	PHB	PHB	PIX	PHB	PHB	PHB	0-191	0		N/A
GPU5	NV18	NV18	NV18	NV18	NV18	 X 	NV18	NV18	PHB	PHB	PHB	PHB	PHB	PIX	PHB	PHB	0-191	0		N/A
GPU6	NV18	NV18	NV18	NV18	NV18	NV18	 X 	NV18	PHB	PHB	PHB	PHB	PHB	PHB	PIX	PHB	0-191	0		N/A
GPU7	NV18	NV18	NV18	NV18	NV18	NV18	NV18	 X 	PHB	PHB	PHB	PHB	PHB	PHB	PHB	PIX	0-191	0		N/A
NIC0	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X 	PHB	PHB	PHB	PHB	PHB	PHB	PHB
NIC1	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X 	PHB	PHB	PHB	PHB	PHB	PHB
NIC2	PHB	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X 	PHB	PHB	PHB	PHB	PHB
NIC3	PHB	PHB	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X 	PHB	PHB	PHB	PHB
NIC4	PHB	PHB	PHB	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X 	PHB	PHB	PHB
NIC5	PHB	PHB	PHB	PHB	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X 	PHB	PHB
NIC6	PHB	PHB	PHB	PHB	PHB	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X 	PHB
NIC7	PHB	PHB	PHB	PHB	PHB	PHB	PHB	PIX	PHB	PHB	PHB	PHB	PHB	PHB	PHB	 X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
```

The displayed topology omits trailing whitespace only. The original output remains in `state/initial_hardware.json`.

Correctness can use disjoint pairs concurrently.
Clean service, calibration, and QPS measurements use one exclusive quiet lane.
Approved burners are absent from that lane.

Unrelated processes are not signaled.
The common pool contains 184 public sources: 28 ShareGPT and 156 LongBench.
The split contains 24 calibration and 160 test sources.

Exact model-tokenizer prefixes define input lengths.
Optional 16K uses its own smaller test pool and an eight-source calibration extension.
Cyclic offered cohorts explicitly record repeated source use.

These data are neither original Azure prompts nor a natural output-length distribution.

All main generation is controlled with `ignore_eos=True`.
Internal exact token parity does not establish original-HF fidelity or downstream answer accuracy.
No Dynamic A, model-depth reduction, approximation, quantization, or training runs in this study.

Finite-cohort throughput is not infinite-horizon sustainable capacity.
