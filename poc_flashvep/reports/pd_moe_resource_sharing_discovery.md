# Direction C. MoE prefill–decode resource sharing

## Model, physical ownership and provenance

The pinned BF16 Qwen3-30B-A3B revision is `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. Its config specifies 48 layers, 128 experts with top-eight routing, hidden width 2048, SwiGLU expert width 768, 32 Q heads and four KV heads of width 128. The checkpoint is available as 16 safetensors shards. [Official checkpoint config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)

The shard used in the layer-0 replay was verified against the publisher's ETag SHA-256 before use. The pinned model's `tokenizer.json` and `tokenizer_config.json` are byte-identical to the pinned Qwen3-8B versions, so the native EP smoke can use the same frozen chat-token trace. The weight receipt records shard sizes and publisher ETags. [Weight receipt](../pd_resource_complementary_parallelism/results/moe/qwen3_30b_weight_receipt.json)

For physical EP2, rank 0 owns experts 0–63 and rank 1 owns experts 64–127. Each owner loads its BF16 gate/up and down weights from the official layer-0 checkpoint. Two ranks each send one copy of a token to every **distinct** owner of its selected experts. The receiver applies the vLLM unquantized fused expert kernel only to locally owned expert IDs, then returns the partial contribution to the submitting rank. The token's weighted output is recombined there. The replay has NCCL all-to-all and legacy DeepEP intranode HT paths for the activation and routing payload. It separately accounts for the exact tensor payload excluding local rows and protocol overhead. [DeepEP revision](https://github.com/deepseek-ai/DeepEP/tree/73b6ea4a439ba03a695563f9fd242c8e4b02b37c)

The rank input is computed from actual LongBench text with the pinned tokenizer, real embeddings and layer-0 Q/K/V/out weights, Q/K RMSNorm and RoPE, optimized FA3 attention, and the layer's post-attention RMSNorm. Decode rows use the final teacher-forced token after a real document prefix. The two ranks use disjoint document identities. The isolated decode unit is one genuine batched attention/routing step. This controls operator shape independently of serving output length.

The reference loads an unsharded copy temporarily, compares the reconstructed EP outputs against the same BF16 grouped expert implementation, and releases the reference copy before timing. A relative L2 above 2% stops timing. The replay holds routing and source indices fixed during timing, includes real collectives and all dependent expert work, and excludes scheduler cost, live layout construction and transfer between separate P/D replica groups. A separate native harness uses vLLM DP2/EP2 within each P/D pool over TP1 attention, matching the official runtime ownership semantics. [vLLM expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

## Why physical overlap is screened before integration

The concurrent trial runs decode dispatch, combine, expert compute, or attention against a separate prefill action on each of the two devices. CUDA events bracket each dependent operation and record critical-rank wall time on device. The same buffers, weights and plan also run isolated, serially, on equal-priority streams, and on streams prioritizing decode. NCCL's communication kernels can consume SMs, and both phases share HBM and L2 on each GPU. Priority streams cannot provide a hard SM allocation or interrupt previously launched kernels. The prefill route remains real even when all eight experts happen to have the same physical owner.

An initial qualifying point requires at most 5% extra critical-rank decode p50 and p95 relative to isolated execution, useful completed prefill expert work, and 10% pair makespan savings relative to the serial implementation. Even a qualifying pair leaves a further gate for any cross-pool activation handoff, the dependency chain, and a strong existing online overlap baseline.

## Prior-work boundary

MegaScale-Infer splits attention and expert execution into separately scaled pools and interleaves microbatches over its many-to-many transport. ExpertPlex directly shares experts between prefill and decode and already addresses phase interference with tile-granular persistent expert kernels, bounded cooperative preemption, attention-initiated one-sided communication, and cross-stack placement. SGLang integrates DeepEP and overlaps dispatch/combine or expert compute across microbatches. The isolated pair can establish headroom at this model and topology, while the phase-sharing placement and concurrent primitives already have these mechanisms as prior art. [MegaScale-Infer](https://arxiv.org/abs/2504.02263) [ExpertPlex, sections 3–6](https://arxiv.org/html/2607.18002v1) [SGLang expert parallelism source](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)

Earlier project measurements tested Qwen3-VL Expert Parallel and Vision Encoder overlap. Their negative stage outcomes were bounded to co-running the encoder with that DeepEP path. The current measurement uses Dense LM attention and expert tensors derived from separate prefill and decode phases, and directly observes the shape-specific two-rank critical path. [Project negative-space record](https://github.com/jyssys/MLLM-EP/blob/d13ee5a8173ec09f458aa5c118e8903fdafff801/poc_flashvep/autonomous_moe_ep_night_discovery/KNOWN_NEGATIVE_SPACE.md)

## Physical EP2 on H200

`MEASURED_PHYSICAL` and `CORRECTNESS_ONLY`. Two H200s on NVSwitch ran each transport with D4 and D8 rows per rank at an 8,192-token cached context and one teacher-forced position. Prefill chunks used 128, 512 or 2,048 rows per rank. Each cell includes 30 timed samples after eight warmups. The ranks synchronize before each sample. A preceding CUDA event allows both stream queues to be submitted, and the scored interval begins after that event. The NVSwitch lease ran these cases without another research GPU process on the pair.

The largest error against the unsharded BF16 reference was **0.272% relative L2** over the observed prefill and decode probes. The largest owner imbalance was 1.25× mean at D4, with expert assignment totals of 24 and 40. D8 selected 68 unique experts with owner counts 56 and 72. The P512 case at D8 selected 117 experts, with owner counts 3,774 and 4,418. Each physical rank held 576 MiB of layer-0 expert weights for 64 experts. The other rank's weight shard was present only for the pre-run reference check. Across both ranks, activation dispatch sent 32 KiB at D4 and 64 KiB at D8. Combine sent the same respective volumes back. P512 sent 3.99–4.00 MiB of activation payload per operation across the two ranks. The raw data records routing metadata separately. These counts derive from the actual route. Hardware wire counters and protocol overhead remain unmeasured.

| Backend / decode rows per rank | D attention | D dispatch | D expert GEMM | D combine | D full EP path | P512 full EP path |
|---|---:|---:|---:|---:|---:|---:|
| NCCL all-to-all / 4 | 39.9 | 49.8 | 95.3 | 24.0 | 146.9 | 354.0 |
| NCCL all-to-all / 8 | 53.7 | 61.2 | 112.5 | 25.8 | 165.9 | 362.6 |
| DeepEP HT / 4 | 40.1 | 101.2 | 95.1 | 26.3 | 411.6 | 569.1 |
| DeepEP HT / 8 | 53.2 | 96.2 | 112.0 | 29.0 | 398.0 | 560.7 |

All entries are critical-rank CUDA event p50 in µs. Each component runs separately, so component medians and dependent-path medians have different scheduling and should not be added. The NCCL fixed plan pre-packs activation rows, while DeepEP runs its native intra-node dispatch and size discovery. DeepEP HT launches with a configured 20-SM communication budget and uses its default dynamic receive-count path. This path's host enqueue and GPU progress are included. The pair benchmark does not cover DeepEP LL or an optimized capture of a live dynamic route. The installed Triton kernel used its default BF16 configuration for 64 local experts because this vLLM build lacks a tuned H200 EP64 entry.

| Backend / decode rows per rank | Joint pair pass | Qualifying configuration |
|---|---:|---|
| NCCL all-to-all / 4 | 4/24 | D expert GEMM with P128 or P512 dispatch on equal-priority or decode-priority streams |
| NCCL all-to-all / 8 | 3/24 | D expert GEMM with P128 dispatch on both stream modes, or P512 dispatch on equal-priority streams |
| DeepEP HT / 4 | 0/24 | Closest P128 dispatch with D expert GEMM, 31.77% paired savings, D p50 +5.42%, D p95 +4.41% |
| DeepEP HT / 8 | 0/24 | Closest P128 dispatch with D expert GEMM, 39.28% paired savings, D p50 +4.69%, D p95 +5.35% |

The seven passing replay cells saved **22.34–30.53%** paired p50 makespan against their serial control with **3.19–4.36%** D p95 slowdown. Their additional P work was dispatch only. At P2048, even D expert GEMM with P dispatch increased D p95 by **10.23–22.98%** across both transports and stream modes. Every pairing of P expert GEMM with D attention failed the decode guard, and D attention p95 increased **32.51–544.22%**. D dispatch or combine paired with P expert compute also failed. These trials establish both a bounded overlap case for small independent communication and a clear interference cost for sending prefill expert compute onto the active decode ranks at the measured shapes.

The NCCL and DeepEP cases are separate replay runs on the same inputs and ownership. Layer-0 input documents differ across the D4 and D8 cases to keep document identities disjoint between the ranks. The 160-second lease ended during B1 startup, so B1 produced no complete sample and was excluded. The initial launcher attempt stopped before model execution because two system ELF utilities were absent. The recovered run used the isolated environment's prefixed `readelf`, and records the loaded binary SHA-256 and buffer configuration. [Backend ELF receipt](../pd_resource_complementary_parallelism/results/moe/deepep_elf.txt)

- [NCCL D8 communication and compute](../pd_resource_complementary_parallelism/results/moe/nccl_a2a_batch8/moe_ep_communication_compute.png)
- [NCCL D8 pairwise decode slowdown and serial-relative savings](../pd_resource_complementary_parallelism/results/moe/nccl_a2a_batch8/moe_ep_operator_pairing.png)
- [DeepEP HT D8 communication and compute](../pd_resource_complementary_parallelism/results/moe/deepep_ht_batch8/moe_ep_communication_compute.png)
- [DeepEP HT D8 pairwise decode slowdown and serial-relative savings](../pd_resource_complementary_parallelism/results/moe/deepep_ht_batch8/moe_ep_operator_pairing.png)

## Native four-GPU PD baseline

`MEASURED_ONLINE`, N1 baseline pilot. GPUs 4–5 formed P-EP2 and GPUs 6–7 formed D-EP2. Native vLLM used BF16, TP1 attention, independent DP2 scheduling inside each pool, native NCCL AG/RS all-to-all, sparse Triton experts, FlashAttention 3, chunked prefill, CUDA graphs, and physical NIXL KV handoff. Startup logs confirm `AgRsAll2AllManager` on both pools and linear ownership of 64 of 128 experts per rank. The model took 29.88 GiB per rank and the 24 GiB reserve exposed 262,144 KV token slots per rank. The first launch reached both model endpoints and then hit a client Namespace mismatch after a shared helper interface change. The corrected restart completed both cohorts.

| Input/output | Offered QPS | Completed | TTFT p95 | TPOT p95 | ITL p99 | E2E p95 | Joint SLO | Finite-cohort goodput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192/16 | 4 | 24/24 | 1.239 s | 11.71 ms | 85.26 ms | 1.333 s | 17/24 | 2.438 req/s |
| 16,384/16 | 2 | 24/24 | 1.174 s | 11.64 ms | 16.37 ms | 1.242 s | 21/24 | 1.516 req/s |

The frozen Poisson draws realized 3.647 and 1.823 requests/s over their arrival spans. Joint SLO used TTFT ≤5 s, TPOT ≤33 ms and every ITL ≤34 ms. Maximum ITL was 93.67 ms at 8K and 628.28 ms at 16K, so the 16K pooled ITL p99 did not capture the worst request-level stalls. Native NIXL counters recorded 24 transfers per cohort and 18 or 36 GiB of transfer payload, exactly consistent with 96 KiB of model KV per prompt token. Its transaction timer averaged 147.83 and 184.51 ms respectively, including the recorded native connector path. These values do not estimate isolated NVLink transfer speed. In the central sampled windows, P averaged 27.8–34.5% SM activity across its rank devices and D averaged 6.7–11.1%. This pilot establishes native serving and real KV movement for the two cohorts. P saturation, native DeepEP configuration, EP4, matched dense/MoE load comparability and any online sharing benefit remain `NOT_RUN`. [Aggregate startup, NIXL, GPU and request receipts](../pd_resource_complementary_parallelism/results/moe/native_ep_agrs_pilot.json)

## Reproduction

The replay needs PyTorch 2.11.0+cu129, vLLM 0.26.0+cu129, compatible NCCL, two NVLink-connected H200s, the checkpoint above, and the LongBench document workload from the project's previous LoopCache preparation. The raw ranks identify the source document IDs and source SHA-256. Select the granted two device IDs or UUIDs and run from the repository root.

```bash
runtime=/path/to/mllm_epd_runtime_env
export PATH="$runtime/bin:$PATH"
export CC="$runtime/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$runtime/bin/x86_64-conda-linux-gnu-g++"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m poc_flashvep.pd_resource_complementary_parallelism.moe_ep_pairs \
  --backend nccl_a2a --model /path/to/qwen3-30b-a3b \
  --workload /path/to/ouro_r4_w2_document_source_order.json \
  --out /path/to/moe_ep_run --samples 30 --prefill-rows 128 512 2048 \
  --decode-context 8192 --decode-batch 8
python -m poc_flashvep.pd_resource_complementary_parallelism.moe_ep_summary \
  --raw /path/to/moe_ep_run --out /path/to/moe_ep_summary --figures
```

Use `--decode-batch 4` for the smaller serving geometry. Use `--backend deepep_ht` and an isolated legacy DeepEP overlay for the second backend. Build DeepEP 1.2.1 from commit `73b6ea4a439ba03a695563f9fd242c8e4b02b37c` in an external directory containing its extracted source and extracted Debian RDMA development headers. This invocation used CUDA 12.9 and NVSHMEM 3.4.5 from the existing runtime. The build-only `TORCH_DONT_CHECK_COMPILER_ABI=1` bypassed PyTorch's GCC version guard for GCC 14.4. NVIDIA's compiler compatibility check remained enabled. [DeepEP source](https://github.com/deepseek-ai/DeepEP/tree/73b6ea4a439ba03a695563f9fd242c8e4b02b37c) [RDMA development archive](https://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibverbs-dev_39.0-1_amd64.deb) [CUDA 12.9 GCC support](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html)

```bash
study=/path/to/isolated/deepep_build
revision=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
dpkg-deb -x "$study/libibverbs-dev_39.0-1_amd64.deb" "$study/rdma_headers"
nv_headers=$(find "$runtime/lib/python3.14/site-packages/nvidia" \
  -mindepth 2 -maxdepth 2 -type d -name include | sort | paste -sd:)
cd "$study/DeepEP-$revision"
env -u NVSHMEM_DIR CUDA_VISIBLE_DEVICES='' CUDA_HOME="$runtime" PATH="$runtime/bin:$PATH" \
  CPATH="$runtime/targets/x86_64-linux/include:$study/rdma_headers/usr/include:$nv_headers" \
  CC="$runtime/bin/x86_64-conda-linux-gnu-gcc" CXX="$runtime/bin/x86_64-conda-linux-gnu-g++" \
  TORCH_DONT_CHECK_COMPILER_ABI=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2 \
  DISABLE_AGGRESSIVE_PTX_INSTRS=1 OMP_NUM_THREADS=1 \
  nice -n 15 "$runtime/bin/python" setup.py build_ext --build-lib "$study/overlay"
cp -a deep_ep "$study/overlay/"
export PYTHONPATH="$study/overlay${PYTHONPATH:+:$PYTHONPATH}"
```

Run the benchmark from the repository root after installing the overlay. The raw compressed rank files and JSON reductions can regenerate the figures. Use the base runtime for the native AG/RS pilot. Set the following four variables to the selected GPU UUIDs to enable the shared client's NVML observer.

```bash
export CUDA_VISIBLE_DEVICES="$P0_UUID,$P1_UUID,$D0_UUID,$D1_UUID"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m poc_flashvep.pd_resource_complementary_parallelism.moe_ep_native \
  --model /path/to/qwen3-30b-a3b --out /path/to/native_ep2_pilot \
  --case ep2_pd_8k:/path/to/8192x16-n24.json:4 \
  --case ep2_pd_16k:/path/to/16384x16-n24.json:2 \
  --max-model-len 16512 --max-seqs 64 --all2all-backend allgather_reducescatter
```

## Candidate decision

**NO_GO_INCREMENTAL for the tested Direction C placement.** Physical EP2 and Phase C operator correctness passed. The decode-protected overlap moved small prefill dispatches in the fixed-route NCCL path. Every trial that moved prefill expert compute onto the active decode ranks exceeded the decode guard. DeepEP HT failed the joint local guard in all 48 cases. The surviving NCCL pattern matches generic dual microbatch overlap. Source comparison also places independent cross-phase expert pooling and fine-grained decode protection within ExpertPlex's scope.

The H200 replay does not establish whether MoE has stronger resource complementarity than matched dense inference across serving workloads. The two model families have different activations, expert dispatch topology, and attention geometry. Cross-pool activation transfer, expert stragglers, combined scheduling, and candidate online performance were `NOT_RUN`. The native AG/RS pilot measures the isolated P/D role baseline.
