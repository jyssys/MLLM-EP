# Direction D — Heterogeneous prefill parallelism

## Screening contract

This experiment compares legal placements in a two-H200 prefill team. All variants execute the same original BF16 Qwen3-8B layers using cuBLAS linear projections, vLLM fused RMSNorm and SwiGLU, and FlashAttention 3. The token inputs come from the frozen Qwen3 non-thinking workload. The first layer consumes the original token embeddings. Multi-layer cases use the real output of the previous Qwen layer. The same activation and position sequence is used for every variant at a given shape.

| Physical arm | Attention | MLP | Conversion on the critical path |
|---|---|---|---|
| Uniform TP2 | Half of Q/KV heads, all rows | Half of intermediate channels, all rows | All-reduce after attention and after MLP |
| Contiguous CP2 | Half of token rows and full heads, earlier rank sends its K/V to the later rank | Full MLP weight on the local token rows | Prefix KV send, concatenate, final output gather |
| Zig-zag CP2 | Balanced early/late token groups and full heads | Full MLP weight on the local token rows | KV all-gather, reorder, final output gather |
| TP attention and row MLP | Same head slicing as TP2 | Full MLP weight on the local token rows | Reduce-scatter after attention, all-gather after MLP |
| CP attention and TP MLP | Same balanced token layout as zig-zag CP2 | Half of intermediate channels, all rows | KV all-gather and reorder, hidden all-gather, MLP reduce-scatter, final output gather |
| TP1 / two independent requests | Full heads | Full MLP weight | Each GPU completes its own request. At batch one, results represent single-GPU latency. |

A contiguous CP rank attends to every preceding token in its request. The early rank keeps its local prefix and the late rank receives and concatenates it. Zig-zag places the first and fourth quarters on one rank and the two middle quarters on the other. Pairs of equal-length sequences stay logically separate. Each recipient gets a valid full causal history for its local queries. GEMM geometry and causal load balance can change even with the original math.

The numerical screen uses a 0.02 maximum relative L2 for the full hidden result and reconstructed final-layer K/V against the unsharded BF16 reference, on both ranks. This is a layer-level screen. Model logits, greedy generation and online quality require separate qualification. The advance target is at least 10% lower measured critical-path time than the best eligible matched two-GPU layout. Whole-server vLLM is a separate strong control for inference.

## Analytical accounting

Let (S) be total token rows, (H=4096), and (K=8\times128=1024). A BF16 hidden buffer is (2SH) bytes and full per-layer K/V is (4SK) bytes. Unit counts below are per rank over both directions. The expressions are logical NCCL ring or peer payloads, before protocol and memory traffic.

| Arm | Steady layer transmit payload per rank | Layer weight residency per rank |
|---|---:|---:|
| TP2 | (4SH) | 184 MiB |
| Contiguous CP2 | (2SK) on rank 0, zero on rank 1 | 368 MiB |
| Zig-zag CP2 | (2SK) | 368 MiB |
| TP attention and row MLP | (2SH) | 328 MiB |
| CP attention and TP MLP | (2SK+2SH) | 224 MiB |

The four head-wise TP2 collectives over two sequential layers are replaced by two reduce-scatter and two all-gather operations in the first hybrid. The payload halves because the full MLP weights run on sharded rows and there is no MLP output reduction. This requires 144 MiB extra MLP weight per layer per rank, or 5.0625 GiB for 36 layers. Norm vectors contribute a few KiB beyond the displayed weight totals. CP variants also have a one-time final output all-gather in the benchmark. The contiguous plan has roughly 1.5 times as many causal key-query pairs on the slow rank as balanced Q/head sharding. Zig-zag trades extra messages and reordering for even causal work. Full P/D layouts would still have to deliver 36 layers of KV into the chosen D ownership.

## Native controls and related work

The local vLLM 0.26 runtime can select prefill TP independently using `native --p-tp 2` for 2 physical P GPUs and one D GPU. For a matched replica baseline, select `--prefill-workers 2 --p-tp 1` with the same three physical GPUs, batching, request order and NIXL path. The native connector has separate tensor-rank head mappings for P and D. Installed Dense FlashAttention has no `supports_pcp` override and V2 lists non-MLA PCP as unsupported. Testing a standalone FA3 CP control therefore checks a physical possibility while an online vLLM PCP track depends on runtime support. [vLLM context-parallel deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md) · [NIXL compatibility](https://docs.vllm.ai/en/v0.23.0/features/nixl_connector_compatibility/)

The narrow first hybrid keeps full MLP weights on each P rank and moves the sequence layout through MLP. Megatron-style sequence parallelism already uses reduce-scatter and all-gather at tensor-parallel boundaries, commonly leaving MLP weights split and moving elementwise operators to sequence shards. Generic physical axis selection already captures this hybrid's communication tradeoff. A new PD contribution would need an independently favorable prefill/handoff or online scheduling mechanism at equal cost. [Korthikanti et al., MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf) · [Prefill Context Parallel for Qwen3.5 Hybrid Attention](https://github.com/vllm-project/vllm/issues/37995)

## Two-H200 measurements

The test used GPUs 4 and 5 on `henn-0` with their NV18 link. The checkpoint was Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`. All results below are medians of seven timed, warmed repetitions, taking the slower CUDA duration of the two ranks per repetition. The input is a batch of one. Each context length takes a prefix of the same 32K frozen non-thinking prompt token sequence. The frozen trace records public GOV Report provenance and corpus extensions for the controlled length.

| Layers | Input | TP2 ms | Contiguous CP2 ms | Zig-zag CP2 ms | TP attention, row MLP ms | CP attention, TP MLP ms | TP1 ms | Best hybrid reduction vs best uniform two-GPU arm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,048 | 0.975 | 0.963 | 1.016 | 0.902 | 1.132 | 1.465 | +6.35% |
| 1 | 8,192 | 3.684 | 3.690 | 3.580 | 3.522 | 3.942 | 6.512 | +1.63% |
| 1 | 16,384 | 8.363 | 9.335 | 8.061 | 7.890 | 8.558 | 14.935 | +2.11% |
| 1 | 32,768 | 20.898 | 22.446 | 19.705 | 20.164 | 20.796 | 36.949 | −2.33% |
| 4 | 2,048 | 3.724 | 3.433 | 3.581 | 3.583 | 4.075 | 5.942 | −4.37% |
| 4 | 8,192 | 15.221 | 14.981 | 14.756 | 14.472 | 15.515 | 27.108 | +1.92% |
| 4 | 16,384 | 35.003 | 35.371 | 32.142 | 34.108 | 34.613 | 60.362 | −6.12% |
| 4 | 32,768 | 82.738 | 91.698 | 79.123 | 78.995 | 82.391 | 147.938 | +0.16% |

All 48 cases passed the frozen relative L2 screen across both ranks. Worst relative L2 was 0.00598 for hidden, 0.00474 for K and 0.00679 for V. Both CP-only layouts produced bitwise-equal hidden states to the single-GPU reference in these cases. The TP-containing arms use a BF16 reduction order different from the unsharded projection. The checks cover the measured hidden result and reconstructed final-layer K/V.

Stage timing instrumented each arm independently. In 37 of 48 cases the event-instrumented median was within 5% of the corresponding clean median. The remaining stage traces are diagnostic. Every 32K four-layer stage trace passed the observer screen. At that shape the slower-rank stage medians show the following costs.

| Arm | Attention kernel ms | MLP ms | NCCL communication ms | Layout transforms ms | Peak rank TX payload MiB |
|---|---:|---:|---:|---:|---:|
| TP2 | 26.849 | 29.022 | 7.588 | 0.000 | 2,048 |
| Contiguous CP2 | 41.681 | 30.097 | 1.199 | 1.538 | 384 |
| Zig-zag CP2 | 27.456 | 29.426 | 2.861 | 1.671 | 384 |
| TP attention, row MLP | 28.143 | 31.130 | 4.839 | 0.023 | 1,024 |
| CP attention, TP MLP | 26.570 | 28.833 | 9.792 | 1.430 | 1,408 |

Stage columns are sums of medians on the slower instrumented rank. TX payload derives from the exact shape and number of executed NCCL operations. The physical transfer and its waiting time are included in the measured stage duration. Hardware NVLink counters were outside this Direction D profile. Layout includes P-team gathering, permuting, concatenating and restoring the output order. At 32K the hybrid saves 2.75 ms in NCCL time against TP2 but its MLP stage takes 2.11 ms longer. The balanced generic CP baseline reaches nearly the same critical-path time as the fastest hybrid with substantially smaller transferred payload. At 16K the generic CP baseline is the fastest.

### CUDA Graph confirmation

The 16K four-layer batch-one replay compared TP2, zig-zag CP2 and TP attention with an MLP split by token rows. The three arms reached 32.737, 30.747 and 31.520 ms respectively, all within the hidden and KV relative L2 screen. The hybrid took 2.51% longer than zig-zag CP2. Each arm used seven timed graph replays. The balanced generic CP control also led the eager comparison at this shape. [Graph raw data](../pd_resource_complementary_parallelism/results/d_hybrid_four_layers_graph.jsonl.gz) · [Graph latency figure](../pd_resource_complementary_parallelism/figures/d_latency_l4_b1_cuda_graph.png)

## Three-GPU native allocation controls

Three vLLM layouts ran separately on GPUs 4, 5 and 6. The controls used the same BF16 model, NIXL HND cache layout, CUDA Graph configuration, 48 GiB KV cache per GPU, 8,192 batch-token budget, chunked prefill with APC disabled, and 64 fixed 16-token responses on each frozen Poisson schedule. TP2 uses one sharded P worker. The two-P control uses separate replicas. Each completed arm has one server start and four warmup requests.

| Prompt and offered req/s | Layout | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | Joint SLO | Finite goodput req/s |
|---|---|---:|---:|---:|---:|---:|
| 8K at 4.5 | P TP2 / D1 | 1,122.32 | 40.42 | 134.19 | 43/64 | 3.008 |
| 8K at 4.5 | P2 / D1 | 686.58 | 8.87 | 62.64 | 53/64 | 3.723 |
| 8K at 4.5 | P1 / D2 | 1,943.37 | 4.85 | 8.93 | 64/64 | 4.066 |
| 16K at 1.9 | P TP2 / D1 | 1,529.74 | 40.39 | 265.18 | 31/64 | 0.921 |
| 16K at 1.9 | P2 / D1 | 1,251.77 | 11.29 | 20.34 | 59/64 | 1.761 |
| 16K at 1.9 | P1 / D2 | 3,882.21 | 4.17 | 8.44 | 64/64 | 1.773 |
| 16K at 2.6 | P TP2 / D1 | 1,891.72 | 51.78 | 262.28 | 21/64 | 0.844 |
| 16K at 2.6 | P2 / D1 | 1,379.42 | 11.59 | 23.29 | 58/64 | 2.344 |
| 16K at 2.6 | P1 / D2 | 9,875.84 | 4.38 | 6.78 | 26/64 | 0.758 |

All layouts completed every request. All D prompt rows came from NIXL. The measured total matched the full BF16 cache size of 147,456 bytes per token, or 72 GiB for the 8K cohort and 144 GiB for each 16K cohort. The sharded producer required 128 transfer operations per cohort, versus 64 in the two other native layouts. D recorded zero failed transfers, failed notifications and expired KV leases. D-submit-to-first-token p95 was 631–739 ms under P TP2 / D1 and 193–350 ms under P2 / D1. Those spans include waiting and compute. The P TP2 / D1 pilot first failed to compile FlashInfer's MNNVL extension because the conda linker could not locate `libcuda`. A study-local `LIBRARY_PATH` entry exposing the installed driver repaired startup. The successful rerun used the original native fused collectives.

P2 / D1 gave the highest tested goodput at 16K and 2.6 requests/s. P1 / D2 had the highest observed goodput in the other two cohorts. The P1 / D2 and P2 / D1 measurements at 16K and 1.9 requests/s differ by less than 1% on finite goodput. P1 / D2, P2 / D1 and P TP2 / D1 matched the two-GPU native 1P / 1D full greedy output sequences on 192/192, 182/192 and 179/192 requests respectively. The native configuration changes therefore show some output drift. Task accuracy was `NOT_RUN`. These results are one-start pilots. Temperature was sampled at arm boundaries, with different start values across arms, and continuous clock parity was unverified. [Layout health, raw counters and paired output counts](../pd_resource_complementary_parallelism/results/d_native_static_layout_health.json)

## Decision

`NO_GO_INCREMENTAL` on the tested Dense Qwen3-8B geometries. The largest measured incremental reduction is 6.35% in a single 2K layer. The four-layer best is 1.92% at 8K. Both are below the 10% physical advance target. Generic causal balancing explains the benefit of shorter attention spans, and generic row/weight placement explains the reduction in collective payload. The first hybrid spends an additional 5.0625 GiB of projected MLP weight per rank over a 36-layer TP2 stack.

Online hybrid PD, D-side cache admission, P-to-D layout conversion and hybrid tail-latency validation are `NOT_RUN`. The screen executed fused kernels and NCCL in eager mode and confirmed the main comparison with a 16K CUDA Graph replay. Its conclusions apply to these implementations, a batch of one and the listed context lengths. Full-stack TP4, PP and native PCP configurations were outside this physical test. The native allocation pilots establish whole-server controls for future hybrid comparisons.

## Reproduction

The raw per-rank receipts are [one layer](../pd_resource_complementary_parallelism/results/d_hybrid_one_layer.jsonl.gz) and [four layers](../pd_resource_complementary_parallelism/results/d_hybrid_four_layers.jsonl.gz). The four-layer [latency figure](../pd_resource_complementary_parallelism/figures/d_latency_l4_b1_eager.png) and [layout conversion figure](../pd_resource_complementary_parallelism/figures/d_layout_l4_b1_eager.png) are generated by `hybrid_figures.py`.

```bash
module=poc_flashvep.pd_resource_complementary_parallelism.hybrid_prefill
python -m "$module" --model "$QWEN3_8B" --workload "$FROZEN_32K16" \
  --out "$RUN/one_layer.jsonl" --layers 1 --repeats 7 --lengths 2048 8192 16384 32768
python -m "$module" --model "$QWEN3_8B" --workload "$FROZEN_32K16" \
  --out "$RUN/four_layers.jsonl" --layers 4 --repeats 7 --lengths 2048 8192 16384 32768
python -m "$module" --model "$QWEN3_8B" --workload "$FROZEN_32K16" \
  --out "$RUN/hybrid_graph.jsonl" --layers 4 --repeats 7 --lengths 16384 --graph \
  --arms tp2 cp2_zigzag tp_attention_row_mlp
python -m poc_flashvep.pd_resource_complementary_parallelism.hybrid_figures \
  --out "$RUN/figures" --raw "$RUN/one_layer.jsonl" "$RUN/four_layers.jsonl" "$RUN/hybrid_graph.jsonl"
```

Run the measurements in the existing two-GPU experiment lease. The figure command uses an environment with matplotlib. The physical run used PyTorch 2.11.0+cu129 and vLLM 0.26.0+cu129.

Run each native static layout within the same three-GPU lease with a new output directory. Use `p_tp2_d1`, `p2_d1` and `p1_d2` as `ARM`.

```bash
PD_RUN_ROOT="$RUN_ROOT" PD_MODEL="$QWEN3_8B" PD_PYTHON="$RUNTIME/bin/python" \
  bash poc_flashvep/pd_resource_complementary_parallelism/run_dense.sh "$ARM"
```

For the recovered TP2 start, `LIBRARY_PATH` also pointed to a writable study directory containing `libcuda.so` symlinked to the installed `/lib/x86_64-linux-gnu/libcuda.so.1`. The library was visible at build time. The native analysis reducer reads each cohort's `before_*.prom` and `after_*.prom`, request timing JSONL and engine command receipt.
