# Timing contract

## Physical scope

Only physical GPUs 4, 5, 6, and 7 can execute experiment kernels.
TP1 uses GPU4. TP2 uses GPUs4–5. TP4 uses GPUs4–7.
Logical CUDA ranks follow that order. All models use dense TP with EP1, DP1, PP1, and CP1.
There is no MoE Dispatch or Combine All-to-All in this experiment.

The guard checks GPU UUIDs and process ownership. It stops only verified project burners on the selected devices.
Unused assigned devices retain burners. The guard restores burners after success, failure, and timeout.
Other users' processes remain untouched.

## Matched runtime

All primary configurations use the same installed vLLM and BF16 policy.
The V2 model runner uses FlashAttention 3, native compilation, and CUDA Graph capture sizes 1, 2, 4, and 8.
The runtime disables prefix caching, chunked prefill, and asynchronous scheduling.
The maximum model length is 32,768. The maximum batch size is eight, but the measured hardware lane is B1.
The KV pool uses the same 0.85 HBM utilization policy. Its actual capacity differs with model and R.
Each configuration shuts down before the next configuration loads.

The eager smoke has a different cache policy. It establishes feasibility only and never enters the headline timing table.
The retained-graph diagnostic also has a separate label. Its graph introspection cannot change the primary timing policy.

## Ledger W: clean client wall

`E2E_ms` measures one public `LLM.generate` call after model load, graph capture, and warmup.
It includes prefill, 64 generated tokens, scheduler work, output transfer, and finalization.
Model load, compilation, ownership inspection, and RPC configuration occur outside this interval.

`prefill_ms` means scheduled-to-first-token wall time from native request metrics.
It includes the initial model forward, initial sampling, and engine handling. It is not pure attention or GPU prefill time.
`TPOT_ms` equals `(last_token_ts - first_token_ts) / 63` in milliseconds.
The denominator is 63 cached-decode intervals, not 64.
`output_TPS` equals 64 divided by client E2E seconds.

The teacher hook preserves native logits and native argmax computation.
It then copies one frozen token ID into the sampled buffer before the runtime updates request state.
This adds one small GPU copy per token to every primary arm. No per-token GPU-to-CPU read occurs in that hook.
The separate diagnostic compares the same natural trajectory with and without forced copies.
This hook is not a proposed inference optimization.

## Ledger E: outer GPU intervals

The observer records every actual `execute_model` and `sample_tokens` interval for one 64-token request.
Events resolve only after the request. There is no per-layer host synchronization.
The runner also makes an empty trailing `execute_model` call. The analysis excludes it from actual mathematical forwards.
Raw `model_forwards` counters count API calls. They do not prove that every call executes model layers.

An interval can contain host gaps and rank waits. The clocks are local to each GPU.
Cross-rank clock alignment is not established. The report does not invent cross-rank waiting milliseconds.

## Ledger K: kernel service

A separate profiler captures one 16K prefill and three cached forwards.
The source marker `_compute_slot_mappings_kernel` separates those four actual forwards.
Kernel names classify attention, KV writes, GEMMs, reductions, memory operations, and remaining work.
The trace includes custom TP reductions, NCCL reductions, and output gathering where the backend exposes them.

Attention includes fused KV reads. Generic GEMM names do not always distinguish QKV, FFN, output projection, and LM head.
The combined GEMM category states this limit. It does not invent separate direct timings.
Kernel service can overlap other work. Its sum is not an additive E2E percentage.
An interval union is GPU busy time, not a cross-rank critical-path decomposition.

## Ledger C: critical-path limits

Clean client wall ratios support the TP scaling comparison.
The available traces only partially identify internal critical paths.
Pure communication time, transport bandwidth, and cross-rank synchronization delay remain unresolved without additional clock validation.
`TPExcessVsIdeal` is a diagnostic residual. It is not removable latency.
No hypothetical removal of all collectives supports a new framework claim.

## Observer validity and uncertainty

Clean timing never uses profiler wall time.
The analysis reports signed observer differences against three clean controls.
One-profile differences include noise and drift. They do not establish an independent observer confidence interval.
Observer inflation above 10% invalidates fine timing attribution for that trace.
Kernel counts remain diagnostic even when timing attribution fails.

All primary ratios form within each request before aggregation.
The bootstrap uses 5,000 draws with a fixed analysis seed. Intervals are descriptive for this bounded cohort.
Configuration order is partly counterbalanced. Four targeted independent sessions repeat two short-context contrasts in reverse family order.
The experiment does not repeat the full grid across randomized independent sessions.
Other users can use GPUs0–3 on the same host. Their CPU or fabric activity remains an uncontrolled source of drift.
The experiment does not change GPU clocks or power limits.
