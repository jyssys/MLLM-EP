# Memory accounting and missing counters

## Native weights and HBM

baseline_summary.csv records exact learned parameter counts and bytes.
All learned parameters remain GPU-resident BF16.
Foil also retains its native FP32 RoPE buffers.
timing_summary.csv records peak PyTorch allocated bytes in each clean timing cell.
These values are not device-wide reserved HBM.
The raw runtime metadata also records allocated and reserved memory after loading.

The native generation paths do not create a KV cache.
No cached-decode KV-capacity claim is applicable.

## Expert-output reuse cache

The reuse cache is separate from attention KV.
At each recurrent site, it stores previous-pass expert IDs, raw outputs, and normalized input.
It retains one tensor state per site inside one complete forward.
Every new forward clears this state.

The original teacher-forced runs did not record the peak cache counter.
Their summary cells remain NOT_RECORDED.
They must not appear as measured zero-byte caches.

Natural-generation runs did record controller cache peaks.
generation_quality_summary.csv reports those actual maxima by model, task, and arm.
The raw generation rows retain per-example counters.
These counters exclude temporary gather tensors, allocator fragmentation, and reference-oracle snapshots.
They therefore do not measure total implementation memory overhead.

## Structural estimate for a complete one-pass cache

For B1, prefix length L, width D, top-k K, and S recurrent sites, the retained tensor size is:

`S × L × (2 × K × D + 2 × D + 8 × K)` bytes.

The formula assumes BF16 outputs and inputs with INT64 expert IDs.
It describes tensor contents, not a measured allocation peak.

| Model | L | D | K | S | Structural tensor bytes | Structural MiB |
| --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | 512 | 1536 | 2 | 28 | 132,349,952 | 126.21875 |
| IFM MoE Huginn | 512 | 1536 | 2 | 12 | 56,721,408 | 54.09375 |
| Foil-1 | 512 | 1024 | 2 | 1 | 3,153,920 | 3.0078125 |

These are STRUCTURAL_ESTIMATE values.
They are not measured speedups or HBM traffic.
Actual cache reads, writes, and allocator costs need separate profiling before any production claim.
