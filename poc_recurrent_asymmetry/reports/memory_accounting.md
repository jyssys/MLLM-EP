# Memory accounting

## Distinct quantities

The loaded model has 2.672249 GiB of unique parameter storage.
Four passes share that storage.
This does not prove four complete HBM weight reloads per forward.
Actual HBM traffic and L2 reuse counters are unavailable.

The audited logical KV formula is:

```text
B × history × 4 depths × 24 layers × 2(K,V) × 16 heads × 128 × 2 BF16 bytes
```

At R4, each request needs 786,432 logical KV bytes per historical position.
The official HF cache has 96 independent key storages and 96 independent value storages.
The allocated KV tensor storage matches the logical formula in the measured prefill cells.
Allocator-reserved memory, peak memory, and total process memory are different quantities.

| B1 history | Logical KV, GiB | Allocated KV, GiB | Measured peak total, GiB |
|---|---:|---:|---:|
| 1K | 0.75 | 0.75 | 3.514 |
| 4K | 3.00 | 3.00 | 5.944 |
| 8K | 6.00 | 6.00 | 9.184 |
| 16K | 12.00 | 12.00 | 15.665 |

The peak includes the 32 cached forwards and their growing KV.
At 8K/B8 and 16K/B4, logical KV is 48 GiB.
Measured peak allocation is approximately 54.52 GiB in both cases.
The 16K/B8 preflight requires approximately 96 GiB for KV alone.
The guard skips that cell before allocation.

## Traffic interpretation

The HF append path concatenates the prior cache with fresh K/V.
An approximate operand-volume proxy counts an old read and a new write.
That proxy is about twice the existing KV capacity per cached step.
At B1/16K it is approximately 24 GiB per step, before small fresh rows.
This is structural accounting, not measured HBM traffic.
It does not include every kernel, allocator, or cache effect.

`ERR_NVGPUCTRPERM` prevents hardware-counter attribution.
No result here claims measured DRAM bytes, L2 hit rate, or HBM bandwidth saturation.
The measured cache interval supports a runtime-cost observation, not a traffic-counter claim.

## Native TP storage

The native ownership audit records distinct depth-indexed attention modules.
TP4 owns four KV heads per rank.
TP2 owns eight KV heads per rank.
TP1 owns sixteen KV heads.
The runtime reserves approximately 32 GiB of KV pool storage per rank.
This reservation is not the live KV footprint of one request.

The ownership JSON includes actual parameter storage and KV backing tensors.
TP4 rank 0 has 717,624,322 unique parameter bytes.
Its unique KV pool storage is 34,357,641,216 bytes.
The native cache uses pages and writes fresh rows.
It does not reproduce the reference cache's full-prefix `torch.cat` append.

## Candidate capacity bounds

| Placement model | Weight ownership | KV ownership | Required transfer |
|---|---|---|---|
| Conventional CP microbenchmark | Full operator weights remain outside the measured attention operator | One historical sequence shard per GPU | Query and exact output/LSE merge |
| Layer-stationary wavefront | Six physical layers per GPU, reused across four loops | All four depths for those layers | Hidden state across stage and wrap boundaries |
| Loop-stationary wavefront | Full tied 24-layer block on each GPU | One depth per GPU | Hidden state across three depth boundaries |

At B1/16K, balanced layer/depth KV ownership is approximately 3 GiB per GPU.
A conservative loop-stationary bound also retains all 2.672249 GiB of model weights per GPU.
This capacity estimate fits assigned HBM with substantial headroom.
It is not an allocation measurement of an implemented pipeline.
Workspace, graph pools, transfer buffers, and allocator slack still require implementation validation.
KV stays local to its assigned depth/layer in these wavefront models.
The simulator therefore does not invent full-KV transfers at each stage.
