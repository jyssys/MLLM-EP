# Folded tensor/sequence layout: transfer screen

Status: source audit and explicit volume accounting.
No folded runtime or candidate speedup ran.
This screen supplements C04. It does not create a new candidate.

## Primary-source facts

[Folding Tensor and Sequence Parallelism](https://arxiv.org/html/2604.26294v1), by Shyam, Golubeva, and Anthony, folds both shards onto one device axis.
Version 1 dates from 29 April 2026.
Section III broadcasts attention weights and exchanges sequence KV.
Its MLP schedule rotates weight shards while each rank accumulates its local token outputs.
Section IV, Table II, charges weight movement separately from activation communication.
Section VI uses MI300X hardware.
Its forward-throughput measurements do not establish H100 cached-decode latency.
The paper also measures training and reports training-memory expressions.
Those expressions cannot become inference KV capacity.
I inspected Sections II–IV and VI–VII for this transfer screen.
I did not locate an implementation link in the inspected article.
This bounded absence does not prove that no public code exists.

## Ouro-specific arithmetic

The pinned Ouro geometry has hidden width 2048 and MLP width 5632.
The following quantities exclude embedding, normalization, and the output head.

```text
attention projection weights = 4 * 2048^2 * 2 bytes = 32 MiB/layer
MLP weights = 3 * 2048 * 5632 * 2 bytes = 66 MiB/layer
```

The [CPU script](../scripts/folded_layout_screen.py) applies the paper's declared per-rank volume convention.
It does not measure wire traffic or HBM traffic.
It does not turn protocol bytes into latency.
Its [raw record](../artifacts/folded_layout_volume_screen_v1.json) preserves all assumptions.

| Devices | Full-prefill volume crossover, fresh rows | MLP-only volume crossover, fresh rows | Status |
|---|---:|---:|---|
| 2 | 16640 | 8448 | CPU accounting |
| 4 | 13909.3333333333 | 8448 | CPU accounting |
| 8 | 13129.1428571429 | 8448 | Unmeasured projection only |

The full-prefill crossover assumes the stated attention broadcasts and full-sequence KV exchange.
It is not a cached-decode formula.
In decode, historical length 8192 does not provide 8192 fresh MLP rows.
The B1 path has one new token row at that step.
An exact sequence layout still needs its declared attention reads and state publication.

## Recurrence test

Repeating this uncached schedule four times multiplies both protocol volumes by four.
Weight identity alone does not lower the ratio.
Persistent retention can avoid a future weight transfer only if the relevant receiver retains those exact weights.
Retaining every visited shard across every physical layer creates replicated weight ownership.
That alternative needs a generic replicated-weight/SP baseline and explicit capacity accounting.
Selective retention remains possible. This screen neither measures nor rejects every selective schedule.

The current C1–C8 decode target supplies too few fresh rows to assume large prefill-GEMM amortization.
This is an analytic caution, not a physical slowdown result.
No unique recurrence-specific removed message follows from the inspected mechanism.
The current classification remains `GENERIC_ONLY` for this direct transfer.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES='' /home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.folded_layout_screen
```

The script refuses to overwrite its versioned output.
