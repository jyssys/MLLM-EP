# Multistage pipeline oracles

## Measured local pipeline

The legal complete-wave schedule is:

```text
communication stream: combine(i-1) -> dispatch(i+1)
compute stream:       expert(i) -------------------
```

DeepEP retains communication-stream order while the independent expert wave runs
on the compute stream. Exact expert and combine outputs have cosine >=0.99999988
and relative L2 0 in the diagnostic.

## Scope-correct economic mapping

Projecting the median of the three measured shape points across 31 MoE layers and
all denoising waves gives an intentionally optimistic steady-state
independent-wave service upper. A second column makes the even stronger and
unrealistic assumption that the best observed shape saving occurs at every wave:

| dataset | sampled-shape median service upper | best observed saving at every wave | direct request latency oracle |
|---|---:|---:|---:|
| GSM8K | 4.35% | 4.84% | 0% / not legally realizable within one request |
| HumanEval | 4.59% | 6.73% | 0% / not legally realizable within one request |

This upper bound ignores pipeline fill/drain, scheduling, and queueing. Creating
independent waves by splitting the best-static mini32 batch is not free: the
preceding RAWS PoC found mini32 is the best static point and a valid per-wave
dynamic oracle of only 0.65%. Therefore the diagnostic 3-stage gain cannot be
turned into direct request E2E by assuming zero-cost fragmentation.

Even as a service-throughput candidate it is below the 8% promotion gate and is
heavily adjacent to TBO, COMET, StreamEP, and X-Stage. No multistage production
prototype is justified.
