# Overlap candidate shortlist

The list is a shortlist of hypotheses, not an optimization implementation. Prior paired encoder+DeepEP and encoder+expert measurements remain explicitly negative.

1. **CPU scheduler/request preparation + DeepEP dispatch** — CPU-side work is cross-request independent and has low direct GPU-resource overlap risk. Evidence: FULL_SERVING_OBSERVED. Risk: host work may not cover the communication window. Cheapest next PoC: one NVTX-marked CPU preparation plus exact-route dispatch.
2. **Vision merger/projector + DeepEP combine** — a short encoder tail versus a communication phase. Evidence: FULL_SERVING_OBSERVED. Risk: the merger shares memory resources and the prior full encoder+combine pair was negative. Cheapest next PoC: one natural merger unit with exact replay.
3. **Decode attention + DeepEP dispatch** — cross-request independent but conditional. Evidence: FULL_SERVING_OBSERVED. Risk: decode attention is latency-sensitive and may contend for HBM. Cheapest next PoC: one mixed prefill/decode trace.

Full-serving CUDA kernels and NVTX phase ranges are now captured; compatibility labels remain conditional because this atlas does not schedule concurrent work.
