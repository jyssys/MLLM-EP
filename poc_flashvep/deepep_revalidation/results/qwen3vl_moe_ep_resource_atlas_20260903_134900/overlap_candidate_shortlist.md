# Overlap candidate shortlist

Candidates below are conditional/inferred only; prior encoder+DeepEP and encoder+expert measurements are explicitly negative.

1. **CPU scheduler/request preparation + DeepEP dispatch** — CPU-side work is cross-request independent and has low direct GPU-resource overlap risk. Evidence: INFERRED. Risk: host scheduling may not cover the communication window. Cheapest next PoC: NVTX-marked CPU preparation with one exact-route dispatch.
2. **Small Vision merger/projector + DeepEP combine** — short memory-oriented encoder tail versus communication. Evidence: INFERRED. Risk: the merger may share HBM/L2 and the prior full encoder+combine pair was negative. Cheapest next PoC: one natural merger unit with the exact replay.
3. **Decode attention + DeepEP dispatch** — cross-request independent but conditional. Evidence: INFERRED. Risk: decode attention is latency-sensitive and can contend for HBM. Cheapest next PoC: one mixed prefill/decode trace.

No candidate is labelled HIGH: the available real paired measurements are MEASURED_NEGATIVE, and the bounded replay has no full-serving NVTX ranges.
