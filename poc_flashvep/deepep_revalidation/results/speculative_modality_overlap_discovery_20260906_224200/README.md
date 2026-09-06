# Speculative modality-overlap discovery

This directory contains the bounded, trace-driven discovery harness for
Qwen3-VL MoE EP.  It deliberately separates:

1. fresh live vLLM/DeepEP captures;
2. partial expert-output quality analysis; and
3. a conservative overlap oracle.

No routing, model math, scheduler, or production kernel is changed.  A
candidate is promoted only if quality, overlap headroom, verification cost,
and an end-to-end critical-path estimate all pass.

`analyze_partial.py` reads the raw expert-output capture emitted by the
existing `visual_expert_functional_redundancy` hook without requiring a GUI or
the vLLM runtime.  Its outputs are diagnostic evidence; they are not an
online speculative implementation.
