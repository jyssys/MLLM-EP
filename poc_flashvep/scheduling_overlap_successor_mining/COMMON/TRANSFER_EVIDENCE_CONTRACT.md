# Transfer evidence and limits, before MoE/MLLM timing

## Native versus trace-driven evidence

The three official code bases do not share the project's vLLM 0.20 DeepEP
runtime. Preserve their actual execution semantics rather than relabeling a
vLLM run as a paper reproduction.

1. Layered Prefill: native Qwen3-30B-A3B TP2 implements the scheduling mechanism.
   Its current model does not implement the Qwen3-VL vision encoder, MRoPE or
   DeepStack injection. A measured Qwen3-VL layer-cost trace can diagnose the
   equal-contiguous-group assumption, but cannot alone establish Layered
   Prefill request-level gain on Qwen3-VL.
2. FastPP: native dense PP4 and Qwen3-MoE PP4 are source-supported. The pinned
   Qwen3 expert implementation is not DeepEP, even when an EP-looking flag is
   exposed. Real Qwen3-VL EP4 operation traces may inform a partition/cost
   diagnostic; a PP×EP oracle needs a faithful dependency/request timeline and
   cannot be presented as native PP×EP performance.
3. NanoFlow: the official H100 Qwen1.5-MoE EP4 path is a supported, ungated
   numerical baseline. A bounded fixed-plan adapter may expose its existing
   splitter/executor, but the adapter is neither a new method nor the paper's
   searched optimum. Graph, nano-split, stream placement and model outputs each
   need live verification. Native fixed-cohort results do not certify continuous
   batching under changing shapes.

## Promotion restrictions

- A port/build/API failure is an environment or port limitation, never a method
  failure. Correctness-invalid runs are excluded from performance evidence.
- Native text/MoE request-level comparisons have precedence over hypothetical
  MLLM projections. Every trace/oracle table names its actual baseline runtime.
- Measured layer-cost partition savings, pipeline bubbles and operation overlap
  are diagnostics, not automatically direct request E2E savings.
- A request-level oracle must replay the same arrivals, admissions, dependencies
  and output work. One may not add independently critical rank durations or
  subtract unrelated cross-GPU absolute CUDA timestamps.
- Encoder cost dilution is Amdahl's law, not a new MLLM-specific failure.
- Neither old project's policy hooks nor its measured latencies become fresh
  evidence in this study. Reusing exact real-image inputs and a validated renderer
  is permitted; model/routing/scheduler policy modifications are not inherited.
- If only a diagnostic is feasible for one milestone, report its narrower scope
  explicitly in the common milestone table before ranking.
