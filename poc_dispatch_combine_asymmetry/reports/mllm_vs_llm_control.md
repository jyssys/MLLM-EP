# Same-model modality and standalone LLM control

| Arm | tokens | Dispatch ms/layer | Expert | Combine | mean request D/C ratio |
|---|---:|---:|---:|---:|---:|
| VL visual low | 175 | 0.9131 | 0.4020 | 0.1131 | 8.085 |
| VL visual medium | 721 | 0.9305 | 0.4507 | 0.0826 | 11.339 |
| VL visual high | 2029 | 1.1092 | 0.5264 | 0.1246 | 9.051 |
| VL text low | 174 | 0.4703 | 0.3973 | 0.1249 | 3.796 |
| VL text medium | 724 | 0.5498 | 0.4298 | 0.1219 | 4.747 |
| VL text high | 2033 | 0.5184 | 0.5114 | 0.1331 | 3.927 |
| Qwen3 LLM text low | 174 | 1.4125 | 0.4005 | 0.4538 | 3.114 |
| Qwen3 LLM text medium | 724 | 1.4733 | 0.4376 | 0.4471 | 3.296 |
| Qwen3 LLM text high | 2033 | 1.4405 | 0.5070 | 0.4140 | 3.559 |


Same VL model: high visual Dispatch rises 115.3% with paired CI excluding zero. Combine does not materially rise. Standalone LLM has large generic Dispatch/Combine asymmetry too, and its Dispatch is GREATER than VL visual in these native blocks. Therefore MLLM_VISUAL_DISPATCH_EXCESS_STRONG criteria fail.

Both models use SP TP partitioning (not two full input copies) and exactly the same loaded MoE geometry and backend. Actual route IDs/weights vary because checkpoints/contexts differ. Actual source layout and padding/dummy rows are exported; all4 expert owner maps verified for every captured layer. The models' non-MoE producer, positional and attention paths need not have identical readiness; no architecture-causal conclusion is drawn from unpaired absolute differences.

Volume alone does not explain native timing: token counts change nearly12x, but large control/readiness floors persist. Useful traffic and skew are tabulated separately. Visual/text traffic differences are modest relative to native timing excess; token-and-source-matched replay tests bare route transport separately.
