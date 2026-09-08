# Honest introduction test — candidate not promoted

1. MoE serving can be memory-bound during small-batch generation even when only
   a few expert assignments are executed per token.
2. SERE reduces the expert working set by mapping secondary assignments into a
   calibrated,shared batch-primary set without reducing the original slot count.
3. This design explicitly depends on the composition of that primary set and
   exposes S and rho to control the approximation trade-off.
4. In our bounded Qwen-VL tests,aggressive B1 substitution causes substantial
   task-quality loss,and larger or more conservative primary sets reduce it.
5. Crucially,existing S4/rho.5 recovers100%/88.9% of the observed ChartQA/GQA
   loss at approximately the same port latency.
6. The remaining measured port cost has a large algorithmic no-op component,
   so it does not establish a new10% quality-matched successor opportunity.
7. Therefore these data support careful configuration and reproduction guidance,
   not a claim that a new MLLM-specific execution principle is necessary.

The positive successor Introduction requested by the research contract cannot
honestly be written: the crucial "simple tuning cannot solve it" sentence is
falsified by the experiment. This is the reason to decline paper promotion.
