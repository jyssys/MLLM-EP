# Whole-invocation switching oracle

For every real replay, `O0` chooses the faster of the production vLLM fused
expert and PyTorch grouped for the entire invocation. PyTorch grouped wins all
102 scope/case/restart comparisons. Consequently:

- current production to grouped replacement: 22.08%/20.52% owner-local
  operator gain for GSM8K/HumanEval;
- Amdahl projection of that replacement: 6.51%/6.06% request E2E;
- perfect per-invocation switching **beyond the strongest whole kernel: 0%**.

This matters for baseline discipline. Comparing a new hybrid only to the
untuned production default would manufacture a positive result from an existing
PyTorch backend. RefineGEMM must beat grouped whole-invocation execution.

Figure: [13_whole_invocation_switching_oracle.png](../figures/13_whole_invocation_switching_oracle.png).
Numbers: [REFINEGEMM_ORACLES.csv](../REFINEGEMM_ORACLES.csv).

