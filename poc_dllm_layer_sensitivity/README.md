# LLaDA2 Layer/Decision Sensitivity PoC

This directory contains the isolated true-EP4 causal study for layer-wise
refinement necessity and decision contribution versus physical cost. Raw logs
and tensor captures are intentionally ignored; all aggregated evidence,
figures, scripts, source patches, and reports are versioned.

Primary configuration: physical GPUs 0–3, dense TP4 + routed EP4, submitted
batch 32, mini-batch 32, generation 32, block length 32.

Final status: `CHARACTERIZATION-SIGNAL`.  The key retained finding is that
immediate unmask-decision agreement under routed-output staleness does not
certify future-trajectory safety.  No candidate met the 8% quality-safe,
post-liveness request-level gate, so no optimization prototype was built.

Start with `reports/final_decision.md`; the publication-style consolidated
report is `../poc_flashvep/reports/dllm_layer_decision_sensitivity_poc.md`.
