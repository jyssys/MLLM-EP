# Training-Free EP-Aware Unmasking PoC

Discovery order: confidence slack, near-tie EP-cost heterogeneity, matched
one-step oracle, and only then full counterfactual trajectories if justified.

Observer-heavy unmask capture is opt-in via `EP_UNMASK_ATLAS_DIR` and a
`PYTHONPATH` prefix containing `instrumentation/`.  The stock dInfer and
model code are left unchanged.  Clean latency runs must omit this prefix.

The working contract is
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/training_free_ep_aware_unmasking_poc_spec.md`.

Current decision: `NO-GO` at the confidence/physical one-step screen. The
direct alternative request-E2E oracle was not measured; the gate authorized
stopping before a causal full rollout. See `reports/final_decision.md` for
the measured-versus-derived boundary and answers to the spec's 20 questions.

Raw run data under `results/` are local and intentionally git-ignored;
cross-task atlas artifacts and reports are tracked in this PoC directory.
