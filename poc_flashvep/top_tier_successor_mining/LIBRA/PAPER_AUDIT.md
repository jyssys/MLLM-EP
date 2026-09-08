# Libra — final ICLR paper audit

[Final proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ff1ac9a659085fed0735362cafe5e53-Abstract-Conference.html).
The final 22-page PDF, including algorithms and E2E tables, was downloaded and
read. Status: PAPER UNDERSTOOD. User-supplied supplementary source is now
available and native-system sanity has begun; see `SUPPLEMENT_AUDIT.md`.

## Important mechanism

The current layer input is evaluated with the **next layer's gate**. Predicted
token-expert demand drives two-stage replication: local-demand remote experts
first, then hottest-expert replication from overloaded to underloaded GPUs.
Actual routing is never approximated. Incorrect prediction affects placement cost
and locality, not model math.

MoE execution is split into local and remote work. CPU token-sharding runs during
local work. AllGather dispatch broadcasts hidden states before token-sharding
decisions are complete. Copy-engine expert replication for the next layer overlaps
the current layer using even/odd double buffers. This is **not** the stock DeepEP
execution path with a predictor hook added.

## Original system and limitations already acknowledged

- SGLang 0.4.10, Cython planning/sharding, PyTorch SymmetricMemory.
- BF16 Qwen3-235B-A22B and GLM-4.5-355B on 8 H200; prefill primary.
- N=8 additional experts per GPU, alpha=.5 in evaluation.
- Prediction accuracy is per-token top-k set overlap, not previous top-k reuse.
- Short-context local compute can fail to hide CPU overhead (Appendix D).
- Replication capacity depends on memory and available compute/copy window.
- 19.2% headline is throughput vs baselines, not universal request-E2E gain.

## Transfer questions

Measure genuine lookahead prediction for text/vision/conditioned decode, then
translate prediction error into actual missed-replica benefit. Accuracy alone
cannot establish a material performance failure. DeepStack and attention updates
are possible explanatory variables, not preselected conclusions. Match token
volume, source demand and layers before interpreting modality.

Required prior-art adversary: PROBE already combines a gate-initialized learned
lookahead predictor, hiding-window-aware replication and communication scheduling.
Simply adding a trained predictor or avoiding A2A/prefetch interference is crowded.
