# Experiment log

* 15 successful Qwen3-VL serving runs (3 stock, 3 global sync, 3 unconditional
  comm-stream drain, 3 invocation-ID oracle, 3 previous-dispatch simple).
* 773,760 MoE invocation records; all use physical GPUs 1–4 only.
* `previous_event_present=false` for every record; native readiness query is
  unavailable in the installed DeepEP binding.
* Event-wait proxy strongly predicts dispatch tails, but selective stream
  drain/oracle did not meet the E2E gate and can increase p50/tails.
