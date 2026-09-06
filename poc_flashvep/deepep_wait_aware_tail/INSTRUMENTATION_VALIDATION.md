# Instrumentation validation

* 15 completed serving runs on TP2/DP2/EP4 with identical hook and model path;
  773,760 invocation records were parsed.
* Every invocation has same-device CUDA event durations for layout, dispatch,
  expert, combine, and whole-MoE.  Cross-GPU CUDA clocks were never compared.
* `previous_event_present=false` for all records because DBO is disabled and
  the local vLLM ubatching context is empty.
* The DeepEP `EventHandle` binding exports only `current_stream_wait()`; no
  readiness/query method exists.  The observer therefore reports a
  downstream same-device wait proxy and explicitly marks native readiness as
  unavailable.
* Hook overhead is limited to event creation/recording and JSON append after
  the stock call.  Policy intervention is off for stock runs and is recorded
  per invocation for diagnostic runs.
