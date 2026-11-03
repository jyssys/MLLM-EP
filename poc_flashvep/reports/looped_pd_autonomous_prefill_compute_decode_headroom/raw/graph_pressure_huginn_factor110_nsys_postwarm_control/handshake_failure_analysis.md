# Capture handshake failure

Classification: instrumentation control failure. This run does not test the intended profiling hypothesis.

The warm-up marker reached `ALL_COMPLETE`.
The coordinator wrote `postwarm_capture_requested.json`.
The worker did not create `postwarm_capture_ready.json`.
No `capture_events.jsonl` start marker was recorded.

The captured P-worker stack stopped at `vllm_rlt/engine/worker.py:197`.
The native worker was idle. It calls `engine.step()` only when requests remain unfinished.
The first handshake depended on `StageEngine._step()` and therefore could not start on an idle worker.

The watchdog terminated the run after 109.28852810151875 seconds.
The dispatcher recorded return code -15 and `complete=false`.
This failure does not establish a CUDA, NIXL, or model-correctness failure.

The next run uses the native stats command to wake the control path.
It has a new namespace: `graph_pressure_huginn_factor110_nsys_postwarm_stats_handshake`.
The first run and its source archive remain unchanged.
