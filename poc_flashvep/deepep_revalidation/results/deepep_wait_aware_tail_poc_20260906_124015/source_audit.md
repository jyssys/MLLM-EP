# Source audit (run root)

Local vLLM `0.20.0+cu129` `DeepEPHTPrepareAndFinalize` calls
`dbo_get_previous_event(self.buffer.capture)` before
`Buffer.get_dispatch_layout`/`Buffer.dispatch`, waits on returned dispatch
`EventOverlap` before expert execution, and passes another previous event into
`Buffer.combine`.  `dbo_get_previous_event` returns no event when DBO is off
in this run.  DeepEP `EventHandle` is a C++ wrapper around `torch::Event` and
the installed pybind exports only `current_stream_wait()`; there is no query
operation.  DeepEP's asynchronous intranode notify/barrier remains the
likely internal dependency observed in the preceding Nsight run.

The local hook records same-device CUDA event durations for layout, dispatch,
expert, combine, and `EventOverlap.current_stream_wait`, plus intervention
metadata.  It does not alter routing or model math.
