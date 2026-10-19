# Correction: frontend RPC serialization boundary

## Corrected scope

The earlier review missed the frontend serializer boundary.
Its statement of no source blocker was too broad.
The inspected inner executor supports callable dispatch.
That support does not establish callable transport through the frontend.
I did not test that transport before the earlier handoff.

The frozen [earlier review](process_lease_diagnostic_independent_review_v1.md) remains unchanged.
Its SHA256 remains `57c6e7ec22371122e25042f2896cb83d0254cdaf2873fe79dc7093c9ce9dc8d1`.
This addendum supersedes its end-to-end readiness implication.
Its CPU spawn/title observations remain limited to their stated CPU execution path.

## Reproduced v2 failure

The retained v2 failures report a frontend `TypeError` for a function object.
They occur before the original strict lease check.
See `outputs/native_tp2_dp2_cohort_base14_v2_diagnostic/workers/replica*/failure.json`.
The original missing-identity cause remains unresolved.
This new failure does not explain it.

Installed `MPClient.collective_rpc` sends the method through `call_utility`.
See `vllm/v1/engine/core_client.py:935–942`.
`call_utility` constructs the utility request, then `_send_input` calls `MsgpackEncoder.encode`.
See the same file at lines 861–881.
This boundary precedes the inner multiprocess executor.
The encoder rejects unsupported objects when insecure serialization is false.
See `vllm/v1/serial_utils.py:221–234`.
Function support in the inner executor cannot bypass that earlier rejection.

The independent CPU check reproduced the same exception with insecure serialization disabled.
The method-string request encoded successfully with all three expected guard keys.
Its encoded size was 146 bytes in one frame.
This size describes the CPU fixture only.
It is not a runtime cost measurement.

## String-method alternative

[HybridCohortProbe](../hybrid_cohort_probe.py) subclasses the existing `CohortProbe`.
Its bound `inspect_guard_identity(self, expected)` method calls the same allowlisted snapshot helper.
The TP2×DP2 worker now sends `'inspect_guard_identity'`, not a function object.
Only the new TP2×DP2 control selects the new extension class.
The inspected extension mechanism resolves that qualified class name before worker construction.
See `vllm/v1/worker/worker_base.py:258–287`.
The string executor branch uses `getattr(self.worker, method)`.
See `vllm/v1/executor/multiproc_executor.py:986–1000`.
`WorkerWrapperBase.__getattr__` forwards the lookup to its worker.
See `vllm/v1/worker/worker_base.py:333–334`.

The retained [CPU regression](../scripts/rpc_string_boundary_cpu_v1.py) checks the actual encoder and decoder.
It checks function rejection and string-request roundtrip.
It also checks bound dispatch through the actual `WorkerWrapperBase` attribute-forwarding method.
The utility response passes an encoder and typed-decoder roundtrip.
The [artifact](../artifacts/rpc_string_boundary_cpu_v1.json) retains the command, timestamps, and source hashes.
Insecure serialization stayed false.
`torch.cuda.is_initialized()` stayed false.
The combined diagnostic and TP2×DP2 suite passed 23 CPU tests in 0.395 seconds.

## Limits

This correction has no identified source blocker at the tested serializer and bound-dispatch boundaries.
The CPU check did not start an engine or construct a model worker.
It did not exercise socket transport or the dynamic extension inside a live native worker.
It does not prove process ownership, lease validity, or successful startup.
The allowlist and strict UID, PGID, SID, guard, and ancestry requirements remain unchanged.
No timing can start after an invalid lease.
Insecure serialization must remain disabled.
No GPU ran during this independent review.

The review began at 2026-10-19 16:55:58 UTC.
The effort ledger separates this interval from coordinator elapsed time.
