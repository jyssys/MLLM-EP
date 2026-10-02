# ROUTE_CONTROLLED_TRUE_EP_REPLAY

Actual captured BF16 inputs and native route IDs/weights, selected layers0/24/47 only. Four physical ranks0,1,4,5; one world4 CUDA group plus CPU Gloo group; DeepEP HT default20SM, 1GiB NVLink buffer, RDMA0, QP1, stock EP4 dispatch/combine config. Token counts matched174/721/2029; original source TP logical sequence restored, trimmed, resharded, native SP-padding and idle-DP dummy rows retained. All rank producers are ready at an explicit barrier OUTSIDE CUDA timers.

No attention/router/expert model compute is executed in this payload replay; received hidden values serve as combine payload. It is not native model latency or a semantic-quality test. It measures actual physical DeepEP dispatch/layout/combine. Warmup3/repeat10, aggregate three selected layers within each repeat; one critical rank maximizing D+C, no layer-independent bootstrap.

High visual Dispatch 0.1932ms versus text 0.1957ms; paired delta -0.0024ms, CI [-0.006195627624789874, 0.00115381197383006].
Combine visual 0.0842ms/text 0.0876ms.
Native visual excess vanishes in this bounded replay, favoring producer/runtime readiness rather than intrinsic visual payload transport. This does not prove all48 layers/all images have identical transport.

Initial launcher used a nonexistent venv torchrun executable and exited before replay/model work; cleanup restored burner. Corrected entrypoint python -m torch.distributed.run ran successfully. No retries of a measured result or changes to acceptance thresholds.
