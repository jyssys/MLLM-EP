# Runtime/backend proof

- `runtime_proof.json` in the TP-only run records TP4/DP1 and `MoEPrepareAndFinalizeNoDPEPModular`.
- `runtime_proof.dp0.json` and `runtime_proof.dp1.json` record TP2/DP2/EP4, `use_sequence_parallel_moe=true`, and `all2all_backend=deepep_high_throughput`.
- Real DeepEP startup log emitted `Using DeepEPHTAll2AllManager all2all manager.` and the model worker emitted `Using DeepEPHTPrepareAndFinalize`.
- Every retained real-DeepEP layer row has `prepare_finalize_backend=DeepEPHTPrepareAndFinalize` and `expert_backend=TritonExperts`.
- vLLM source (`vllm/model_executor/layers/fused_moe/config.py`) defines `use_all2all_kernels = dp_size > 1 and use_ep`; therefore the TP2/DP2/EP4 run is the active all-to-all path while TP4/DP1 is not.
