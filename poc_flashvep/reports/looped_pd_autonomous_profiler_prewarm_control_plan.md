# Final Huginn collector-activation control

The CUDA 12.9 bundled Nsight 2025.1.3 control also stopped making response progress. It completed 135 measured responses. Its captured P stack is inside `nixl_cu12._api.transfer`. The watchdog retained the log and partial trace. The collector version alone did not remove the failure.

One final control starts the same collector before worker readiness and request warmup. Collection continues until process exit. This removes profiler activation and deactivation from active serving. It does not change NIXL, the model, KV ownership, scheduling, or frozen gates.

The control reuses the existing Huginn L2048/O16 clean 384-request probe run. It uses the same 1.971 QPS arrival trace and 12 requested 16-token probes. The actual completion count remains measured. The new namespace is `nsys2025_prewarm_huginn_s1_nsys384`.

The hypothesis is limited: activation during live transport may contribute to the stall. A successful run would not establish this as the unique cause. A failed run would show that this timing change is insufficient. No output mismatch or profiling perturbation can be ignored. One successful run would still lack three-start confirmation.

This is the final new Huginn profiler configuration in the authorized extension. The existing launch cutoff and GPU stop remain unchanged. All earlier failures remain in the final manifest.
