# Dispatch–Combine asymmetry discovery (TRUE EP4)

Completed fresh measurements on physical GPUs **0,1,4,5**, not recycled absolute timings from earlier GPU sets. No optimization method or latency simulator is implemented.

Main result: Dispatch≫Combine occurs in standalone Qwen3 MoE too. VL visual has additional native Dispatch delay over same-model text, concentrated in the first layer on an idle-DP waiting rank. It disappears in all-producer-ready physical route replay. This is evidence favoring producer readiness/rendezvous, **not an intrinsic visual wire-transport penalty**.

Start with [final summary](reports/final_summary.md), [instrumentation limits](reports/instrumentation_audit.md), and [reproduction](reports/reproduction.md). The complete 3-arm/3-size table, uncertainty, byte accounting and source-backed decomposition are included. One frozen input per size: repeat confidence intervals are not image-population confidence intervals.

## Resource handoff

GPU measurements are finished. At the user's subsequent request, the 0/1/4/5 burner was stopped and zero remaining compute PIDs were verified. **Do not run GPU reproduction commands or restart the burner without a new GPU allocation.** Remaining verification/publication is CPU-only. Other users' 2/3/6/7 workloads were not touched.

## CPU-only evidence verification

From repository root, with Python/numpy available:

```bash
CUDA_VISIBLE_DEVICES= python -m unittest discover -s poc_dispatch_combine_asymmetry/tests -v
CUDA_VISIBLE_DEVICES= python -m poc_dispatch_combine_asymmetry.scripts.verify_bundle
CUDA_VISIBLE_DEVICES= python -m poc_dispatch_combine_asymmetry.scripts.figures
```

The verifier reaggregates all 90 final clean waves from SHA-verified gzip raw records, rejects the known invalid first attempt, reconstructs 432 layer-level traffic accounts from actual compact route IDs, and checks all 1,728 expert ownership maps. This is independent **CPU reanalysis**, not another physical timing run.

`artifacts/communication_accounting.json.gz` is the complete logical table; readers support compressed publication and uncompressed local generation. `compact_routes.npz` contains IDs, routing scalar weights and owner maps only, **no hidden states, model tensors or images**. `compact_raw/` preserves measured wave/stage records and the excluded attempt. Large Nsight databases and bounded hidden captures remain local; classified kernel records are published.

Reports and figures are descriptive evidence. Profiler/internal-hook durations are explicitly diagnostic/UNVALIDATED where perturbed. No pure data-movement interval was separable: EffectiveBW is N/A, not an invented link bandwidth.
