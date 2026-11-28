# Research decisions

| Decision | Premise and experimental response |
|---|---|
| Recover exact Main400 before changing workload mix | Match the historical request and media receipts, and verify every recovered file and prompt identity. Keep post-processor visual/text ratios apart from image counts. |
| Start with the pinned original 7B model and NativeStrong | Reuse the H200 EC GPU IPC/NIXL PD path, original BF16 weights and modality positions. Use real supported V1 CUDA Graph/FA3 scheduling. Freeze APC off and per-occurrence media identities. |
| Calibrate load before offering it | Independent E/P/D use original images, actual E embeddings and valid native P KV. Select candidate QPS as fractions of matching P reference. Cold paths, empty cache and running Decode stay observable. |
| Study Very-long natural first | Unmodified requests near 25K input with many real pages provide physical E work and long LM prefills. Fixed16 leaves substantially more D room than the native mixed Natural EOS workload. Persist with N64/N128 and report that the Very-long frozen 24-source roster cycles. Run disjoint larger Main400 selections separately. |
| Keep Observer Guard unchanged | Clean/profile/clean includes all contract latency and duration comparisons. Failed captures remain diagnostic. Full-cohort priming and later lower-rate scheduler observation address cold runs and measurement intervention. |
| Physically test the whole E→D path | First execute exact E→D Fixed16 N4, then compare same on-demand routing, original NativeStrong and the inherited one-shot inline claim. Treat the first on-demand router as a weaker implementation whenever the new remote export is on the critical path. Distinguish legacy native vocabulary and measured result from source geometry. |
| Compare placement on existing three H200s | Hold the helper's physical E LM residency in both E/D probe arms. Keep the unmatched native/no-extra-LM comparison separate. Use the same post-processor cohort and same selected slots. |
| Limit the novelty candidate | Track actual image-event readiness and flow, same-length text readiness replay, matched E geometry, and the representation-only routing ablation. Existing whole-request deflection and existing same-device multimodal chunking provide controls. |
