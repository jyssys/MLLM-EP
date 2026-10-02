# Combine decomposition

Installed DeepEPHTPrepareAndFinalize._finalize uses the dispatch handle's rank/channel prefix matrices, source indices and send_head. TopKWeightAndReduce combines expert contributions locally first. Buffer.combine reuses the handle and does not recompute dispatch layout or perform fresh dynamic receive-count polling/allocation based on new routes.

| Boundary | Label | Status |
|---|---|---|
| A local weighted reduce/layout | native finalize composite DIRECTLY_TIMED | exclusive preparation not separately timed |
| B reverse payload movement | intranode::combine KERNEL_CLASSIFIED | fused with reduce/scatter/polling; pure wire N/A |
| C source-row reduction/scatter | same combine kernel + output.copy_ | fused/unresolved exclusive duration |
| D barriers/epilogue | cached_notify_combine KERNEL_CLASSIFIED; final event wait/output copy composite residual INFERRED_RESIDUAL | source attribution only, not fabricated independent ms |

buffer.combine API interval is DIRECTLY_TIMED in the perturbed diagnostic lane only. Reverse movement carries one source contribution per token/destination after local expert reduction, not8 independent full hidden rows. Useful logical D/C hidden bytes are equal under destination dedup; smaller Combine time cannot be explained merely by fewer useful hidden bytes. Cached metadata/less fresh count coordination is a source-backed candidate, not a counterfactual measured speedup.
