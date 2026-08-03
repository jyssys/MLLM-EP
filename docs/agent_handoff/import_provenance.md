# Import provenance and artifact policy

Date: 2026-08-03 KST

## Provenance

The source directory `/home/esjung/MLLM-EP` was a restored research archive
without `.git` metadata. Its original branch, commit IDs, dirty-worktree state,
and author chronology were therefore unavailable. This Git repository begins
with a truthful snapshot through FlashVEP Phase 1b rather than fabricated
historical commits.

The GitHub import preserves project source, environment manifests, reports,
prompts, final analysis, final Phase 1b stage events, request measurements,
derived Nsight CSVs, and earlier compact result summaries. It excludes model
weights, caches, credentials, vendored upstream archives, Python bytecode,
failed/superseded smoke-run directories, and bulky raw profiler streams.

## Large local evidence omitted from Git

These files remain in the original local archive and can be transferred out of
band if an independent reviewer needs them. Paths are relative to the archive.

| SHA-256 | Path |
|---|---|
| `aaf4105035787caee208f5e9487df3e4469c8799fcbb5a22aa4683ec55b92df4` | `poc_flashvep/results/phase1b_tp2dp2_nsys_224/trace.sqlite` |
| `32d8fa8813555620706baeb90b6f56bae0252f60ad2142e563ab62cf0af5f6c2` | `poc_flashvep/results/phase1b_tp2dp2_nsys_224/trace.nsys-rep` |
| `3881f34c14f250d6510263d772c4d7f1d46f56539e6002c80d27ce90d02444e7` | `poc_flashvep/results/phase1b_tp2dp2_vision896/audit.jsonl` |
| `5d34fe6cd6c9e31b40402f32bc60915c3d0769d2525d34f45e21fa1c28f33008` | `poc_flashvep/results/phase1b_tp2dp2_vision896/stage_events.jsonl` (superseded by tracked `stage_events_v2.jsonl`) |
| `bfe230bb2fd86f968b709dc0eb785d9ccd9707089094eb9b3b4ebe61f5a6156b` | `poc_flashvep/results/tp4_phase1_vision896/stages.jsonl` |
| `a868ca76d4a25a99fc690ebc3c4fe206bbc384456bd9028881cb24bfa272088b` | `poc_flashvep/results/tp4_phase1_vision896/lean_stages.jsonl` |

The tracked `stage_events_v2.jsonl`, `analysis_final.json`, gate JSON, report,
and derived `nvtx_kernels_nvtx_kern_sum.csv` are the canonical compact evidence
for ordinary agent review.
