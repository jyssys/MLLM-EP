# MLLM EPD checkpoint v3

Status: ACTIVE and INCOMPLETE. Main100 remains active. Main400 and earlier evidence remain unchanged.
Original eager P/D token continuation passed15/15 isolated correctness cases, including splits inside image spans.
This is not compiled C3 token serving or a full task-quality pass.
Stock compiled execution differed first at the first post-attention RMSNorm.
Actual P/D reduction blocks were2048/4096, despite matching original inputs and generated kernel source.
A common compiled-only dynamic_scale_rblock=False control made that norm and four following GEMM captures strict.
Full prompt KV strict equality increased only from1/28 to2/28 layers. The full compiled gate remains unmet.
All16 greedy tokens matched in this one diagnostic request. No performance conclusion follows.
QPS0.5 CUDA-event clean/profile/clean passed the frozen5% observer gate in one independent initialization.
The events bracket execute calls. They do not prove exclusive kernel activity, physical idle or safe headroom.
QPS2/4 observer failures remain diagnostic. Low-load qualification does not qualify overloaded runs.
An extra original full model co-resided with E. Memory fit is separate from safe extra-prefill execution.
E and D helpers each passed all28 KV layers for128/256/512/1024/2048-token real prefixes.
The helpers used original online E embeddings. These were correctness-only, not suffix or task-completion results.
The first prefix probe failed because an unused image received an inline descriptor with no clone ACK.
The corrected probe uses existing on-demand descriptors. Issued descriptors still require clone completion before release.
The preserved CPU red test, CPU green tests and10/10 GPU cases verify this ownership correction.
Full task quality, compiled token C3, additional-prefill safety and serving improvement remain unresolved.
Live C25 low-load CUPTI and C26 prefix-cost records are excluded from this checkpoint.
CPU regression before the final cost-followup edits:99PASS. Later prefix tests:5PASS.
No complete goal or final GPU closeout is claimed. Approved burners run on unused GPUs.
