# Workload contract

The experiment uses eight metadata-selected requests per model. The domains are coding, mathematics, and document text.
The domain counts are 3, 3, and 2. No latency or model output determined selection.

`results/frozen_workload.json` records source revisions and token hashes.
`results/frozen_tokens.json.gz` contains the actual input IDs and teacher IDs.
Ouro reuses the previous frozen input pool for continuity. These requests are not fresh held-out data.

The primary prompt lengths are 1,024 and 16,384 tokens. Every request produces exactly 64 teacher-forced tokens.
The confirmation lane uses 32,704 input tokens and 64 output tokens. It contains three requests, one per domain.
The maximum total position count is 32,768. No RoPE extrapolation occurs.

The tokenizers differ. Equal token counts control sequence shape, not semantic difficulty.
Within one model and request, every TP, R, and execution family receives identical input and continuation IDs.
The 64 continuation IDs come from source text. They are hardware controls, not model answers or accuracy evidence.

Each configuration uses three warmup requests per prompt length. It then uses three technical repeats per request.
The analysis averages technical repeats before comparison. It resamples requests with 5,000 paired bootstrap draws.
Ranks, layers, recurrent passes, and output tokens are not independent samples.

A separate bounded natural lane uses EOS-enabled generation. That lane reports output length, stop reason, and repeat parity.
It does not provide benchmark accuracy. Generated code is not executed.
Ouro R1 and R2 change effective model depth. They are not quality-preserving alternatives to pretrained R4.

The protocol permits 24 fresh held-out requests only after a plausible recurrence-specific effect appears.
A negative decisive control does not require a larger held-out campaign.
The optional Qwen3-8B reference cannot replace either required Qwen model.
