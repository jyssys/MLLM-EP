# Prefill secondary analysis

The selected implementation performs full-R4 prompt prefill.
Its adaptive gate controls subsequent decode tokens.
The first generated token belongs to the full-depth prefill path.

The v2 physical probe and the new v3 runs record actual prefill recurrent work.
That work equals four times the processed prompt-token count.
Chunking can divide the prompt into native cache-budget chunks.
It does not introduce adaptive prompt-token early exit.

The status is `PREFILL_FULL_DEPTH`.
There is no evidence for a prefill-depth placement advantage in this runtime.
The DP baselines can use known prompt length as a generic cost feature.
Every scheduling model includes prefill occupancy and TTFT.

No new prefill engine or token-selective prefill policy was implemented.
