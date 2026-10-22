# Fairness and complete-token exposure

Native fairness counts issued recurrent batches.
It does not wait for a complete R4 output token before resetting the prefill counter.
The cost dataset contains 176,787 observed inter-token gaps.
Of these, 43,216 contain at least two prefill submissions between observed outputs.

`full_token_exposure.csv.gz` preserves each request ID, gap, duration, and interruption count.
The count uses actual host submission timestamps between client-observed output timestamps.
It is not a count of completed prefill quanta or a causal estimate of added delay.
Some submitted work overlaps already pending operations.

The next token can still require multiple loops and coda/sampling.
Therefore a shorter G2 quantum need not create one completed output between quanta.
Repeated short interruptions can worsen p95 even when extreme individual stalls shrink.

No complete-output-token-aware fairness policy was executed.
Its effect remains NOT_RUN.
It must remain a separate ablation from C/G choice in any future online comparison.
