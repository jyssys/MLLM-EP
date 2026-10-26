# Correctness and quality

| Model | EP | Checks passed | Max logit relative L2 | Max |ΔNLL| | Top-1 equal / positions | Native greedy |
| --- | --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| IFM Ouro | 8 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| IFM Huginn | 4 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| IFM Huginn | 8 | 105/105 | 0.0 | 0.0 | 30497/30497 | 8/8 exact |
| Foil-1 | 4 | 105/105 | 0.0 | 0.0 | 30733/30733 | 8/8 exact |
| Foil-1 | 8 | 105/105 | 0.0 | 0.0 | 30733/30733 | 8/8 exact |

The frozen limits remain in `methodology.md`.
All six EP cells meet those limits without relaxation.
The native repeated-logit check is exact.
The owner tensor checks confirm exact checkpoint subsets.
The distributed execution count is 112 calls for either IFM model and 18 for Foil.
The ready observer also checks every request's physical-site call count.
Its eight-token independent outputs match sequential B1 references.
Four complete steady-generation repetitions also retain identical output token IDs.

These checks establish sampled runtime fidelity, not complete downstream accuracy.
GSM8K/MATH-500 teacher forcing is not answer scoring.
EOS-disabled hardware generation is not natural answer evaluation.
There is no candidate optimizer to qualify because the headroom gate failed.
Candidate A/B numerical tests and downstream answer accuracy are NOT_RUN.

The earlier single-GPU skipping/reuse NO-GO remains unchanged.
The earlier FA4-versus-explicit-attention limitation also remains unchanged.
No earlier artifact was rewritten.
