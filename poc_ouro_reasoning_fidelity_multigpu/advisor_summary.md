# Advisor summary

The earlier 64-token forced outputs did not establish reasoning accuracy, and the earlier public AIME baseline had substantial censoring.
The new campaign executes 460 natural quality generations across the frozen caps without changing the checkpoint, R4, BF16, or source-selected P1 prompt.
At the largest sampled cap, AIME completed accuracy is 52/90, or 57.78%, with a problem-bootstrap 95% interval of 43.33%–72.22%.
The official 64.7% score lies within that interval, but the private harness and judge prevent an exact protocol-reproduction claim.

Natural termination remains 79/90 for sampled AIME and 21/30 for greedy AIME, while MATH100 reaches 87/100 completed correct and 91/100 EOS outputs.
HF and native distributions satisfy the functional confirmation limit, but strict state limits and ordinary-session long-output reproducibility remain unresolved.
The small TP1/2/4 audit completes two MATH problems correctly, but its differing output lengths and missing concurrent TP×DP measurements prevent a serving-layout conclusion.
The final decision is weak quality, partial reference fidelity, and partial systems evidence, so no recurrence-specific framework or measured new-method gain follows.
