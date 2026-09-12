# Quality, latency, and throughput

## Quality protocol

The same checkpoint revision, temperature 0, threshold decoder, block length
32, generation budget 128, BF16, prompt set, and evaluation code were used for
TP4 and routed EP4. GSM8K uses 32 bounded examples and five independent engine
restarts. HumanEval uses 32 bounded examples as a one-restart screen.

| Task | TP4 | EP4 | conclusion |
|---|---:|---:|---|
| GSM8K-32 | 5/32 in every restart | 5/32 in every restart | score preserved |
| HumanEval-32 | 4/32 | 4/32 | score preserved |

Surface-string exactness is lower (GSM8K 15/32 and HumanEval 19/32 for the
paired outputs) because BF16 collective/reduction ordering changes the
threshold-decoder trajectory. The task metric, not byte-identical strings, is
the quality gate.

## Longer-run performance

For GSM8K-32, unpaired medians are:

| Topology | BCT | generated tokens/s | NFE |
|---|---:|---:|---:|
| TP4 | 34.396 s | 110.97 | 519 |
| EP4 | 38.396 s | 99.41 | 504 |

The paired EP wall gains are +29.51%, -27.42%, -33.80%, +8.02%, and -54.45%.
Thus the median is negative and the topology effect is not robust to shared
machine state. HumanEval's single pair favors EP4 by 28.00%, but one run cannot
overrule the five-run GSM8K result.

## Throughput regime

At submitted batch 16 and each topology's best tested static microbatch, EP4
reaches 238.41 tokens/s while TP4 reaches 291.86 tokens/s. At the *matched*
microbatch 8, EP4 reaches 238.41 versus TP4 216.29 tokens/s. This is evidence
that decomposition and execution granularity interact; it is not evidence
that EP4 is uniformly faster.

## Correct interpretation

The substrate is quality-valid for characterization. It is not bitwise exact,
and the bounded task sample is not a full model-quality evaluation. Since no
semantic method was implemented, no broader accuracy claim is made.
