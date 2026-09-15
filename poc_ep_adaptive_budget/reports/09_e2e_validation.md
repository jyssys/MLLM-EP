# 09 — Clean E2E and reproducibility validation

All headline BCT values are actual end-of-generation wave times summed over
each matched controlled pool. No per-step savings were summed as a request
oracle. GSM8K comparisons hold the same official problem IDs, input-length
sorting, submitted=32, mini=32, gen-request=128, B32, dtype, model, decoder
and EP4 path. HumanEval full uses matched mini16 because mini32 OOMed after
three of six waves. Promoted quality uses executable HumanEval pass tests
and GSM8K numeric benchmark answers, not token identity.

| Clean cohort | Baseline score/NFE/BCT | Phase score/NFE/BCT | Direct BCT gain | Independent restarts |
|---|---|---|---:|---:|
| GSM8K 512 | 68/512; 1,300; median 100.336 s | 67/512; 1,190; median 87.361 s | 12.93% descriptive | 3 each, randomized pairing order |
| GSM8K full 1,319 | 144/1,319; 3,226; 227.782 s | 146/1,319; 2,965; 212.089 s | 6.89% | 1 each |
| HumanEval full 164 | 13/164; 672; 47.155 s | 14/164; 613; 45.294 s | 3.95% | 1 each |

For n=512, baseline restart BCT ranged 92.418--113.864 s; phase ranged
85.558--95.165 s, with overlapping ranges. Paired per-wave bootstrap CIs
are conditional on one process and *not* independent restart evidence.
Full-pool baseline absolute benchmark scores were low (GSM8K 10.9%,
HumanEval 7.9%), limiting a broad quality-preservation claim. The benchmark
has a crucial cohort confound: the first 32 GSM8K questions scored 13/32 in
an n=32 pool but 5/32 inside the n=512 length-sorted canvas, with different
generated lengths. All promoted comparisons remained within the same pool;
cross-pool quality or raw latency was never used as an effect size.

The clean all-phase-0.9 hook reproduced stock 32/32 final answers and NFE
exactly. The observer-heavy baseline and phase traces also reproduced their
respective clean n=32 final answers 32/32, but BCTs were 12.037 and 15.062 s
versus clean about 7.65 and 7.12 s. Exact owner-rank routing and `8*M`
conservation were checked in the structural trace; observer stage timing is
not mapped to clean E2E gains. The model/runtime hashes and per-launch GPU
UUID, owner, free-memory and NVLink audits are in [baseline truth](00_baseline_truth.md)
and the [attempt log](../ATTEMPT_LOG.csv). A random TCPStore port collision
and one HumanEval mini32 OOM were excluded, not counted as quality failures.

No online serving queueing, TTFT, MATH/MBPP, post-Epoch physical compaction,
low-overhead per-step MoE time, or EP-conditioned live runtime was measured;
none is claimed. Different forward counts also alter future trajectory:
full GSM8K final-answer sequence identity was 48.7%, HumanEval 53.7%, even
though benchmark score changes were small. Source:
[full GSM8K output drift](../results/full1phase_gsm8k1319_p00/output_drift_vs_full1_gsm8k1319_t900.json)
and [HumanEval output drift](../results/hemini16phase_humaneval164_p00/output_drift_vs_hemini16_humaneval164_t900.json).
