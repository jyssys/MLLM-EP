# Strongest baseline

The reproduced baseline is the previously selected production-like point, not a
weak default: submitted batch 32, `mini_batch_size=32`, generation budget 32,
diffusion block 32, threshold 0.9, config 42, BF16, dense TP4 plus routed EP4,
DeepEP normal/high-throughput communication, and fused owner-rank experts.

## Fresh clean measurements

| dataset | independent restarts | median request-pool wall | run range | median throughput | NFE |
|---|---:|---:|---:|---:|---:|
| GSM8K-32 | 3 | 5.8449 s | 5.7624--6.0893 s | 455.25 token/s | 66 |
| HumanEval-32 | 3 | 7.2621 s | 7.1177--7.4404 s | 455.65 token/s | 86 |

Peak allocated-device readings reached 77,693 MiB (GSM8K) and 80,945 MiB
(HumanEval). GPU utilization samples include model load and teardown idle time,
so their median is not reported as steady-state utilization.

## Component mass

The following is observer-heavy critical-rank CUDA-event attribution divided by
the clean wall; it is an upper-bound ledger, not an additive clean timeline.

| component | GSM8K clean-E2E upper share | HumanEval clean-E2E upper share |
|---|---:|---:|
| router | 6.43% | 7.22% |
| dispatch | 12.07% | 13.32% |
| owner expert | 29.47% | 29.55% |
| combine | 6.41% | 6.66% |
| shared expert | 2.98% | 3.50% |
| final TP gather | 2.17% | 2.55% |
| whole MoE attribution | 59.52% | 62.81% |

Thus dispatch and expert execution contain enough mass to justify Stage 1;
shared-expert overlap cannot independently pass the 5% kill gate even under a
perfect implementation.

## Correctness

All fresh clean restarts and all diagnostic-replay generations have identical
ordered answer hashes within dataset. The bounded baseline scores inherited from
the already validated harness are 5/32 GSM8K and 6/32 HumanEval; the short
generation budget makes these unsuitable as model-quality claims, but exact
identity is a strong intervention check. The overlap diagnostic runs after the
normal layer output, so model semantics, NFE, and accepted-token behavior are
unchanged.

Raw data: `BASELINE_CLEAN.csv`, `QUALITY_CORRECTNESS.csv`, and the timestamped
result root.
