# Evidence coverage and remaining gaps

The audit matched eleven required historical files against commit `c458f6a2a764f667a0da6784bcb2592b36626837`.
It also checked all 375 lossless archives against their recorded SHA256 values and server originals.
Historical request-level token timestamps are available.
Thus absolute request-max-ITL compliance can be recomputed without a new GPU run.
Historical cold-order and warm-up lanes remain separate from held-out confirmation.

## New evidence collected

| Earlier gap | Current coverage | Remaining limit |
| --- | --- | --- |
| Decode-only calibration | Solo GPU4 and four-worker calibration. Fixed SLO registration | One paired device episode cannot identify a universal interference correction |
| Counterbalanced C/G controls | 316 continuity cohorts. Ten shortlist cohorts across six process blocks per primary shape | Other full-grid settings retain three cohorts |
| GPU quantum surface | Separate CUDA-event profiles across C128/256/512/1024 and G1/2/4, plus Mixed | Few profile processes per cell. Not a validated online cost predictor |
| GPU timeline | Four successful Nsight captures and portable CSV exports | Heavy timing is not clean E2E critical-path attribution |
| Natural serving | 128 Poisson and 48 burst policy runs with common arrivals | Per-arm/rate requests remain below 1,000. Caps and two timeouts remain visible |
| Representative prompts | Frozen conversation proxies and genuine article excerpts | Not ShareGPT or a mathematical accuracy benchmark |
| Observer validity | Twelve paired cohorts preserved tokens/depth exactly. Mean time ratio 1.002225, 95% CI [0.997524, 1.007551] | Covers Atomic C1024/P≈2K/N4 only |
| Device/control fairness | All policies share their GPU/process block. A final GPU4 quiet policy control is available | Some rates have unbalanced device allocation. Quiet and concurrent runs occur at different times |

## Unresolved questions

- The near-tie LM-head diagnostic does not explain every historical hidden-state divergence.
- New bounded batched output parity is not universal despite 48 exact isolated probes.
- Full GSM8K/MATH accuracy remains NOT_RUN.
- Complete-response cap sensitivity above 512 tokens remains NOT_RUN.
- Qualified sustainable capacity and qualified request-tail P99 remain unproved.
- Split C1024 and Mixed C1024 lack open-loop rate sweeps.
- Live adaptive C-only, G-only, and Joint controllers remain NOT_RUN.
- Physical HBM traffic, achieved GEMM utilization, and isolated graph-workspace bytes remain unmeasured.
- Multi-GPU serving portability remains NOT_RUN.
- This task does not add per-request TP/PP support.

The old discovery grid is exploratory because it contains one cohort per cell.
The old repeated lane contains ten cohorts per shortlisted configuration across three process starts.
Its fixed Atomic-first order remains a confound.
The previous `CONDITIONAL_FIXED_QUANTUM_ONLY` decision stays unchanged.

Independent workers on physical GPUs4–7 accelerate this study.
They do not fill the missing multi-GPU serving-portability evidence by themselves.
