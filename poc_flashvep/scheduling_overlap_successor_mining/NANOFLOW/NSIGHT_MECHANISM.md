# Native split mechanism, not a performance headline

Fresh CUDA/NVTX-only profiles: `nanoflow_runs/nsight_resume_20260909_v1/`.
One unsplit and one FFN-split2 graph engine, B16/C512, same real text cohort.
No global GPU counters, CPU sampling, or unrelated-process sampling.
Analysis excludes prefill and first graph capture; 30 steady target steps per
device, 120 same-device observations per plan. CUDA intervals are unioned, not
summed across GPUs. CPU NVTX marks associate launches with the request step.

| Median metric | unsplit | split2 |
|---|---:|---:|
| Profiled request step, ms | 9.262 | 12.701 |
| Kernel interval union, ms | 6.026 | 8.542 |
| Non-NCCL kernel union, ms | 5.397 | 7.871 |
| Resident NCCL kernel union, ms | 0.627 | 2.399 |
| Compute/NCCL intersection, ms | 0 | 1.627 |
| NCCL intersection fraction | 0 | 0.700 |
| Kernel count | 702 | 1,254 |

The native operation splitter creates real overlap but increases launches and
resident collective duration. Resident NCCL time includes waiting/progress and
is not bytes-active network time. Split2 is slower in this diagnostic. Profiled
times are excluded from clean E2E comparisons. This is a manual supported plan,
not the paper's auto-searched optimum; small-batch overhead is already known.

Nsight kernel device IDs 0–3 are process-visible ordinals under the enforced
`CUDA_VISIBLE_DEVICES=4,5,6,7`. The exported GPU inventory separately lists all
physical GPUs and must not be joined to kernel IDs as if they were physical IDs.
CUDA-graph node virtual stream IDs do not count actual source-level streams.
