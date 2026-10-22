# Execution plan and GPU-worker amendment

The full English specification remains the experiment contract.
The latest user instruction changes resource use, not model parallelism or research gates.

## Independent workers

Physical GPUs4–7 each execute one independent single-GPU experiment process.
Each process uses one visible CUDA device, its own model, warm-up, logs, and result files.
The dispatcher assigns the next unfinished job to a free worker.
Completed job and case records prevent duplicate execution after resume.
Failure records remain intact. A failure does not count as a completed measurement.

Every policy block compares Atomic, Split, Grouped, and Mixed on the same physical GPU.
Policy order rotates across workers and independent process starts.
GPU identity remains a statistical blocking factor.
The analysis must not interpret different GPU assignments as policy effects.

Each worker has separate CUDA, Triton, and FA4 cache directories.
The jobs retain independent model contexts and warm-up records.
`GPU_policy_crossover.csv` records policy allocation by physical device.
`GPU_interference_calibration.csv` records the solo/concurrent control.
Future Nsight and observer controls execute outside clean serving jobs.

SLO threshold sweeps reuse the same measured request events on CPU.
They do not require duplicate GPU generation for each threshold.
Low-rate open-loop idle intervals remain part of the workload.
Burners or unrelated jobs must not fill those intervals on a measured device.

Only recognized user-owned burners can stop.
Unrelated processes must remain untouched.
A short quiet-device control compares GPU4 alone with four simultaneously active experiment workers.
The other devices can remain idle during this explicit interference control.
Afterward, the dispatcher keeps available workers supplied with jobs.
When no experiment remains, the recognized burners return on GPUs4–7.

## Stages

1. Audit the frozen previous files and lossless raw archives on CPU.
2. Read primary SLO definitions and record their exact denominators.
3. Run warmed decode-only and isolated-prefill calibration. Freeze SLO values and candidate arrivals.
4. Reuse historical C/G surfaces. Add counterbalanced fixed-grid and matched staging controls where evidence is insufficient.
5. Run open-loop Poisson and seeded bursts with all offered requests in the denominator.
6. Report queue stability, completion/cap/EOS, uncertainty, and unresolved output differences.
7. Decide characterization, Adaptive Joint headroom, and multi-GPU portability separately.

No adaptive controller starts before its discovery gates pass.
Independent concurrent workers do not constitute DP serving-capacity scaling.
The paused native-vLLM Loop PP program remains paused.

## Output location

The authoritative English specification names `ouro_chunk_loop_slo_sensitivity_discovery/`.
This directory is the primary artifact location.
The user-facing `_study/` directory contains an index to these artifacts.
Historical results and their `CONDITIONAL_FIXED_QUANTUM_ONLY` decision remain unchanged.
