# Active-support / MLP-boundary discovery PoC

Study date: 2026-09-15 (KST). Scope: LLaDA2.0-Flash BF16 routed MoE,
64 resident experts per owner rank, hidden 4096 / intermediate 1024,
four physical H100s 4--7. **No new CUDA kernel has been implemented.**

The branch is independent of the previous RefineGEMM PoC. It references its
frozen exact EP4 route corpus and clean full-request control but does not
rewrite existing experiment files.

- [Main report](reports/final_decision.md)
- [GPU/source runtime audit](reports/00_environment_runtime.md)
- [Controlled support and metadata](reports/01_support_metadata.md)
- [SwiGLU/MLP boundaries and Nsight](reports/02_boundary_nsys.md)
- [Real routes and request oracles](reports/03_replay_oracles.md)
- [Backend/prior-art attack](reports/04_prior_art.md)
- [SYNTHETIC_CONTROLS.csv](SYNTHETIC_CONTROLS.csv) and
  [REAL_REFINEMENT_MAPPING.csv](REAL_REFINEMENT_MAPPING.csv)
- [REQUEST_ORACLES.csv](REQUEST_ORACLES.csv)
- [PHASE_COST_SUMMARY.csv](PHASE_COST_SUMMARY.csv)
- [METADATA_SUMMARY.csv](METADATA_SUMMARY.csv) and
  [MATERIALIZATION_SENSITIVITY.csv](MATERIALIZATION_SENSITIVITY.csv)
- [NSIGHT_KERNELS.csv](NSIGHT_KERNELS.csv): persisted extracted GPU kernel
  stats; large binary profiles remain local per existing gitignore.

GPU measurements must **not** run with the utilization burn. The benchmark
checks `CUDA_VISIBLE_DEVICES=4,5,6,7` and the physical UUID before using
logical cuda:0 = physical GPU 4. Every launch also needs a read-only
`nvidia-smi -i 4,5,6,7` process ownership and available-memory audit.
All operator experiments are owner-local replay, not an integrated EP4
expert-backend replacement or measured end-to-end speedup. `compacted_all`
is a future-known live-row sensitivity, not a measured Epoch baseline.

Run CPU-only reproducibility checks:
`python poc_active_support_boundary/scripts/analyze_support_boundary.py`
and `python -m unittest discover -s poc_active_support_boundary/tests -v`.
