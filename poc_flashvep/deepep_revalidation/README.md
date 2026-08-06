# FlashVEP DeepEP overlap revalidation

This directory contains the bounded vLLM 0.20 / DeepEP revalidation PoC. It
uses only physical GPUs 4,5,6,7 and keeps the exact DeepEP build in a separate
virtual environment.

Run order:

```bash
./install_deepep_env.sh RESULT_DIR
./run_deepep_smoke.sh RESULT_DIR
./run_vllm_backend_matrix.sh RESULT_DIR
./run_operator_replay.sh RESULT_DIR
./run_nsight_best.sh RESULT_DIR
python3 analyze_results.py RESULT_DIR
```

The operator replay reuses the captured layer-24 workload and the model-loaded
Triton expert weights. DeepEP dispatch and combine are invoked directly to
remove scheduler effects. Reduced expert outputs are explicitly double-buffered
per microbatch because the vLLM workspace manager may otherwise alias E(next)
with the input consumed by C(previous).

Large `.nsys-rep` and exported SQLite files stay under `large_local_artifacts`
and must not be committed. Only compact summaries and the report belong in Git.
