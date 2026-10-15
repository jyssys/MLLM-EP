# Looped Transformer multi-GPU KV discovery

**Decision:** `NO_GO_NATIVE_TP_SUFFICIENT` within the tested Ouro regimes.
**Coverage:** `MODEL_B_TP_BLOCKED`, `PARTIAL_MODEL_COVERAGE`.

The real Ouro checkpoint runs on physical GPUs 4–7 with native TP1, TP2, and TP4.
Huginn runs as an official single-GPU reference. Its native TP2/TP4 path is unsupported.
No new KV placement method or candidate speedup is claimed.

Start with [report.md](report.md).
Use [reproduce.md](reproduce.md) for fresh-output reproduction commands.
Read [timing_contract.md](timing_contract.md) before interpreting latency fields.

Key evidence:

- [Native request statistics](results/native_summary.json)
- [Actual cache and weight ownership](results/tp_ownership_proof.json)
- [Four-GPU finite-cohort throughput](results/cohort_throughput.csv)
- [Observer validity](results/observer_validity.json)
- [Correctness scope](results/correctness.json)
- [Conceptual ownership and transfer ledger](results/oracle_costs.json)
- [Gate decision](results/verdict_gates.json)
- [Final audit](results/final_audit.json)

Raw evidence remains in `results/raw/`.
Pretrained weights and live burner logs remain outside Git.
The GPU guard restores scoped burners after each experiment.
Seven CPU tests check accounting, immutable inputs, and shared-module observation.
