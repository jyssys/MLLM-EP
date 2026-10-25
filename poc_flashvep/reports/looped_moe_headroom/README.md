# Looped MoE expert-efficiency headroom

Final decision: **NO_GO_HEADROOM for the tested native B1 candidate grid**.
P3 optimization did not run.
No measured candidate E2E speedup is claimed.

| Model | Best tested NLL-eligible arm | Omitted work | ΔNLL [95% CI], nats/token | Optimistic timing model | Achieved E2E speedup | P2 / P3 |
| --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | O1_drop_0.2_renorm0 | 16.84% | 0.0136 [0.0112, 0.0159] | 1.0462× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |
| IFM MoE Huginn | O1_drop_0.3_renorm0 | 19.06% | 0.0133 [0.0109, 0.0156] | 1.0525× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |
| Foil-1 | O1_drop_0.2_renorm0 | 9.16% | 0.0122 [0.0072, 0.0175] | 1.0151× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |


The timing column is an analytical estimate, not an achieved speedup.
It uses prefix-512 expert event spans and realized omitted rows.
It assumes proportional time removal and zero bookkeeping cost.
The NLL frontier does not establish downstream accuracy equivalence.

All three specified pretrained BF16 checkpoints ran on actual single GPUs.
The IFM baseline retains a backend-fidelity qualification.
Both official generation APIs and the adapters produced the same eight-token smoke output.
The strict FP32 attention diagnostic produced a token difference on another prompt.

Read [report.md](report.md) for Tables A–D and model-specific findings.
Read [go_no_go.md](go_no_go.md) for the decision and [reproduce.md](reproduce.md) for commands.
[validation.json](validation.json) records integrity checks.
[run_manifest.json](run_manifest.json) links runs to versions, hardware, and source hashes.

Raw data and failed runs are preserved in lossless [archives/](archives/).
The archive manifest records file-level SHA-256 checks.
Local originals remain in raw/ and logs/.
Checkpoint weights and full language passages are not committed.
