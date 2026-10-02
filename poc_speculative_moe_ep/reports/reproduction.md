# Reproduction

Measured clean TRUE EP4 on GPUs2,3,6,7, output cap64/EOS, first16 frozen requests (8+8). Route features come from a separate trace pass with identical request/prefix/draft IDs.

See ../README.md for exact audit/launch/campaign/analysis commands, environment paths, append-only names and watchdog/cleanup. Full local logs live in raw/, which is ignored by git. Compact exported raw numerical events, actual route arrays, owner proofs, outputs and hash manifests are under artifacts/compact_*. Model weights and answer keys are excluded.

```bash
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.physical_validation --version my_clean_analysis phase_b_clean_v1_g0 phase_b_clean_v1_g2 phase_b_clean_v1_g4 phase_b_clean_v1_g8
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.verify_bundle --evidence compact_phase_a_v1
/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.offline_reproduce --version phase_b_v2
/home/esjung/anaconda3/bin/python -m pytest -q poc_speculative_moe_ep/tests
```

Resume completed campaign IDs skips successful immutable arms; failed IDs require a new prefix. Main unrelated staged/dirty diff hashes are preserved in repository_safety.json. An instrumented trace fit is never promoted as a clean headline. Final burner mask must be explicitly2,3,6,7 because run_utilize.sh currently defaults to different GPUs.

Canonical per-invocation ledger is `phase_b_v2/verification_metrics_draft_aligned.parquet`: previous-frame proposer time is associated with the current verification; current-frame proposer time is named `next_draft_ms`. Earlier raw tables are retained and explicitly superseded for this column only. All EP/target labels, route features and grouped-CV/oracle results are unchanged. The offline command reproduces every MAE/R2 and fold from the published ledger without a GPU or local raw directory.
