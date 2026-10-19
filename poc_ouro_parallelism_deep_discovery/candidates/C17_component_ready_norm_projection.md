# C17: Component-ready norm/projection contraction

| Required field | Contract |
|---|---|
| 1. Identity | Revision 1. Promotes provisional R3-A to a stable dossier identity. |
| 2. Mechanism | Contract fresh completed feature tiles before the final RMS scalar completes. |
| 3. Ouro property | Repeated physical operators and frequent norm boundaries. Neither implies equal activation inputs. |
| 4. Target | Both Base sizes, fixed R4/original adaptive, C1/2/4/8, P2/P4, projected P8. |
| 5. Baseline | Strong native TP plus generic norm/projection fusion under the same numerical contract. |
| 6. Sequence | Completed raw tiles feed projection partials and norm statistics, then the final scalar joins them. |
| 7. DAG | Every tile waits for its actual producer. Final output waits for all required reductions. |
| 8. Concurrency | Fresh tile products can overlap norm statistics. Future recurrent inputs remain unavailable. |
| 9. Exactness | Real-arithmetic identity. The simple BF16 rewrite fails a constructed equality test. Model quality NOT_RUN. |
| 10. Memory | Original KV plus partial accumulators and tile publication state. No free whole-core residency. |
| 11. Messages | Feature transfer, norm reduction, and output reduction depend on the actual layout. |
| 12. Critical path | Potential norm wait removed. Tile control, joins, reductions, and occupancy costs remain unmeasured. |
| 13. Prior | FlashNorm and generic tile dataflow directly overlap the mechanism. |
| 14. Prior failure | Unlike C06, this uses one tiled fresh contraction. It still must preserve required BF16 boundaries. |
| 15. Falsifier | The CPU cancellation example rejects universal BF16 identity. Primary prior rejects the current novelty claim. |
| 16. Evidence | Source and CPU only. See the full R3-A contract in the round3 report. |
| 17. Compatibility | Shapes apply to both sizes. Adaptive additionally retains the original hazard and KV policy. |
| 18. Counterfactual | R1 and an alias-aware ordinary unroll expose the same local operator opportunity. |
| 19. Complexity | No distributed implementation is justified by this screen. |
| 20. Status | GENERIC_ONLY. No candidate E2E or free-running quality result. |

Evidence: `reports/candidate_round3_independent_axes.md` and `reports/tile_dataflow_prior_audit.md`.
The original initial-candidate freeze remains unchanged.
