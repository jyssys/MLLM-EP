# Collective-pair probe: no admitted comparison

## Outcome

Neither P2 run qualifies for the comparison table.
Both complete their six per-shape correctness checks and timing cells.
Neither completes error-free worker teardown and the outer guard.
P4 is NOT_RUN under the declared stopping rule.

The first run uses the frozen [v1 source](../artifacts/hybrid_collective_pair_probe_v1.py).
Its workers remain alive without local cleanup records after the final timing cell.
The coordinator terminates only the verified owned launcher.
The [decision record](../artifacts/hybrid_collective_pair_P2_v1_abort_decision.json) preserves the target and timestamp.
The guard then records failure and scoped burner restoration.

The separate teardown-only retry preserves the complete measured-body AST.
It adds diagnostic markers and changes cleanup order.
Its first marker identifies `destroy_NCCL_group`.
The one-shot Python stack remains inside `destroy_process_group` after 30 seconds.
The 300-second outer limit then terminates the owned job.
Its guard also restores the scoped burners.

This observation locates the blocked Python-to-native call in the retry.
It does not establish the native root cause or the first run's blocked call.
It does not measure a model's decode stall.

## Evidence

- [P2 v1 raw directory](../outputs/hybrid_collective_pair_P2_v1/)
- [P2 retry raw directory](../outputs/hybrid_collective_pair_P2_v2_teardown/)
- [Retry rank-0 cleanup marker](../outputs/hybrid_collective_pair_P2_v2_teardown/cleanup_progress_rank0.json)
- [Retry rank-0 stack](../outputs/hybrid_collective_pair_P2_v2_teardown/cleanup_stack_rank0.txt)
- [Retry guard status](../outputs/hybrid_collective_pair_P2_v2_teardown/guard_status.json)
- [Independent cleanup audit](hybrid_collective_pair_cleanup_note_v1.md)
- [Teardown review](hybrid_collective_pair_teardown_retry_review_v1.md)

## Interpretation

The raw timings remain diagnostic evidence only.
They do not establish an admitted custom-AR versus AG/RS service ratio.
The result adds no full-layer, candidate E2E, quality, or HBM-traffic evidence.
It does not reject every generic attention/FFN hybrid.

The earlier remote-full-MLP fixture remains a separate measured implementation and scope.
Its result does not substitute for this missing AG/RS comparison.
The current ownership contract therefore retains its physical-cost uncertainty.
No additional teardown retry or model-engine implementation follows in this campaign.
