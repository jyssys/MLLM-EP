# What the resident-KV operator does not measure

The operator starts with valid KV already resident on its owner rank.
This is the same resident-input contract as the local attention control.
It does not measure initial prefill distribution or growing-cache maintenance.

A full model must deliver each newly projected K/V value to its persistent KV owner.
At B1, that new K+V payload is 8,192 bytes per physical layer and recurrence depth.
Repeating the shape over 96 namespaces gives 786,432 bytes per generated token.
These are structural byte counts, not measured live communication.
The implementation could combine Q and current K/V into one input packet.
It would still need correct remote writes, context lengths, and completion dependencies.
The current operator broadcasts Q only.

A persistent shard map must also handle request growth.
Recomputing balanced boundaries after each token would require ownership changes.
An implementation can instead append to a persistent owner or use cyclic pages.
Either choice requires a measured imbalance and metadata analysis.
The standalone fixtures do not validate that serving policy.

Full-model prefill cannot first retain an impossible full cache on one owner and then claim distributed capacity.
Capacity extension requires a legal incremental placement path.
No such integrated path was built here.

The 96-fold operator envelope therefore omits important integration costs.
It must not be reported as measured E2E latency.
All valid tested B1 contexts fit one H100.
Batch-capacity extension must also beat independent DP replicas with the same GPU count.
The study does not prove that comparison.
