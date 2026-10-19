# v4 storage provenance: checkpoint 3

Seven completed jobs pass the graph and cache-root checks.
The [machine checkpoint](../artifacts/storage_provenance_v4_checkpoint3.json) records the 2026-10-19 13:32:16 UTC queue snapshot.
The queue still runs.
Base2.6B P4 alias remains pending.

This checkpoint adds Base2.6B P4 copies to [checkpoint 2](storage_provenance_v4_checkpoint2.md).
All four ranks have 1538 top-level weight operands.
Each depth has 384 block operands.
The embedding and recurrent normalization contribute the other two operands.

The new job passes the layer, role, BF16-shape, and source-use checks.
Its AOT and ordinary compile paths remain inside its job directory.
Its forced Inductor root remains below its job-local AOT root.
Its Triton group children remain inside its job-local Triton root.
The [job export](../analysis/storage_provenance_v4/storage_copies_base26_P4_fully_isolated_v4/manifest.json) retains host-graph gzip files and hashes.

The seven jobs have the expected compiled operand distinction.
No runtime binding capture occurred in this audit.
These counts do not prove numerical parity, cached decode-kernel addresses, or a timing effect.
The parent agent must evaluate these claims separately.

The source and check procedure remain those in checkpoint 2.
The standalone binding runner and metadata observer pass 19 CPU tests.
The combined test run also includes six graph-validator tests.
The combined result is 25/25 passing CPU tests.

This addendum extends the combined audit segment to 2026-10-19 13:32:18 UTC.
The segment started at 2026-10-19 13:16:12 UTC.
The total elapsed wall time is 966 seconds.
This is one combined interval, not an additional 966-second interval.
No GPU execution occurred in this segment.
