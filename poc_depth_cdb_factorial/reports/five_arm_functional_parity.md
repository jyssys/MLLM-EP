# Five-arm functional validation

## Probe

The probe uses the same eight frozen discovery requests in each configuration.
It permits natural EOS and caps output at 16 tokens.
This heavy diagnostic disables CUDA graphs and checks terminal KV copies directly.
It is not a headline latency or benchmark-quality measurement.

The functional A0 probe loads idle peer workers for control-frame consistency.
Only physical GPU4 processes its requests.
All reported A0 performance runs instead use physical GPU4 alone.

## Configuration and physical execution matrix

| Arm | Actual request GPUs | Gate / KV | Local refill | Hindsight placement | Actual loop work / fixed-R4 equivalent | Exact KV checks |
|---|---|---|---|---|---:|---:|
| A0 | 4 | q=.5, R2–R4 / `last_exited` | Off | Off | 394 / 480 | 360 |
| A1 | 4/5/6/7 | q=.5, R2–R4 / `last_exited` | Off | Off | 394 / 480 | 1368 |
| A2 | 4/5/6/7 | q=.5, R2–R4 / `last_exited` | Off | On | 394 / 480 | 1344 |
| A3 | 4/5/6/7 | q=.5, R2–R4 / `last_exited` | On | Off | 394 / 480 | 1488 |
| A4 | 4/5/6/7 | q=.5, R2–R4 / `last_exited` | On | On | 394 / 480 | 1848 |

All 6408 terminal KV checks passed with exact tensor equality.
Each check compares copied K and V rows against the exiting depth's rows.
The actual decoder-layer call count matches the native prefill and recurrent counters in every arm.
The expected call count is `24 × (4 × prefill batches + recurrent batches)`.
The probe confirms actual early-exit work reduction of 17.9167%.
It does not merely select an output after executing fixed R4.

The no-refill counter is zero in A0, A1, and A2.
The refill counter is 67 in A3 and 70 in A4.
Its definition counts fresh tokens that join a nonempty recurrent work queue.
It does not count cross-GPU request moves.

## Output parity

All five configurations produce the same eight complete token sequences in this bounded probe.
All five produce the same eight exit-depth sequences.
The A0 comparisons and A1↔A2 and A3↔A4 comparisons each pass 8/8 token and depth equality.
`artifacts/functional_summary.json` contains the ordered output/depth checksums.

This short result does not prove long-answer parity under changed batching.
The speed and full-answer quality reports evaluate that limitation separately.
Those reports retain any first divergent token and changed EOS or output length.

## Source validity

The no-refill arms call the authors' `_run_no_refill_loop`.
The refill arms call the authors' `_run_scheduler_loop`.
The wrapper does not replace recurrent compute, gate decisions, or KV-copy semantics.
All arms share the same native model runner and precision.
See `source_scope_audit.md` for the exact scheduling differences.
