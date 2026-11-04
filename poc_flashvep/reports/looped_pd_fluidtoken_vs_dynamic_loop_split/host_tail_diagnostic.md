# Residual Huginn Decode tails

## Scope

These diagnostic runs use Huginn R32, BF16, 2048 input tokens, and 16 output tokens.
They retain the same 512 prompts, arrival seed, graph geometry, native transport, and D-local probes.
The frozen ITL p99 limit is 192.821253818 ms.
The observers record host intervals longer than 20 ms. These runs are not clean performance comparisons.
Nested intervals overlap. Their durations must not be added as exclusive E2E components.

## Measured observations

| Run suffix | ITL p99 (ms) | Maximum ITL (ms) | Output parity | Interpretation |
|---|---:|---:|---|---|
| `graph4_host_pause_diagnostic512` | 194.652082697 | 642.851889133 | Exact | Whole-run ITL gate failed |
| `graph4_ipc_pause_diagnostic512` | 191.695302166 | 535.045262426 | Exact | One diagnostic pass does not establish repeatability |

Both runs completed all 512 background requests. Their probe checks retained exact outputs and full recurrent work.
The first run recorded a 642.852 ms token interval for sample 147.
Sample 148 had a nearby 640.099 ms interval. Neither interval overlapped a Prefill probe.

The first run's largest measured D garbage-collection interval was 0.959 ms.
The coordinator's largest measured interval was 2.434 ms.
Larger process-lifetime collection intervals occurred during startup, outside the measured window.
These measurements do not support garbage collection as the main cause of the large Decode tails.

The second run recorded no IPC send or receive call above the 20 ms observation threshold.
It also recorded no NIXL submission, NIXL polling, connector progress, or memory query above that threshold.
This observation does not exclude shorter delays or unobserved execution paths.

The coordinator still had long message handlers:

| Message kind | Recorded calls above 20 ms | Maximum enclosure (ms) |
|---|---:|---:|
| reserved | 2 | 352.238092571 |
| prefill_complete | 11 | 247.288489714 |
| released | 2 | 189.015749842 |
| commit | 3 | 99.302798510 |
| activated | 1 | 92.220731080 |

## Next falsifiable hypothesis

The campaign-specific `append` functions use synchronous file open, serialization, write, and close.
The existing durable writer redirects another function identity. Its alias audit does not include these campaign-specific functions.
Thus, `async_evidence=True` does not prove that every measurement callback avoids foreground file I/O.

The next diagnostic measures these exact `append` and `write` calls.
If their intervals explain the long message handlers, a matched asynchronous-routing control will test the hypothesis.
The control preserves every record and snapshots JSON before enqueue.
It reuses the existing durable background writer and its terminal flush.
It does not change CUDA kernels, model math, NIXL calls, or the reservation protocol.

CPU tests check captured aliases, strict JSON, record order, immutable snapshots, and complete flush.
GPU state and output qualification must pass before serving measurements expand.
The candidate must also pass fresh whole-run SLO checks and same-window GPU gates.

## Evidence

See the two named namespaces under `raw/`.
`host_pauses_*.json` and `python_gc_*.json` contain the measured host intervals.
`ipc_pause_analysis.json` summarizes the second run.
The state files preserve the largest token intervals and their temporal overlap with host calls.

Current root-cause status: **ROOT_CAUSE_INCONCLUSIVE**.
Temporal overlap alone does not establish causation. No IPC or garbage-collection fix is claimed.

## Targeted evidence-I/O result

`recovery_huginn_graph4_evidence_io_diagnostic512` completed all 512 requests.
Its ITL p99 was 200.100804418 ms. Maximum ITL was 433.382026851 ms.
TPOT p95 was 137.477172383 ms.

The observer found long calls in the campaign-specific append functions:

| Owner | Calls above 20 ms | Maximum append enclosure (ms) | File at the maximum |
|---|---:|---:|---|
| P worker | 35 | 286.899460480 | `P_submission_readiness_1713311.jsonl` |
| D worker | 36 | 271.818818524 | `memory_progress_1713312.jsonl` |
| Coordinator | 23 | 299.750117585 | `lifetime_events.jsonl` |

These three maximum intervals ended within 0.047 ms of each other.
Their starts were 8416533.447302993, 8416533.462391399, and 8416533.434498416 on the host monotonic clock.
The coordinator's enclosing handler lasted 299.857007340 ms.
The append call accounts for approximately 99.96% of that host enclosure.

This is direct evidence that foreground evidence callbacks can block serving progress.
It does not identify the exact filesystem or kernel lock responsible for the shared pause.
It also does not explain the separate NIXL/CUPTI stall by itself.

The matched control routes these aliases through the existing asynchronous durable writer.
The control remains provisional until exact-state checks, clean latency, and independent joint gates pass.
The historical raw logs and failed verdicts remain unchanged.
