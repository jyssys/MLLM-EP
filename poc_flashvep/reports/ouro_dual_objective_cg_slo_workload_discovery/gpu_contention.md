# Physical devices and interference controls

Only physical GPUs4–7 execute this campaign.
Each process sees one authorized device as logical CUDA0 and checks its actual UUID.
Each worker uses a separate CUDA context, cache paths, log, run ID, and CPU affinity block.
OMP, MKL, and OpenBLAS use one thread.
No distributed process group forms.
This is independent single-GPU experiment parallelism, not TP/PP serving.

The new calibration completed 24 model-process starts and 144 measured cohorts.
It compares 1/2/4 workers on both approximately 2K/N4 and 8K/N8 contexts.
Each device/count cell has two independent starts and three repeated cohorts per context.
The same-GPU/start throughput drift ranges from -2.114% to +2.293%.
No observed cell crosses the registered 3% review threshold.
Two starts do not prove universal interference neutrality.
The validation lane can use four workers. Decisive performance confirmation will use one measurement worker.

GPUs0–3 had pre-existing owned burners at the start.
The authorization prohibits their use in this study, so this campaign does not stop or restart them.
Their shared-host effects remain part of the fixed background condition.
A single measurement worker here does not mean an eight-device quiescent host.

The resource wrapper checks UID, burner session ownership, approved logs, and active CUDA descendants before signaling.
It signals only authorized GPU4–7 burner sessions.
Unrelated processes cause a refusal rather than a forcible takeover.
The dispatcher restores owned burners when a worker queue finishes or exits.
Isolation-reserved devices remain free of burners only during declared controls or clean confirmation.

Data: `contention_raw.csv`, `contention_paired_drift.csv`, `controls_summary.json`, and raw telemetry.
