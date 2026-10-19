# Both-size and both-mode baseline extension

## Purpose and limits

The earlier calibration contains eight requests per domain.
Its wide quality intervals cannot characterize the original gate's request-level work variation well.
This extension collects more natural baseline trajectories before final cost synthesis.
It does not select a candidate or change a gate.
It does not test a new architecture.

The new protocol selects 32 requests per domain from existing local request metadata.
All 80 original campaign IDs remain excluded.
This includes the 64 frozen candidate held-out IDs.
Earlier pilot and protected factorial request IDs also remain excluded.
The selector reads protected request metadata, not protected results.
Selection uses no model outcome, generated length, accuracy, or policy gain.
The original protocol and its outputs remain unchanged.
The inherited source split and this campaign's partition are different fields.
The new requests belong only to baseline_extension here.

## Frozen execution

Both pinned Base checkpoints use the existing staged reference.
Each model runs fixed R4 and original-gate adaptive mode on one eligible GPU per arm.
Every arm uses BF16, batch 1, greedy natural generation, EOS, and the 8192-token cap.
Adaptive mode retains q=0.5, minimum R2, maximum R4, full-depth prefill, and last_exited KV.
The existing loader checks active gate tensor identity against the checkpoint.
Per-pass CUDA Graphs remain enabled, as in the earlier calibration.
This is not another graph-ON comparison.

The existing exit callback records actual decode depths.
The first output token comes from full R4 prefill and has no decode-exit record.
Each generated trajectory must contain exactly one fewer depth record than output tokens.
The script changes no attention, weight, threshold, sampling, or scheduling rule.

The four independent jobs can execute concurrently on physical GPUs 4/5/6/7.
Concurrent job durations do not multiply campaign elapsed time.
Each guarded job restores its owned burner after completion or failure.
No unknown process may receive a signal.

## Required analysis

Report `correct/N`, accuracy, parsing failures, EOS, stops, and truncation per domain and arm.
Use request-level bootstrap intervals.
Report adaptive-minus-fixed paired outcomes on the same new IDs.
That difference describes operating points, not a new method's quality cost.
Do not claim noninferiority from this small extension.

Report request-level mean depth, depth variance, actual loop work, and depth frequencies.
Keep token-weighted summaries separate from request-weighted summaries.
These trajectories do not reconstruct simultaneous CDB membership or cross-GPU ready frontiers.
No future depth may enter a causal scheduling claim.
Independent one-GPU request walls do not establish architecture speedup or clean component attribution.

Only completed arms enter benchmark summary rows.
Partial outputs and failures remain available with explicit status.
Do not change the token cap or tolerance after outcomes.
No original held-out request executes in this extension.

## Reproduction

Run scripts/freeze_baseline_extension.py once with a new artifact name.
Use scripts/baseline_extension.py under the existing scoped GPU guard.
Use a unique output directory for every arm or failed attempt.
Raw request files use lossless gzip JSON to limit storage.
Their compressed form changes no recorded token, text, or depth event.
