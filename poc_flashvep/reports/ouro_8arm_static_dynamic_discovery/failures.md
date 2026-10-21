# Failure ledger

## Historical strict TP4 gate

The previous P512 strict gate remains FAIL. This study does not change its thresholds.
The old operator audit passed 192/192 operations. That result does not certify full-model accuracy.
The new study permits labeled exploratory task evaluation after finite-output and ownership checks.

## Burner startup race

`numerical_native_TP1_v1` encountered insufficient free memory during burner startup.
The launcher now waits for burner worker creation and checks scoped process ownership.
The failed log remains available. The successful retry is `numerical_native_TP1_v2`.
The operator did not terminate any unrelated GPU process.

## Initial task TTFT observer

The native offline frontend emitted only final request results.
Therefore, `native_TP1_pilot_v1` has invalid TTFT and TPOT fields.
The recorded output IDs, answer scores, and total completion times remain usable.
The v2 frontend records cumulative token publications without changing sampling settings.
Separate v1/v2 accuracy and token-parity rows remain available.
The two native TP1 runs differ on 2/50 MATH outputs and 0/64 GSM8K outputs.
This observation does not establish the cause of those two differences.

## Self-IPC initialization

`probe_S4_v1` failed while reconstructing its own CUDA buffer through a multiprocessing queue.
The new runtime directly references its own buffers. It imports only other ranks' buffers through CUDA IPC.
`probe_S4_v2` passed initialization. Later checks exposed a separate Graph error.

## Graph input lifetime

The initial masked Graph path did not retain an arange tensor allocated before capture.
The Graph mask kernel still referenced that tensor during later replays.
Memory reuse corrupted the mask and changed outputs. Duplicate input probes exposed this error.
`graph_diagnostic_v1` separates masked eager execution from masked Graph execution on identical teacher-forced inputs.
The old Graph produced large additional errors. The eager path did not reproduce that additional error.
The fixed runtime retains the tensor for the lifetime of each Graph.
`graph_diagnostic_v2` passes bitwise Graph-versus-eager logits at all eight aligned steps.
This new implementation error does not explain the historical TP4 strict-gate failure.
All pre-fix probe outputs remain diagnostic failures. They are not benchmark accuracy samples.

## TP Graph teardown

`probe_T4_graph_v1` produced bounded outputs but did not exit normally after communicator teardown.
The operator terminated only this identified project probe after checking that all four GPUs were idle.
The fixed runtime destroys Graph references before the NCCL process group.
`probe_T4_graph_v2` exits normally with code zero. Its four ranks complete the same bounded probe.
The first probe's raw completion marker does not certify process completion.
The campaign now also requires a successful launcher completion record.

## Protected concurrent job

One attempted quality launch overlapped the preceding probe's teardown.
The GPU lease guard rejected that launch. It did not terminate the preceding GPU processes.
The quality campaign restarted after the probe completed. No answer samples came from the rejected launch.

## Tokenizer-library input mismatch

The checkpoint tokenizer files were identical, but two Transformers versions encoded repeated spaces differently.
At GSM8K question 45, native input used 357 tokens. The first custom run used 356 tokens.
Both inputs had the same prompt text SHA256. Text identity did not prove token identity.
The operator stopped the custom quality run when the input audit exposed this mismatch.
It did not stop any other user's job. The partial raw outputs remain available.

`frozen_prompt_ids.json` fixes explicit input IDs for all 1,819 questions.
The canonical tokenizer uses the native frontend's Transformers version.
All 342 completed native task records exactly match these canonical IDs.
Every subsequent runtime receives those IDs directly. It does not tokenize the prompt again.
The custom Single run resumes only missing or input-mismatched complete C4 cohorts.
`effective_source_manifest.json` records the source of each reused or repaired cohort.
The reuse rule depends on input identity and completion, never answer correctness or latency.
The old mismatched row cannot enter a paired cross-arm comparison.
# Dynamic skipped-owner KV length metadata

A new CPU assertion reproduced a stale valid-length field after an exit-row import.
The copied K/V tensors contained the new row, but the skipped-loop owner's cache length did not advance.
This stale field could truncate that owner's diagnostic snapshot.
The import now advances the valid length to include the copied row.
The regression test still checks current-row identity and unchanged historical rows.
This correction precedes the first Dynamic GPU result. It does not change the running Static campaign.

