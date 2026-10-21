# Next actions

1. Complete the remaining custom T4 actual-generation pilot. S4/L4/O4 completed 114 canonical requests each.
2. Run the official CDB smoke with current-row KV-copy checks.
3. Compare forced-exit and live-gate SD, LD, and OD probes.
4. Check the TD Graph path and its actual compute skipping.
5. Complete the same Dynamic pilot questions for each valid arm.
6. Complete the user-approved 400-question target per valid arm. Retain the pilot and execute only its 286 additional questions.
7. Execute matched-natural and frozen replay plans. Use five clean repeats per selected replay shape.
8. Capture separate four-GPU timelines and derive communication and idle records.
9. Produce paired accuracy, serving tables, and all eight arm-status rows.
10. Check protected source hashes. Commit only the new study files. Push the new branch.
11. Review the completed report snapshot and write the final evidence-based decision. Automatic execution alone does not issue a verdict.

Use `python -m poc_flashvep.ouro_8arm.campaign --stage static-quality` for the current stage.
Use the interpreter specified in `commands/`. Successful completed runs are not repeated.
Do not start a second GPU campaign while the first campaign still owns the devices.
Restore authorized burners after all experiment processes exit.

The current continuation controller waits for the Static quality campaign to exit.
Its metadata is `state/continuation_controller.json`. Do not launch a competing GPU campaign.
The continuation includes Dynamic probes, Dynamic pilots, the 400-question extension, and performance lanes.
The complete 1,819-question suite is no longer the requested target.

The original Static campaign PID is 1191699. Verify its command and ownership before any process action.
The newest continuation PID is in `state/continuation_controller.json`.
Capacity interruptions in chat did not stop the GPU campaign at this checkpoint.
Do not start a duplicate controller or repeat a successful pilot.
After the current campaign exits, the continuation validates Dynamic semantics before new Dynamic task runs.

Review preparation-inclusive timing as well as prepared-service timing.
The prototype currently rebuilds request-bound graphs per cohort.
An apparent serving-only speedup does not prove a cold-cohort latency improvement.
The heavy lane must validate CUDA logical-device IDs against the launch mask and recorded UUIDs.
It must not use `TARGET_INFO_GPU.id` as the CUDA-visible rank.
