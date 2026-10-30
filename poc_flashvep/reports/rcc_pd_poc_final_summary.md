# RCC-PD PoC final summary

## Decision

**Fixed-cut RCC execution: `ALREADY_PRESENT`.**
**New fixed-cut mechanism: `NO_GO_NOVELTY`.**
**Conditional admission and serving capacity: `NOT_RUN`, not an experimental NO-GO.**

Both models can co-batch imported continuation rows with decode rows at different logical depths.
Actual GPU matrix inputs and kernel traces confirm this execution.
However, existing Recurrence-Split A plus Native Mixed already provides that path.
This study adds a controlled harness and diagnostic evidence.
It does not add a new fused execution operator.

The English spec permits a Stage-1 stop when no additional fusion mechanism exists.
The study applies that stop rather than builds a larger admission or migration controller.
This decision does not prove that conditional admission cannot help.
Its benefit against strong DP_Mixed and generic borrowing controls remains untested.

## Correctness prerequisite and new evidence

The reused numerical release is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
Its branch is `flashvep/looped-pd-b-numerical-diagnosis`.
The remote SHA was checked before RCC GPU execution.
Both models passed that release's internal fixed-depth qualification.

Ouro retains Down and small-row RMSNorm corrections.
Huginn also retains its adapter correction.
The source checkout and model weights remain unchanged.

New RCC evidence contains 36 completed two-GPU runs and 288 measured request instances.
Those requests produce 7476 output tokens.
All measured cohorts pass exact full-depth row-work accounting.
Outputs, exit depths, and stop reasons match across PD_G1, A_sep, and Native Mixed.
The prompts repeat across starts and arms.
The counts do not imply 288 independent prompts.

Two selected prompts per model also pass final expected-valid KV identity.
The last prompt hidden row matches before first-token sampling.
The checks cover 192 K/V planes per Ouro request and 264 per Huginn request.
All applicable source-to-destination byte checks pass.

The finite mixed-path pilot is not a full 200/400-request RCC qualification.
Original-HF equivalence is also separate.
The predecessor HF triage remains 8/8 for Ouro and 5/8 for Huginn.
Benchmark answer accuracy is `NOT_MEASURED` in this study.
No original-model-lossless claim is made.

## Measured fixed-cut performance

The table covers eight 128-token prefix requests per independent start.
Output cap is 32, with EOS enabled.
Each arm has three starts on physical GPU 0 and GPU 1.
All latencies are cohort makespans in milliseconds.
Speedups use paired geometric means, not ratios of rounded means.

| Model | PD_G1 mean makespan | A_sep mean makespan | A + Native Mixed mean makespan | Mixed vs A_sep, 95% CI | Mixed vs PD_G1, 95% CI |
|---|---:|---:|---:|---|---|
| Ouro R4, cut 2 | 2072.686 | 2301.229 | 1892.877 | 1.21× [1.09×, 1.39×] | 1.10× [1.04×, 1.17×] |
| Huginn R32, cut 16 | 5394.360 | 5406.167 | 5465.167 | 0.99× [0.96×, 1.02×] | 0.99× [0.91×, 1.05×] |

Evidence class: `MEASURED`.
The intervals resample three independent start pairs.
They are coarse and do not estimate sustainable serving capacity.

Ouro demonstrates a useful existing-Mixed operating point.
Huginn does not show a repeatable makespan gain.
Ouro Mixed has higher ITL P99 than PD_G1 despite its lower mean completion time.
Huginn Mixed has higher ITL P95 than A_sep.
No model receives an SLO-safety or SLO-goodput GO.

## Physical execution and cost interpretation

Selected diagnostic cores contain 1793 combined rows.
Each combines one decode row and seven 256-row continuation blocks.
The execution preserves per-request, per-depth KV metadata.
The shared projections consume genuinely combined matrices.

The selected Ouro core contains 337 GEMM/GEMV kernels.
The selected Huginn core contains 52 GEMM kernels.
Their total kernel counts are 1745 and 309, respectively.
Qualified Down and adapter tiling remains visible in those counts.
One shared model call does not mean one GEMM for an entire transformer core.

A_sep and Native Mixed transfer the same bytes at the same cut.
For the quiet Ouro cohort, each transfers 406847488 bytes in eight bounded fragments.
PD_G1 transfers 805339136 bytes in 16 fragments.
For Huginn, the split arms transfer 1449000960 bytes in 24 fragments.
PD_G1 transfers 2854579200 bytes in 48 fragments.
The split transfer saving belongs to existing A, not to new co-batching.

Graph limits are identical across arms, but batch shapes differ.
Large mixed batches use eager fallback above the eight-row graph limit.
The study therefore does not attribute all cohort gains to isolated GEMM efficiency.
It also does not claim optimal production graph tuning.

## E2E component shares

`rcc_pd/e2e_component_shares.csv` contains percentage shares for six serial host-observed intervals.
The denominator is summed request E2E time within each start.
It is not the GPU critical path or the cohort makespan.
The intervals include delivery, P queue, P prefill, handoff observation, D resume, and post-first-token generation.

Post-first-token generation contributes about 79%–81% of Ouro's summed E2E in this probe.
It contributes about 91%–92% for Huginn.
These are phase observations, not pure computation shares.
Selected-core GPU kernel shares are separate in `operator_fusion_evidence.json`.
Exclusive full-E2E GPU dispatch, FFN, attention, and DMA shares remain `NOT_MEASURED`.

## Phase and resource matrix

| Phase | Ouro | Huginn | Reason |
|---|---|---|---|
| Prior numerical release | CORRECTNESS_READY | CORRECTNESS_READY | Existing matched-runtime release passed |
| Existing A/Mixed source audit | PASS, ALREADY_PRESENT | PASS, ALREADY_PRESENT | Existing combined recurrent path |
| Fixed-cut physical fusion | PASS | PASS | Real row packing, matrix operands, and GPU kernels |
| New finite output/state pilot | FINITE_PILOT_PASS | FINITE_PILOT_PASS | Scoped output and selected-state checks |
| Fixed-cut cohort gain | CONDITIONAL, positive existing-Mixed result | NO_REPEATABLE_GAIN in this regime | Three independent starts |
| New execution mechanism | NO_GO_NOVELTY | NO_GO_NOVELTY | Candidate equals existing A + Native Mixed |
| Conditional admission | NOT_RUN | NOT_RUN | Stage-1 discovery stop |
| W1–W4 serving and capacity | NOT_RUN | NOT_RUN | No later-phase GO |
| 4GPU / 8GPU RCC | NOT_RUN | NOT_RUN | No Stage-2 GO |

Both models use physical GPU 0/1 in the quiet lane.
Huginn diagnostics also use GPU 2/3.
No 4GPU or 8GPU RCC result is claimed.
The missing serving combinations appear in `rcc_pd/unmeasured_serving_matrix.csv`.

The quiet lane contains 795 fabric-process samples with zero observed processes outside its assigned pair.
Each independent start also checks for an empty fabric before launch.
Heavy diagnostics may run on disjoint pairs with owned burners elsewhere.
Their times are excluded from clean performance claims.
Owned burners are restored after the workers release their devices.

## Constraints and next decision

The existing A adapter transfers a whole prompt inside one configured chunk.
Legal long multi-chunk migration remains unimplemented.
Huginn also has a 4096-token total context limit.
A 512-token output cap permits at most 3584 prompt tokens.
No model limit or precision is changed to fill a requested table.

A useful next study must identify an incremental mechanism beyond Native Mixed and generic prefill borrowing.
Pre-cut admission credits and bounded D-side continuation work are possible questions.
They are not measured improvements here.
Their evaluation needs representative long-output requests, frozen SLO calibration, and strong equal-budget controls.

## Reports, code, and publication

Branch: `flashvep/rcc-pd-cross-depth-cobatching-poc`.
The published commit contains this report and the scoped RCC artifacts.
The final handoff records its exact SHA after remote verification.

Main files:

- `rcc_pd_existing_a_scheduler_audit.md`: code paths, compatibility, and novelty.
- `rcc_pd_fixed_cut_cobatching_results.md`: measured performance and physical fusion evidence.
- `rcc_pd_slo_admission_results.md`: unimplemented admission protocol and its gates.
- `rcc_pd/correctness_matrix.md`: predecessor and new finite correctness scope.
- `rcc_pd/full_results.csv`: all measured cohorts.
- `rcc_pd/e2e_component_shares.csv`: host-observed percentage breakdown.
- `rcc_pd/reproduce.md`: exact execution and analysis commands.
- `poc_flashvep/rcc_pd_poc/`: isolated harness, probes, analysis, and validation.

Existing raw data, dirty/staged work, external engines, and model checkpoints remain preserved.
The validator compares the pre-existing Git index hash before publication.
No force push is used.
