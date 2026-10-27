# Experiment progress

The full specification was read before implementation.
The study preserves the previous single-GPU and TRUE EP reports.
The new branch starts from `a5b37f72245969782a77c93a1b0f0bb700c3494f`.
The initial staged tree is `4ae6d943920bc10a26216bb7add7024659fa289c`.

The official Part I evaluator rejects FA4 by design.
The previous native-versus-EP FA4 checks therefore do not establish official benchmark reproduction.
A separate dependency target will build the official FA3 implementation.
This build does not replace packages in the existing worker environment.
Only the native BF16 head-dimension-64 forward path is required by these checkpoints.

The backend guard has an explicit source branch.
The diagnostic uses that exact guard as the red/green test.
An uncertain-cause hypothesis search is unnecessary for this explicit compatibility restriction.
Kernel import, ABI compatibility, and checkpoint-level quality remain separate validation steps.

Unused GPUs retain owned burners.
No unrelated process is eligible for termination.

## Completed native quality stage

The separate FA3 build passed the official evaluator guard and actual checkpoint forwards.
Both IFM models completed all 14,042 MMLU questions and all 1,165 eligible ARC-Challenge questions.
Ouro scored 8,979/14,042 on MMLU and 599/1,165 on ARC token-normalized accuracy.
Huginn scored 9,379/14,042 and 592/1,165, respectively.
The scores meet the compatibility rule frozen in `accuracy_protocol.md`.
This compatibility check is not a statistical equivalence test.

Foil completed 5,153 standard LAMBADA questions, 5,153 OpenAI LAMBADA questions, and 2,325 English XWinograd questions.
Its correct counts are 1,632, 2,056, and 1,603, respectively.
All three scores meet the declared reference-compatibility rule.
The official HellaSwag distribution is unavailable.
Its original GitHub repository reports a DMCA takedown.
No unofficial mirror replaces that distribution.

## Completed initial TRUE EP checks

All three checkpoints passed native-versus-EP checks at EP4 and EP8 on the FA3/native backend contract.
Each smoke also exercised the generic current-owner source control.
These checks do not replace full distributed benchmark evaluation.
The full official EP evaluations remain queued behind clean matrix profiling.

## Measurement and dispatcher corrections

A whole-vocabulary finite check exhausted memory at B16 and prefix 2,048.
The replacement checks every logit in bounded views.
The model forward and tensors remain unchanged.
The subsequent corpus runs completed that condition on both IFM models.

Concurrent state writers shared one temporary filename.
The regression reproduced 146 failures in 200 concurrent writes.
Each writer now uses a private temporary file before atomic publication.
The same 200-write regression passes.
The failed attempt and its logs remain preserved.

## Held-out Foil corpus

The official builder scanned identifiers from all 140 training shards.
It rebuilt 7,826 disjoint documents with 8,000,545 tokens.
The builder saved token hashes, document identifiers, and its non-overlap proof.
Pure shifted-token NLL evaluation remains a separate queued GPU stage.
The training loss of 2.499 is not the held-out target.

## Added whole-E2E percentage reporting

The user requested component percentages for the complete E2E interval.
The new observer covers every forward and output completion in a 32-token rollout.
The primary component table uses exclusive endpoint-stream intervals.
It retains communication waiting and unclassified gaps instead of fabricating pure network costs.
The phase table separately identifies first-output completion and remaining decode outputs.

The attribution campaign waits for active quality and control campaigns.
No active GPU experiment lost its devices for this addition.
The reporting code shows `PENDING_MEASUREMENT` until an actual attribution run completes.
CPU regression tests check interval clipping, nested accounting, phase totals, and observer-variant separation.
These CPU tests do not establish GPU correctness, timing validity, or model quality.

## Resume audit and remaining analysis

The resume audit found active GPU workers. It did not duplicate those experiments.
All 90 required TRUE EP performance cells completed before this audit.
Full official EP4 quality evaluation continued on both four-GPU groups.
The later EP8 quality, held-out loss, and native hardware controls remain ordered behind it.

The code now prepares complete report tables from actual saved records.
Preview reports explicitly say `IN_PROGRESS` and retain unavailable values.
The final analysis watcher waits for the GPU campaigns before release.
An isolated local Git regression preserved unrelated staged and untracked files.
That regression is not a commit or push of the actual study branch.

## Stronger generic metadata ablation

The original bridge replicates current router outputs on every rank.
A generic control can derive its count matrix from those existing exact outputs.
It can therefore remove a redundant count All-Gather without using future routing.
This is a generic runtime control, not recurrent novelty.
Its GPU diagnostic compares all derived matrices with actual collective results.
Its clean path retains dispatch, expert GEMMs, combine, and restoration.
The metadata-only variant requires exact logits and generated tokens.

The CPU formula tests cover required shapes and uneven source-row shapes.
They do not prove model correctness or GPU speedup.
The full generic GPU ablation is queued after native and independent-process controls.
No active quality experiment was interrupted for this addition.

## Completed distributed quality stage

Both IFM models completed full MMLU and ARC evaluations at EP4 and EP8.
Foil completed all three selected full downstream tasks at both EP degrees.
All fourteen paired native-versus-EP benchmark comparisons contain zero changed correctness outcomes.
Foil also completed native, EP4, and EP8 evaluation over 7,997,535 held-out targets.
All three modes returned the same NLL, 2.4493475213060765 nat/token.
The paper reference is 2.4492 nat/token.
The official HellaSwag distribution remains unavailable.

## Native OOM recovery

The first native hardware follow-up stopped at IFM Ouro B16/prefix-2,048.
The post, attribution, and finalization workers then reported failure rather than false completion.
The original failed states and logs remain preserved.

A minimized actual-checkpoint run reproduced OOM at prefix 2,053.
The same 32-output-token run completed with expandable allocator segments.
The correction changes allocation, not model arithmetic.
The recovery reuses fourteen complete cells and their 551,936 router records.
It does not rerun or alter those earlier measurements.
Affected native comparison rows retain the allocator difference.

Before the long generic ablation, all three models passed actual EP4 and EP8 metadata-control smoke runs.
These short runs validate transport and exact metadata checks only.
They do not replace full performance or benchmark-quality evidence.
