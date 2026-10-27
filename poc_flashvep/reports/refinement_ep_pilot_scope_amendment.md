# User-approved pilot scope

The user requested a smaller pilot on 2026-10-27.
This amendment supersedes the full-test sample requirement for the current pilot.
It does not change the model or the generation protocol.

| Dataset | Original full test | Current pilot | Selection |
|---|---:|---:|---|
| GSM8K | 1,319 | 128 | Original IDs 0–127 |
| HumanEval | 164 | 32 | Original IDs 0–31 |
| Total per method | 1,483 | 160 | Same IDs across methods |

Selection does not depend on answer correctness or generation length.
The pilot contains 89.21% fewer questions per method.
The subset follows benchmark order and is not a randomized population sample.
The original full manifests remain unchanged.

## Reuse and execution

Both backends completed the selected 128 GSM8K requests before this amendment.
The pilot reuses those raw generations without changing them.
The HF source run completed 650 GSM8K requests before the scope snapshot.
The native EP4 source run completed 1,030 requests before that snapshot.
Those counts describe preserved partial runs, not full benchmark reproduction.
Additional in-flight work can exist outside those snapshots.

The controller stopped only this campaign's verified process sessions.
It stopped the old queue and waiting follow-up controller first.
It preserved completed files and retained incomplete in-flight evidence.
It did not signal unrelated jobs.

The new queue evaluates 32 HumanEval requests in each backend.
The EP4 queue also repeats them through the identity hook.
The evaluator executes generated code inside the existing namespace sandbox.
The controller restores owned burners when each device group becomes free.

The old automatic full-evaluation and isolated-baseline queues are canceled.
They must not restart without a new user instruction.
The current state file is `../results/refinement_aware_ep_expert_allocation/state/pilot.json`.

## Unchanged settings

The model remains `inclusionAI/LLaDA2.0-mini`, non-preview.
The revision remains `dad945cac317da394b390f82c7b40691d8a881ed`.
Both paths retain BF16, their existing attention and KV behavior, and the official chat template.
Both retain temperature zero, threshold 0.95, and block length 32.
The HF path retains 32 maximum refinement steps per block.
GSM8K retains its 2,048-token requested cap.
HumanEval retains its 1,024-token requested cap.
EOS remains enabled.

The native runtime rounds its internal generation capacity to a block boundary.
Its returned token count can therefore exceed the requested cap by up to 31 tokens.
This existing behavior remains separate from the HF protocol.
Within-EP policy comparisons must retain the same behavior.

## Interpretation

Every pilot score must include correct/N and a Wilson 95% interval.
Paired differences must use the same selected questions.
The pilot cannot reproduce the full official benchmark score.
It cannot establish a 0.5 percentage-point quality non-inferiority claim.
One GSM8K error changes the pilot score by 0.78125 percentage points.
One HumanEval error changes the pilot score by 3.125 percentage points.

The original accuracy and headroom targets remain research targets.
This sample reduction does not lower those targets.
It limits the strength of any current conclusion.
No final GO follows from pilot accuracy alone.

## Reproduction

```bash
cd /home/esjung/MLLM-EP-github
python -m poc_flashvep.refinement_ep_allocation.postprocess --pilot --score-humaneval
```

The frozen subset and source references reside in `../results/refinement_aware_ep_expert_allocation/pilot/`.
Do not start another pilot controller while its exclusive lock is active.
