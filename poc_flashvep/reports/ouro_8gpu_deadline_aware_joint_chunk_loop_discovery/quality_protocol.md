# Natural-answer quality protocol

Status: all 342 pilot generations completed and received scores.
The pilot covers 114 cases for each of three fixed policies.
Read `quality_summary.md` for results and limits.

The pilot uses the first 64 GSM8K test and first 50 MATH-500 cases from the preserved dataset manifest.
`quality_manifest.json` records the original manifest hash, prompts, answers, and token IDs.
The recipe is the previous Base-model quality recipe.
It is not the conversational serving chat template.
GSM8K uses the preserved few-shot Q/A prompt.
MATH uses the preserved Problem/Solution prompt.

Every case runs natural greedy generation with a 4,096-token maximum.
Native EOS and the original per-case stop strings end generation.
A length cap remains a truncation.
No forced output token enters this lane.
All prompt and decode tokens execute full R4.
The model checkpoint and BF16 precision remain unchanged.

Atomic C1024, Grouped C1024, and Mixed C512 receive the same 114 cases.
Each GPU receives all three policies for its assigned case block.
Policy order rotates across GPUs.
The pilot remains exploratory.
One historical mismatch now has an exact BF16 GEMM geometry reproduction.
This reproduction does not explain every historical mismatch.

The evaluator must report correct/N, parse failures, caps, and complete-correct counts separately.
Paired accuracy differences require question-level resampling with recorded seed and 95% intervals.
Token parity is a different endpoint from task accuracy.
No equality claim can follow from an underpowered interval alone.

The planned next subset is 300 GSM8K plus 100 MATH cases.
Full 1,319 plus 500 evaluation is conditional on mechanism viability and uncertainty.
Any expansion must freeze its inputs before candidate outcomes select cases.
