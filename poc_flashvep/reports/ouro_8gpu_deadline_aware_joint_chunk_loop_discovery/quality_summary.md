# Natural-answer quality pilot

Evidence: LIVE_MEASURED natural greedy generation.
The checkpoint, prompts, evaluator, and 4,096-token cap follow `quality_protocol.md`.
All 342 generations completed as processes.
Some answers reached the length cap.

| Domain | Policy | Correct / N | Accuracy | Complete-correct / N | Parse failures | Truncated |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | Atomic C1024 | 47/64 | 73.4375% | 47/64 | 6 | 0 |
| GSM8K | Grouped G2 C1024 | 47/64 | 73.4375% | 47/64 | 6 | 0 |
| GSM8K | Mixed C512 | 47/64 | 73.4375% | 47/64 | 6 | 0 |
| MATH-500 subset | Atomic C1024 | 28/50 | 56.0% | 26/50 | 5 | 8 |
| MATH-500 subset | Grouped G2 C1024 | 28/50 | 56.0% | 26/50 | 5 | 8 |
| MATH-500 subset | Mixed C512 | 28/50 | 56.0% | 26/50 | 5 | 8 |

The complete-correct MATH accuracy is 52.0%.
Two extracted-correct MATH answers reached the cap.
The evaluator keeps these endpoints separate.

Each policy generated 6,797 GSM8K tokens and 71,910 MATH tokens.
Each candidate matches Atomic output tokens for all 114/114 questions.
`quality_output_parity.csv` also records prompt, depth, stop, and length agreement.
Equal aggregate token counts alone would not establish output parity.

The paired accuracy difference is 0 percentage points for each domain and candidate.
The question-paired bootstrap 95% interval is [0, 0] percentage points.
The bootstrap uses 5,000 draws with seed 20261023.
Every observed correctness difference is zero, so this interval is degenerate.
It does not prove population equivalence or bound unseen quality loss.

## Coverage limits

These are the first 64 GSM8K and first 50 MATH cases from the frozen manifest.
They are not random samples of each complete benchmark.
All three fixed policies receive identical questions and generation settings.
The pilot mostly tests short benchmark prompts with simultaneous arrivals.
It does not strongly test the 8K staggered-prefill interference regime.

This pilot does not evaluate Adaptive Joint, G3, or a new online scheduler.
The planned 300+100 evaluation and full 1,319+500 evaluation remain NOT_RUN.
No paper-quality equivalence claim follows from this pilot.

## Reproduction and evidence

Run `grade_quality` and `quality_parity` from the experiment package.
`quality_manifest.json` preserves input and source hashes.
`quality_evaluator_lock.json` preserves the evaluator and extractor hashes.
`quality_cases.csv` preserves per-question scores, completion, and source hashes.
Raw generations remain under `raw/p1_quality_pilot_gpu*_v1/`.
