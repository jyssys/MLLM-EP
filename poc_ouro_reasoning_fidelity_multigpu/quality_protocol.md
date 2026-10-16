# Frozen public quality protocol

The primary arm is P1.
It uses the existing frozen instruction and the pinned tokenizer's explicit thinking option.
This choice follows the documented template, not a measured score comparison.
P0 is a diagnostic control without the explicit thinking prefix.
The eight diagnostic AIME questions overlap the final 30-problem cohort.
They cannot support an unbiased prompt-selection claim.

All inputs, references, order, IDs, and prompt hashes freeze before new scoring.
The checkpoint, BF16 precision, R4, and early-exit threshold 1.0 remain fixed.
The primary context limit is 32,768 positions.
No input truncation or RoPE extension is permitted.

The sampled AIME lane contains 30 problems and three seeds.
The seeds are 41015, 41016, and 41017.
Sampling uses temperature 1.0, top-p 0.7, and top-k=-1.
The greedy AIME lane contains the same 30 problems.
The secondary greedy lane uses the previous frozen MATH-100 subset.
It is not the full MATH-500 benchmark.

Each lane starts with an 8,192-token cap.
At least 10% length stops trigger a complete-lane repeat at 16,384 tokens.
The same rule permits a complete-lane repeat at 24,576 tokens.
The effective cap cannot exceed 32,768 minus prompt length.
All cap results remain available.
The campaign compares common prefixes across caps.

Quality batches contain at most four requests.
The engine and batch partition remain fixed across caps.
The graph capture sizes are 1, 2, and 4.
EOS remains enabled.
There is no forced output, extra stop string, or answer-based retry.

The primary denominator includes every generation.
A length-stopped output cannot count as completed correct.
The ledger separately records an extracted matching answer inside a capped output.
Accuracy conditional on completion is selection-biased.
It is not the headline pass@1 estimate.

AIME grading accepts a final integer from 0 through 999.
The parser uses the last balanced boxed answer or an explicit final-answer marker.
MATH grading uses strict Math-Verify 0.8.0 with two-second limits.
No generated code executes.
No manual answer change can alter a score.

The bootstrap resamples problems 5,000 times.
All seeds of a sampled problem remain together.
Layers, tokens, and seeds are not independent benchmark problems.
The operational completion target is at least 90% natural termination.
The runtime-fidelity and evaluator checks remain independent gates.
