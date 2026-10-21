# Frozen evaluation protocol

Evaluate the same 64 GSM8K and 50 MATH-500 pilot IDs for every arm.
Use the pinned base Ouro checkpoint, tokenizer, prompt strings, greedy decoding, and 4096-token cap.
Allow native EOS. Preserve the frozen per-question stop sequences.
Store every actual generated token, prompt checksum, answer, parse status, and stop reason.
Report answer accuracy and completed-answer accuracy separately.
Use the existing GSM numeric parser and Math-Verify symbolic grader without candidate-specific extraction.
Bootstrap paired question IDs, not tokens or layers, with 5000 draws and seed 20261021.
The full public suites contain 1319/500 questions. The latest user instruction reduces this study's target to 300/100 per arm.
The 400-question manifest retains the complete pilot and adds metadata-only random samples with seed 20261021.
Do not extrapolate this subset's accuracy to the complete suites.
The complete 1819-question evaluation is USER_REDUCED_NOT_RUN.
Speedup requires the same completed question set. Report differing output/depth work explicitly.
Controlled replay remains separate from actual answer generation.
