# Frozen quality protocol

The primary model is `ByteDance/Ouro-1.4B-Thinking` at revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`.
Every arm uses BF16, fixed R4, SDPA, the official tokenizer, and greedy generation.
The official chat template receives `add_generation_prompt=True`.
The user content requests step-by-step reasoning and a final boxed answer.
All arms use the same exact prefill and correct independent depth-indexed KV.

Eight GSM8K training questions supply predictor calibration data.
Question IDs follow a fixed hash ordering. Calibration IDs do not overlap the evaluation sets.
The frozen pilot contains 64 GSM8K test questions and 64 MATH-500 questions.
The initial screening wave uses the first 16 questions from each dataset.
The manifest also preserves all 1,319 GSM8K test questions and all 500 MATH-500 questions.

The primary evaluator uses the official fixed-batch `model.generate()` driver for every arm.
The initial compacting driver failed long output parity. Its pilot results do not support the primary quality decision.
New directories preserve the corrected execution results without changing sample selection or predictor coefficients.

The initial output cap is 2,048 tokens. EOS remains enabled.
If any pilot arm truncates, the controller repeats the complete paired screening wave at 8,192 tokens.
This rule uses identical prompt groups and a common cap for all arms.
The larger-cap wave remains a new result. It does not replace or delete the first wave.

The evaluator extracts the last balanced boxed expression or an explicit final-answer marker.
GSM8K additionally accepts its explicit `####` marker. Numeric comparison removes commas and normalizes decimal notation.
Both datasets also accept a standalone numeric final response after the reasoning section.
MATH-500 uses strict Math-Verify equivalence with LaTeX extraction and two-second timeouts.
The evaluator records parsing failures, answer matches, completion, and truncation separately.
Answer accuracy counts a valid matching answer, including a capped output with a valid final answer.
Completed-answer accuracy additionally requires EOS. Both endpoints receive paired confidence intervals.
The native-lane clarification fixes these two endpoints before the corrected quality wave starts.
Truncation alone cannot establish a mathematical accuracy regression.

The early-failure rule requires every W1/W0 candidate's paired CI upper bound below minus 10 percentage points for both endpoints.
This rule screens for large regressions after 16 questions per dataset.
If this rule does not decide the experiment, additional fixed-protocol evaluation is necessary.
No pilot result can establish quality GO.
Quality GO requires both complete canonical datasets and the specification's point-estimate and confidence-interval gates.

The bootstrap samples questions, not tokens, layers, stages, or recurrent rounds.
It uses 10,000 paired resamples with seed 20261017.
Small pilot intervals do not establish 1–2 percentage-point noninferiority.
Published model scores use different evaluation conditions and do not replace the local exact baseline.
