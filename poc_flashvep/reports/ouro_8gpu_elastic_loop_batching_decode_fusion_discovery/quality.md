# Task-quality scope

GSM8K and MATH-500 answer accuracy are NOT_RUN for this ELB campaign.
The conditional quality stage requires a useful physical grouped-decode opportunity first.
Numerical correctness and bounded token agreement do not establish task-accuracy equivalence.

This study does not relabel old benchmark results as ELB measurements.
The earlier dual-objective accuracy pilot remains historical evidence only.
The model here is pretrained Base Ouro-1.4B, not Ouro-Thinking.
This study does not claim an official paper accuracy reproduction.

The structural hardware probes generate their own greedy tokens.
They ignore EOS to hold output length equal across policies.
They are not complete natural-answer evaluations.
Their stop reason is a diagnostic length cap.
That cap cannot count as a successful complete-response SLO result.

If later physical evidence reopens ELB, the first quality comparison requires matched full-answer prompts.
It must include native Mixed, D1/D2/D4, and the actual ELB candidate.
The pilot uses64 GSM8K and50 MATH-500 questions with the preserved evaluator and prompt recipe.
Correct/N, completed-correct/N, parsing, caps, loss cases, and paired intervals remain separate fields.
Any cache or lifecycle bug blocks promotion regardless of aggregate accuracy.
