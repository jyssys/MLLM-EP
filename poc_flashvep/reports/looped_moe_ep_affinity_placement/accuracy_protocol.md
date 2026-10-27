# Accuracy protocol and release limits

## Frozen comparison rule

The study uses the exact three pinned checkpoints.
IFM evaluation uses the official FA3 backend and task specifications.
Foil evaluation uses the official bundled model and LM-Eval-Harness 0.4.12.
The actual worker uses PyTorch 2.11.0 and Transformers 5.14.1.
Foil's documented evaluation versions are PyTorch 2.13.0 and Transformers 5.15.0.
Those software differences remain explicit.

This comparison rule is fixed before inspection of pooled anchor scores.
Some individual MMLU bucket progress scores were inspected earlier.
The rule is therefore not a fully blinded preregistration.

For an accuracy anchor, report the correct count, denominator, score, and Wilson 95% interval.
Call its published level compatible only within the larger of 2 percentage points or the interval half-width.
This rule checks approximate reproduction, not statistical equivalence.
Report every delta even when this compatibility rule passes.
Do not change the rule after pooled scores become available.

For Foil held-out NLL, use the official held-out construction and packing recipe.
Compare only pure next-token cross entropy.
The provisional compatibility margin is 0.01 nat/token.
Report sequence-level bootstrap uncertainty separately.
A different corpus cannot satisfy that reference check.
The published training loss is never a held-out target.

## Official IFM anchors

MMLU uses all 14,042 test questions and five official dev demonstrations per subject.
The official evaluator specifies the item-weighted micro-average.
Report the subject macro-average only as a separate diagnostic if needed.
Do not replace the official metric to match the paper.

ARC-Challenge uses the official 1,165 four-choice test questions.
It uses zero shots and the official token-normalized metric.
It is not the default 1,172-question harness variant.

The original HellaSwag GitHub remote reports a DMCA restriction.
Its official HF dataset endpoint also fails.
The full `--protocol paper-part1` entry point therefore lacks required input files.
Available anchors run through the official `--tasks` entry point with unchanged paper task specifications.
This is an anchor subset, not a completed eleven-task paper suite.
No unofficial HellaSwag mirror substitutes for the unavailable official source.

## Foil anchors

The available primary tasks are LAMBADA standard, LAMBADA OpenAI, and English XWinograd.
Each uses its full official test set and zero shots.
Foil paper Table 5 supplies their 100B-token reference scores.
The requested FineWeb-Edu held-out slice follows the official 8M-token build command.
The builder explicitly checks overlap against all 140 training shards.
Its output and failures remain separate from completed GPU evaluation.

## Native versus EP

Native and EP use the same prompts, tokenizer, caps, metrics, and task code.
EP transport receives a separate resume identity.
Expert routing and score weights remain authoritative.
No quality result comes from a forced output-token stream.
Small logits probes do not certify task accuracy.
Every incomplete or unavailable task keeps an explicit status.

The official model paths recompute the full prefix.
Cached-decode KV validation is not applicable to these paths.
This limitation must accompany performance results.

## Release gate

Paper-level reproduction, EP numerical fidelity, and candidate fidelity are separate gates.
A material native mismatch produces QUALITY_REPRO_BLOCKED.
Conditional profiling can continue, but it cannot establish quality-preserving acceleration.
A candidate requires an actionable 1.15× oracle over optimized generic EP.
No optimization implementation starts solely because routing correlation is high.
