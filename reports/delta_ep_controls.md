# DeltaEP controls

| Target | adjacent dLLM | wrong-token | nonadjacent | generic zero |
|---|---|---|---|---|
| EP4 | 0.0157% | 0.0102% | 0.0105% | 0.0000% |
| EP8 | 0.0203% | 0.0124% | 0.0133% | 0.0000% |

Adjacent refinement is measurably more compressible than every control, so temporal signal exists; the absolute lossless and stage gains are nevertheless too small. AR one-shot has no previous same-edge predictor and therefore 0% temporal saving.
