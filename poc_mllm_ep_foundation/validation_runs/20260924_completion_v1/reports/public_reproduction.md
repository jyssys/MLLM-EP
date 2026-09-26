# Public reproduction

Protocol status: **PUBLIC_PROTOCOL_PARTIAL**. The full public splits and local
judge-free evaluators are fixed, while Qwen's private technical-report wrapper
is not fully published.

| Backend | MMStar | ChartQA |
|---|---|---|
| Stock TP4 | n=1500, score=0.7198 | n=2500, score=0.8560 |
| Actual EP4 | n=1500, score=0.7222 | n=2500, score=0.8540 |
| Reported Qwen | 0.721 | 0.868 |

Official lane: temperature 0.7, top-p 0.8, sampling top-k 20, presence
penalty 1.5, repetition penalty 1.0, seed 3407, max-new-tokens 32768, EOS.
Sampling top-k 20 is unrelated to MoE expert top-k 8.
