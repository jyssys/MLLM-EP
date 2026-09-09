# Track A — DCP × speculative decoding

## Support and topology

Qwen3-VL cannot run DCP greater than one on four GPUs in current vLLM: it has four KV heads and the non-MLA validator requires TP to be strictly greater than KV-head count. The faithful fallback is cached `deepseek-ai/DeepSeek-V2-Lite-Chat`, an MLA MoE model, with TP4/DCP1 or TP4/DCP4 and EP disabled.

The request matrix was rerun in the isolated vLLM 0.26.0+cu129 environment. N-gram speculation uses the V1 runner because the v0.26 V2 runner reports that n-gram is unsupported.

## Clean request results

| Path | E2E p50 (ms) | TTFT p50 (ms) | TPOT p50 (ms) | Throughput (tok/s) |
|---|---:|---:|---:|---:|
| Vanilla | 3575.92 | 159.55 | 54.23 | 143.86 |
| DCP4 only | 6291.71 | 284.88 | 95.35 | 81.60 |
| Spec only | 4561.33 | 846.32 | 59.26 | 110.90 |
| DCP4 + spec | 3901.70 | 133.27 | 60.20 | 128.85 |

The combined path improves median E2E by 37.99% over DCP-only and 14.46% over spec-only, while remaining 9.11% slower than vanilla. An older compatible v0.20 run also directly logged DCP's capacity effect: 138,080 to 552,320 KV tokens and maximum 8K-request concurrency 16.86× to 67.42× (+300%).

## Correctness gate

The greedy-output gate failed:

- vanilla versus DCP-only: 2/8 exact; common-prefix lengths 1, 49, 64, 12, 1, 49, 64, 12.
- spec-only versus combined: 3/8 exact; common-prefix lengths 64, 49, 64, 12, 1, 49, 64, 12.
- within a fixed DCP degree, vanilla versus spec and DCP-only versus combined each agreed 7/8.
- identical duplicated prompts were internally deterministic within a configuration.

This may be numerical/top-k sensitivity rather than future-KV leakage, but the preregistered correctness requirement is greedy/token agreement. Consequently no speed result is admissible as positive method evidence.

## Oracle and prior art

The observed combined-versus-best-single E2E gap is 14.46%, with a separately measured 4× KV-capacity effect. However, current vLLM issue #50391 already defines the per-query rank-local KV-length problem and the exact strategy space; #53673 covers decoupled draft/target CP; #45425 documents a related DCP speculative correctness failure. Reimplementing one proposed strategy has a direct upstream novelty collision.

## Gate

`NO_GO`: correctness failed, combined is slower than vanilla at this load, there was only one restart after a hard kill became evident, and the exact technical space is already active upstream.
