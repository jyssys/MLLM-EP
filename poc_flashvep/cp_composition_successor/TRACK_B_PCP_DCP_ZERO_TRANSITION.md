# Track B — PCP perpendicular DCP / zero-transition KV layout

## Source contract

Current vLLM already implements the central zero-transition ownership primitive for its supported MLA PCP path. `PCPManager.prepare_slot_mappings()` derives global DCP-aware slots, `_convert_to_gathered_slot_mappings()` maps PCP-gathered rows to those canonical slots, and `gathered_kv_write_mask` prevents duplicate decode writes. There is no obvious standalone prefill-to-decode KV copy for this proposal to remove.

PCP perpendicular DCP is itself the subject of upstream RFC #46358. Current issues #51429 and #53573 track validator and collective-correctness gaps in combined configurations.

## Live results

Model: `deepseek-ai/DeepSeek-V2-Lite-Chat`, BF16, exact 16,384-token prompt, 32 decode tokens, vLLM 0.26.0+cu129, EP off.

| Path | TTFT p50 (ms) | E2E p50 (ms) | First ITL (ms) | Steady ITL p50 (ms) | Transition excess |
|---|---:|---:|---:|---:|---:|
| PCP4 only | 99.28 | 1950.19 | 41.03 | 59.55 | 0.00 ms / 0.00% E2E |
| DCP4 only | 103.08 | 2128.45 | 29.40 | 65.74 | 0.00 ms / 0.00% E2E |

The two paths produced identical greedy tokens over all measured repetitions. Cold warmups had large first-ITL spikes, but two warmups removed them; counting those spikes as transition work would be an observer/warmup error.

The validator-admitted TP1/PCP4/DCP4 combined configuration loaded weights but stopped making progress in model warmup. A second 8K/3-output attempt also hit the 180-second bound. Only owned processes were terminated. This is `ENVIRONMENT_BLOCKED`, not a method failure and not claimed to prove issue #53573.

## Oracle and gate

For the working paths, the clean request-level transition-copy oracle is 0%. The target ownership abstraction is already present in current source, while the only missing live point is blocked before serving. Therefore the zero-transition successor premise is `NO_GO` on headroom and novelty; combined runtime reproduction is separately `ENVIRONMENT_BLOCKED`.
