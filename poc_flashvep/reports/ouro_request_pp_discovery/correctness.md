# Correctness

## Reference and acceptance rule

The reference loads the original pretrained HF model on physical GPU4.
It changes only the configured executed loop count for fixed R2/R3.
Layer hooks assert `24 * R * generated_tokens` actual decoder calls.
The reference retains separate KV for every executed loop and physical layer.
Greedy generation feeds each selected token into the next forward pass.
This is free-running generation, not teacher forcing.

Numerical limits were frozen before measurements in [preflight.json](preflight.json).
They require zero greedy-token mismatches.
They permit logit max/mean absolute errors of 0.5/0.05.
They permit hidden-state and KV relative L2 errors of 0.02.
No held-out failure caused a tolerance increase.

## Final runtime results

38 request-arm-depth-shape checks passed. Every checked terminal logit, hidden state, and valid KV element matched bitwise.

| Prompt tokens | Concurrent requests | Executed depth | Request-arm checks | Result |
| --- | --- | --- | --- | --- |
| 16 | 4 | 2 | 8 | PASS, bitwise |
| 16 | 4 | 3 | 8 | PASS, bitwise |
| 16 | 4 | 4 | 8 | PASS, bitwise |
| 128 | 4 | 4 | 8 | PASS, bitwise |
| 512 | 2 | 4 | 4 | PASS, bitwise |
| 2048 | 1 | 4 | 2 | PASS, bitwise |

These checks reuse a small fixed prompt pool. They are not independent benchmark-quality samples.
All 81 primary paired cohorts also produce identical greedy token IDs across placements.
Each primary output contains exactly the forced token count.
Cache ownership and length checks cover every primary request.
The final cache length is `prompt_length + generated_tokens - 1`.
The last emitted token does not enter the cache until another forward pass consumes it.
All cache slots stay on their executing GPU.
The slot key is `24 * loop_index + physical_layer_index`.

## Preserved failures and changes

1. A masked-capacity SDPA prototype changed a near-tied greedy token. The exact attention length replaced that path.
2. A CUDA Graph prototype failed interleaved requests because its position tensor lacked an explicit lifetime owner.
3. The final graph tuple retains that tensor. The same interleaving regression then passed.
4. An early campaign stopped during graph setup. It produced no headline measurements.

The failed outputs, commands, and logs remain under `failures/`, `logs/`, and versioned correctness JSON files.
The final comparison uses only `primary_*_v5_r*` records.
Older eager and graph prototypes do not contribute to headline latency.

## Evaluation boundaries

Benchmark accuracy: NOT_RUN. Forced-length hardware runs do not establish GSM8K or MATH accuracy.
True dynamic-depth/ragged-KV correctness: NOT_RUN.
Fixed R2/R3 truncate both prefill and decode. They do not reproduce full-depth prefill followed by adaptive CDB decode.
The official HF gate collects loop states before its exit selection. That path is not proof of online compute skipping.
No gate, threshold, router, expert, precision, or pretrained tensor changed in this experiment.
This dense model has no MoE router or expert path.

The optional backend controls add 16 reference checks. They also pass with bitwise terminal logits and KV.

## EOS-enabled smoke

- layer_pp4: {'length': 4}, lengths [64, 64, 64, 64].
- loop_pp4: {'length': 4}, lengths [64, 64, 64, 64].

EOS-enabled arm-to-arm token/stop parity: True.
If every request reaches its cap, this smoke tests enabled stopping but not an actual EOS transition.
