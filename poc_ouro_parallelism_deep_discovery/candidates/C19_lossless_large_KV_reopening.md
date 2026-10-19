# C19: Lossless shared-base KV ownership

Status before encoded-size analysis: untested reopening, not a finalist.
The existing C08 negative concerns small hidden messages and unfused processing.
This card changes the payload regime to large immutable historical KV.
It does not erase the C08 result.

| Required field | Contract |
|---|---|
| 1. Identity | Revision 1. Large-KV reopening recorded before its encoded-size analysis. |
| 2. Mechanism | Store exact BF16 key/value words as a shared base and lossless recurrent differences, with explicit page owners. |
| 3. Ouro property | Distinct depth-indexed KV belongs to repeated shared operators. Similarity must improve measured coding, not be assumed. |
| 4. Target | Both Base sizes and modes. Long-context C1/2/4/8, P2/P4. P8 remains modeled. |
| 5. Baseline | Strong native TP, independent lossless compression, and free alias-aware terminal KV for the chosen adaptive policy. |
| 6. Sequence | Publish complete original KV, encode immutable page, retain base/version, decode exact words before the original attention consumer. |
| 7. DAG | A depth's query still waits for its true predecessor. A decoder waits for every named base and residual page. |
| 8. Concurrency | Independent page coding or attention-owner work only. Four queries of one token are not simultaneously ready. |
| 9. Exactness | Byte-exact representation in principle. No GPU codec or full-generation correctness result. |
| 10. Memory | Encoded base/differences, indices, lengths, decoder scratch, and current writable rows. Count both K and V before a model claim. |
| 11. Messages | Native TP already keeps historical KV local. No nonexistent remote-KV transfer can count as a saving. |
| 12. Critical path | Possible reduced storage or real reads, but repeated base/difference decoding can amplify HBM reads. |
| 13. Prior | Generic lossless KV coding and FlashLoop's base/residual organization require direct comparison. Quantization gains are ineligible. |
| 14. Changed premise | Large KV can amortize startup unlike a4096-byte hidden message. Base ownership and decoder cost remain mandatory. |
| 15. Falsifier | Compare actual raw, independent-coded, adjacent-difference, and common-base sizes. Charge uncached reconstruction reads. |
| 16. Evidence plan | Reuse the bounded actual native key captures. CPU encoded size is not GPU traffic or latency. |
| 17. Compatibility | Preserve all fixed states. Adaptive missing-depth aliases receive no replication/compression contribution credit. |
| 18. Counterfactual | Compare an alias-aware unroll and independent coding. A local cache-format gain alone is not a new distributed axis. |
| 19. Complexity | No new decoder kernel, engine, or precision change until legal ownership and physical headroom survive the screen. |
| 20. Initial disposition | CONDITIONAL_MECHANISM, with no shortlisted architecture or achieved speedup. |

The first screen contains selected keys only.
It cannot establish full K+V compression, model memory, or workload-level benefit.
The measured native TP path has no MoE Dispatch/Combine A2A.
