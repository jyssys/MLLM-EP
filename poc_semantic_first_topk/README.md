# Semantic-first EP critical-path adaptive Top-K PoC

This isolated PoC follows the contract at
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/semantic_first_ep_critical_path_adaptive_topk_poc_spec.md`
(SHA-256 `a1f7f33657f789fd07d8ffc29029813b1328dd903af8a90a043b8825801e5a80`).

It starts from the committed vision heterogeneous Top-K evidence and keeps
three evidence levels separate:

1. offline logical allocation oracles over captured real expert outputs;
2. full-model correctness and benchmark-quality diagnostics that still
   execute all eight experts;
3. exact four-rank DeepEP HT captured-layer replays that measure realizable
   communication/expert latency but are not a production variable-K runtime.

Physical GPUs 4,5,6,7 are the only permitted accelerators.
