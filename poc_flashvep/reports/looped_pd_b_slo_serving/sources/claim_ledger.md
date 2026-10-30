# Source and claim ledger

Sources were checked on2026-10-29. These sources do not supply this PoC's measured speedups.

| Source | Supported use | Limit |
|---|---|---|
| [DistServe paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), sections1 and6 | TTFT and request-average TPOT, joint SLO attainment, open-loop request-rate sweep, and equal-resource comparison | Its models and hardware differ. Its SLO values do not calibrate Ouro or Huginn. |
| DistServe, section6 and figure7 | ShareGPT chatbot prompts and Poisson arrivals support our workload family | We select a different filtered subset. We generate natural outputs with safety caps. We do not reproduce its exact traffic. |
| [Splitwise](https://arxiv.org/abs/2311.18677v2), abstract | Phase separation and state transfer already exist | This does not establish loop-frontier novelty or equivalent serving performance. |
| [DuetServe](https://proceedings.mlr.press/v306/gao26m.html), abstract | Adaptive SM partitioning offers another generic prefill/decode interference control | We do not implement DuetServe. DP_Mixed is an RLT baseline, not DuetServe. |
| [ShareGPT dataset card](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | Public English conversation text and declared Apache-2.0 license | The source contains filtering and cleaning bias. Its traces do not contain our generated arrival timestamps. |
| Frozen `frontier.py`, `boundary.py`, and `pd/connector.py` | Producing events, immutable KV ranges, consumer admission, and transfer descriptors | Source structure is not a measured latency benefit. |
| `core/scheduler.py`, `_record_token` | The pinned runtime uses `STOP` for EOS and `LENGTH` for the output cap | Raw finish reasons remain in every request record. |

The logical-layer streaming equivalence is an analytical counterexample.
It is not a second measured runtime.
The absent GenericLayer live arm prevents a novelty GO.
