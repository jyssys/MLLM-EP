# Correctness contract and current evidence

Updated 2026-09-09 during continuation. This is not a claim of benchmark quality
equivalence. No routing, weights, pruning, sampling policy, or output length is
changed as a successor method.

| Path | Passing evidence | Remaining boundary |
|---|---|---|
| Native Layered Qwen3 BF16 TP2 | Independent HF short answers; paired request IDs/lengths; mechanism assignment conservation | Long free continuation differs also across same-policy restarts; no full task-quality equivalence |
| Native FastPP dense and Qwen3 PP4 | HF short answers; EOS index compatibility test; exact 4-stage trace identities | Long-output hashes are diagnostic, not proof a scheduling policy is wrong or quality-neutral |
| Native NanoFlow Qwen1.5-MoE FP16 EP4 | Five 82-token HF smoke paths; same-prefix B4/B16 independent HF comparison | Near-tie argmax differences remain; not a task-accuracy benchmark |
| Native NanoFlow large prefill | All twelve plain/split2/split4 paired comparisons first outputs 4/4 equal; separate resource/HF diagnostic | Resource M8192 mismatch has HF top-two gap zero but remains excluded from performance; not task-quality certification |
| vLLM Qwen3-VL TP2/DP2/DeepEP4 | Real images, same processor token counts within observer pair, complete identity joins | Separate runtime transfer diagnostic, not correctness certification of native VL ports |

## Mandatory separation

- Every explicit port/runtime failure is excluded from performance evidence.
- Native finite-plan pairs with unequal generated tokens are labelled
  `EXCLUDE_PENDING_CORRECTNESS`, not reported as valid speedups.
- Teacher-forced numerical runs, vocabulary-logit copies and Nsight runs are
  excluded from clean request latency comparisons.
- Free-output inequality at an FP16/BF16 near tie does not alone establish a
  semantic bug. Conversely, a tiny KL on a few tokens cannot certify task quality.
- Descriptive performance tables retaining uncertain runs cannot promote a
  successor. A positive candidate requires stronger correctness/quality evidence.
- Same-prefix reference comparisons preserve input token IDs, teacher history,
  causal output position, dtype and checkpoint. HF attention and native kernels
  are deliberately independent, so bitwise equality is not assumed.

## Regression tests used

`FASTPP/test_eos_regression.py`, `test_partition_kv.py`,
`test_mechanism_probe.py`; `NANOFLOW/test_activation_copy.py`,
`test_partial_plan_init.py`, `test_cohort_probe.py`; and
`COMMON/test_request_trace_client.py`. CPU tests are always run with GPUs hidden.
Tests that execute native method bodies require the pinned source path in their
documented environment variables; an unset test path is a harness error, not a
runtime result.

Continuation regression check: 12 FastPP tests and five COMMON tests pass;
NanoFlow partial-plan lifecycle test and parent request-identity assertions pass.
These CPU tests hide all GPUs and do not replace live model correctness.
