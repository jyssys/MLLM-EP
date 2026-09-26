# Public reproduction status

Status: **PARTIAL_PROTOCOL; full public benchmark scores NOT_RUN**

## Official registry

The registry uses the Qwen3-VL technical report Table 3, printed p.16, model
column `Qwen3-VL-30B-A3B / Instruct`.

| Benchmark | Reported reference | Local stock reference | Actual EP4 |
|---|---:|---:|---:|
| MMStar | 72.1 | NOT_RUN | NOT_RUN |
| ChartQA test | 86.8 | NOT_RUN | NOT_RUN |
| RealWorldQA | 73.7 | NOT_RUN | NOT_RUN |
| MMMU | 74.2 | NOT_RUN | NOT_RUN |
| OCRBench | 903 raw | NOT_RUN | NOT_RUN |
| DocVQA test | 95.0 | NOT_RUN | NOT_RUN |

Sources:

- Qwen3-VL technical report: <https://arxiv.org/abs/2511.21631>
- Official evaluation reproduction settings: <https://github.com/QwenLM/Qwen3-VL>
- Local-evaluator candidates: <https://github.com/open-compass/VLMEvalKit>

## What was executed

- Twelve deterministic, image-dependent manual prompts on the stock TP4
  reference runtime.
- The same twelve prompts on physical DPEP EP4.
- Four bounded official-lane prompts on physical EP4.

All completed with meaningful visual answers, no empty output, and no prompt
leak. The four official-lane requests used `max_new_tokens=128`; three hit the
cap. They establish sampler plumbing only.

## Why this is not a reported-score reproduction

The full, version-pinned MMStar and ChartQA public splits and their exact local
evaluators were not available in the workspace. No external judge/API cost was
incurred, and no old TP8 ChartQA smoke was substituted. A 12-request manual
cohort cannot support a 3-point tolerance or a benchmark confidence interval.

Consequently, current model behavior is healthy enough for diagnostics, but
`PUBLIC_REPRODUCTION=MATCHED` is not claimed. The next quality action is to
install/freeze the public MMStar and ChartQA protocols and run both stock and
physical EP4 on the full available splits.
