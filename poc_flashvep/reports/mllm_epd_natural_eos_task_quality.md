# Natural-EOS task scoring on Main400

The original Main400 mix N64 Native EPD run completed 64/64 requests with finish reason `stop`. This reduction verifies original request ID, source, prompt digest and LM input length before scoring the saved response text. The inputs, reference answers and output text were left as recorded. The scores describe one selected cohort and one baseline server start. Candidate-versus-native task noninferiority remains `NOT_RUN`. [Workload audit](mllm_epd_workload_modality_audit.md) · [Result and SHA-256 receipts](../mllm_epd_prefill_heavy_modality_discovery/results/main400_natural_eos_task_scores.json)

| Benchmark | Responses | Task score on this cohort | Metric coverage |
| --- | ---: | ---: | --- |
| DocVQA | 8 | 0.08594 raw ANLS | The original paper specifies ANLS. The pinned related author framework applies lowercasing, trimming and normalized Levenshtein similarity to the entire response. This is a local subset score. [DocVQA paper](https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf) · [Scoring implementation](https://github.com/rubenpt91/MP-DocVQA-Framework/blob/7939f521fe5a82a442f54831b1c0e96daab331d9/metrics.py) |
| Mantis-Eval | 16 | 7/16 = 43.75% | The authors' pinned parser scored all responses as multiple-choice items. The selected cohort contains 0 short-answer items. [Author parser and scoring](https://github.com/TIGER-AI-Lab/Mantis/blob/f3a319202d882a1794d28440b77bba7c58869a9e/mantis/benchmark/eval.py) |
| MP-DocVQA | 16 | 0.15907 raw ANLS | The authors' pinned case-insensitive metric scored all raw responses. Exact answer-string accuracy is 0/16. The run supplies no predicted page index, so answer-page prediction accuracy is `NOT_RUN`. [Task and page metric](https://github.com/rubenpt91/MP-DocVQA-Framework/tree/7939f521fe5a82a442f54831b1c0e96daab331d9) · [ANLS implementation](https://github.com/rubenpt91/MP-DocVQA-Framework/blob/7939f521fe5a82a442f54831b1c0e96daab331d9/metrics.py) |
| MMLongBench-Doc | 24 | Official score `NOT_RUN` | The authors extract a typed answer from the response using GPT-4o before numerical/string/list scoring. This CPU reduction made no external calls. Applying only the downstream scoring function to the whole raw response yields 0.04167 as a `DIAGNOSTIC`. [Author extraction](https://github.com/mayubo2333/MMLongBench-Doc/blob/d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384/eval/extract_answer.py) · [Author scoring](https://github.com/mayubo2333/MMLongBench-Doc/blob/d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384/eval/eval_score.py) |

The original DocVQA and MP-DocVQA responses can contain a correct answer string inside a longer explanation. After lowercasing and punctuation normalization, a reference token sequence occurs in 8/8 DocVQA and 12/16 MP-DocVQA responses. This diagnostic is reference-aware and can match text in a wrong explanation. It changes neither the scored response nor ANLS and supplies no accuracy or noninferiority evidence. [Prior raw-response format finding](mllm_epd_docvqa_raw_answer_score_floor_v1.md)

The scorer source text comes from the previous project's archived per-file snapshots. The reducer checks their exact SHA-256 before reusing the selected scorer functions and includes each source revision and hash in its output. Per-request score records include original prompt and response hashes. Raw answer strings and reference strings stay in the frozen inputs and raw run directories. [Archived scorer sources](https://github.com/jyssys/MLLM-EP/tree/449c3e1873c65206435a6bb9411a889a62e55be7/poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124/references/quality_sources) · [Reducer](../mllm_epd_prefill_heavy_modality_discovery/task_quality.py)

## Full Main400 Native EPD, one start

`native-population/full_main400_eos` submitted the original 400 prompts in frozen shuffled order at 1.6 offered QPS. The scorer joined all 400 saved responses to their original IDs and prompts. All 400 requests completed. Finish reason was `stop` for 399 and `length` for one. The prediction SHA-256 agrees with the run's completion receipt. [Full result and per-request receipts](../mllm_epd_prefill_heavy_modality_discovery/results/main400_full_natural_eos_task_scores.json)

| Benchmark | Completed / planned | `stop` | `length` | Result on submitted responses |
| --- | ---: | ---: | ---: | --- |
| DocVQA | 50/50 | 50 | 0 | Raw ANLS **0.02695** over 50 responses. Raw exact string agreement 0/50. |
| Mantis-Eval | 100/100 | 100 | 0 | Author parser **42/100**. Multiple choice 42/96 and short answer 0/4. |
| MP-DocVQA | 100/100 | 100 | 0 | Raw ANLS **0.04784** over 100 responses. Raw exact string agreement 0/100. |
| MMLongBench-Doc | 150/150 | 149 | 1 | Official GPT-4o extraction `NOT_RUN`. Downstream raw-response diagnostic **0.01342** over 149 scored responses. The capped response is the one null score and remains in the 150-request denominator for completion. |

A reference token sequence occurred in 45/50 DocVQA and 86/100 MP-DocVQA responses under the same diagnostic normalization used for N64. Whole-response ANLS still scores the original text. The N400 baseline increases cohort coverage. It supplies a single Native EPD start with no paired N400 candidate output, so noninferiority remains `NOT_RUN`. No task scores are pooled across benchmarks.

## Paired N64 Native versus Naive deflection, one start

`deflection-chunk4096` ran both Natural EOS arms at 2.955 QPS with P B8192, D B4096, 42 GiB KV and inline encoder transport. The two arms had the same 64 original Main400 prompts, the same saved prompt token IDs, and the same seeded Poisson arrival schedule. Each arm recorded 64/64 completed with finish reason `stop` and zero `length` completions. Both prediction hashes match their completion receipts. [Paired scores and request-level results](../mllm_epd_prefill_heavy_modality_discovery/results/main400_natural_eos_paired_n64_task_scores.json) · [Routing and online scope](mllm_epd_whole_request_deflection.md)

| Benchmark | Scored pairs | Native | Naive | Naive − Native | Exact token sequences | D-local routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DocVQA raw ANLS | 8 | 0.08594 | 0.08594 | 0.000 pp | 7/8 | 0 |
| Mantis-Eval author parser | 16 | 7/16 | 7/16 | 0.000 pp | 12/16 | 1 |
| MP-DocVQA raw ANLS | 16 | 0.15907 | 0.16118 | +0.211 pp | 14/16 | 3 |
| MMLongBench-Doc downstream raw diagnostic | 24 | 0.04167 | 0.04167 | 0.000 pp | 20/24 | 2 |

Exact generated token sequences matched **53/64** requests overall. They matched **4/6** requests that Naive dispatched to D-local and 49/58 that Naive sent through P. Exact sequence agreement measures generation reproducibility. The benchmark columns score the unchanged output text separately. The 11 divergent sequences include nine requests that retained the P→D path, so the paired results cannot attribute all drift to moving LM Prefill. The original comparison is one server start with at most three D-local requests per benchmark. MMLongBench-Doc's official GPT-4o extraction and task noninferiority remain `NOT_RUN`.

## Reproduction

```bash
ns=poc_flashvep/mllm_epd_prefill_heavy_modality_discovery
raw=../h200_epd_modality_runs
archive=../MLLM-EP-mllm-epd.filtered/poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124
score_quality() {
  ../mllm_epd_runtime_env/bin/python -m poc_flashvep.mllm_epd_prefill_heavy_modality_discovery.task_quality \
    --manifest "$raw/dataset/$3.json" \
    --predictions "$raw/$1/raw/$2_requests.jsonl" \
    --run-config "$raw/$1/configs/$2.json" \
    --scorer-campaign "$archive" \
    --out "$ns/results/$4.json"
}
score_quality natural-confirmation mainmix_eos_n64 main400_mix_n64 main400_natural_eos_task_scores
score_quality native-population full_main400_eos natural_media_n400 main400_full_natural_eos_task_scores
score_quality deflection-chunk4096 native_eos_n64 main400_mix_n64 paired_n64_native_scores
score_quality deflection-chunk4096 naive_eos_n64 main400_mix_n64 paired_n64_naive_scores
```

To reconstruct the paired table, join the saved arm results by original `request_id`. Compare the raw `token_ids` lists and resolve the Naive `runtime_request_id` against `deflection-chunk4096/raw/proxy_naive.jsonl` for its actual `deflected` decision.
