# LongBench-ShortAnswer200-v2: 사전 Dataset 계약

상태: `INPUTS_FROZEN / BI0_NATURAL_REFERENCE_COMPLETE`.

GovReport40개를 Qasper40개로 바꿨다. 이는 사용자가 사전에 지정한 Workload 변경이다. 긴 Output이 잘못된 Workload라는 뜻은 아니다. GovReport를 포함한 기존 Mixed200-v1은 그대로 보존했다.

## 입력 동결

| 항목 | 값 |
|---|---|
| Dataset | LongBench v1, `zai-org/LongBench` |
| Source Revision | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` |
| Tasks | HotpotQA/MuSiQue/NarrativeQA/QMSum/Qasper, 각각40개 |
| Qasper Seed | `2026112004`, 기존 Seed2026111999+새 Task Index5 |
| Request Order | 기존 GovReport의40개 Slot을 Qasper로 교체 |
| 유지한 입력 | 나머지160개 Request ID, 원문과 Token ID 완전 일치 |
| Model Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Context / Output Reserve | 40960 / 2048 Tokens |
| Input Limit | 38912 Tokens |
| Truncation | 기존과 같은 Chat Template 후 Head19456/Tail19456 |
| 제외 / Truncation | 0개 / 10개 |
| Dataset SHA256 | `8e58b2f75bd0c0ef274d6e53d9b4044e6aecf3a09a787133ee128b21f702e6d6` |
| Sampling Manifest SHA256 | `93cec9a848d8ab491fccb79c4ceb18e03e7aa74e63dffa0f7fb33981850d7d41` |

Input 평균은15555.375 Tokens다. p50은15661, p90은32978.3, 최대는38912다. Qasper Input 평균은5014.25 Tokens다. 실제 길이는 Qwen3-8B Chat Template 뒤에 측정했다. 반복 Filler는 사용하지 않았다.

Qasper40개는 전체 원본에서 비복원 추출했다. Output이나 Serving 결과를 보고 요청을 교체하지 않는다. Qasper의 Natural Output 평균이100 미만이 아니면 대체 Task 전체를 별도 Version으로 조사한다. 사전 대체 후보는 MultiFieldQA-en, 2WikiMQA 순서다. 개별 요청을 유리하게 교체하지 않는다.

## 생성과 평가 계약

Primary는 BI0, Original BF16, Exact Attention, Greedy, Thinking Disabled다. Natural EOS를 허용한다. 공통 Max Output Cap은2048이다. QMSum을 포함한 모든 요청이100 Tokens 미만이라고 가정하지 않는다.

BI0 Single-GPU C1 Reference에서 실제4723 Tokens를 생성했다. 평균은23.615, p50은7, p95는102.1, 최대는152 Tokens다. Cap Hit는0/200이다. 100 Tokens 이상 요청은12/200이다. 이12개는 모두 QMSum이다.

| Task | 요청 수 | 실제 Output 평균 | 최대 |
|---|---:|---:|---:|
| HotpotQA | 40 | 6.1 | 52 |
| MuSiQue | 40 | 8.7 | 38 |
| NarrativeQA | 40 | 8.175 | 16 |
| QMSum | 40 | 81.3 | 152 |
| Qasper | 40 | 13.8 | 44 |

Qasper p95는30.05 Tokens다. 100 Tokens 이상 Qasper 요청은0/40이다. 사전 평균100 미만 조건을 만족했다. 대체 Task를 사용하지 않았다. Reference SHA256은`930745cf78e0f76bc9be33bdb8e1446f534b92b0340ee4568ba0efc84d33aa87`이다.

Task 품질은 공식 LongBench v1의 Task별 지표를 따른다. 네 QA Task는 정규화한 Token F1을 사용한다. QMSum은 ROUGE-L F1을 사용한다. 정답이 여러 개면 최대 점수를 사용한다. [공식 평가 코드](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py)를 기준으로 한다.

이 Dataset은 Long-context Short-answer 연구용 분포다. Production 전체 Traffic을 대표하지 않는다. 기존160개 입력을 유지했으므로 모두 새로운 Held-out 요청이라고 부르지 않는다.

## Evidence

- [사전 입력 통계](dense_ar_bi0_shortanswer200_20261120/analysis/shortanswer200_v2_pre_serving_contract.json)
- [Sampling Manifest](dense_ar_bi0_shortanswer200_20261120/data/shortanswer200_v2_sampling_frozen.json)
- [Truncation Manifest](dense_ar_bi0_shortanswer200_20261120/data/shortanswer200_v2_truncation_manifest.json)

입력 통계는 `STRUCTURAL_ACCOUNTING / PRE_SERVING_DATASET_CONTRACT`다. 출력 길이는 `CORRECTNESS_ONLY / MEASURED_NATURAL_OUTPUT_REFERENCE`다. 이 출력 길이를 Online Serving 성능으로 해석하지 않는다.
