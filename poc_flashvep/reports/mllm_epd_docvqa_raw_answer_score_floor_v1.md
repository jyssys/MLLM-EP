# DocVQA Raw Answer 점수의 Floor 진단

원래 DocVQA ANLS 결과를 유지했다. 답변과 공식 Scorer는 수정하지 않았다.
일부 답변은 정답 Text를 포함하지만 문장 전체의 Edit Distance 때문에 Raw ANLS가 0이다.
증거 수준은 `DIAGNOSTIC`이다. Task Accuracy 또는 품질 Gate PASS로 사용하지 않는다.

| Run | Requests | Raw ANLS Mean | Reference Text Present |
|---|---:|---:|---:|
| reference_main400_natural_template_v2_v0 | 13 | 0.000000 | 12/13 |
| base_C14_agg_S1_natural100_q2_v0 | 13 | 0.000000 | 11/13 |
| base_C14_agg_S2_natural100_q2_v0 | 13 | 0.000000 | 11/13 |
| base_C14_agg_S3_natural100_q2_v0 | 13 | 0.000000 | 11/13 |
| base_C14_e_pd_S1_natural100_q2_v0 | 13 | 0.000000 | 12/13 |
| base_C14_e_pd_S2_natural100_q2_v0 | 13 | 0.000000 | 12/13 |
| base_C14_e_pd_S3_natural100_q2_v0 | 13 | 0.000000 | 11/13 |

Reference Text Present는 소문자·구두점 정규화 후 연속된 단어열의 포함 여부다.
이 지표는 Reference Answer를 사용한다. 잘못된 설명에도 정답 단어가 포함될 수 있다.
따라서 올바른 Answer Extraction이나 의미적 정답 판정을 대신하지 않는다.
예를 들어 원래 답변 `The total pages scanned is 3.`은 Reference `3`을 포함한다.
원래 Raw ANLS는 0이다. 이 진단 뒤에도 그 점수는 0으로 남겼다.
Main100 Prompt와 Natural Output 계약은 동결 상태다. 결과를 보고 Short-answer 지시를 추가하지 않았다.
MMLongBench-Doc의 공식 외부 Answer Extraction도 여전히 `NOT_RUN`이다.
Raw ANLS 0을 곧바로 모델의 의미적 Accuracy 0으로 해석하지 않는다.
반대로 이 Format 진단으로 전체 Quality PASS를 부여하지도 않는다.
대응 Request CSV는 `analysis/DocVQA_raw_answer_floor_v0.csv`다.
원래 Task Score와 Source Scorer Provenance는 `analysis/C14_Agg_EPD2_N3_quality_v0.json`에 있다.
모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.
