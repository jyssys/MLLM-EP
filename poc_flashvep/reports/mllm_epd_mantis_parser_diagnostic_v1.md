# Mantis 1문항의 공식 점수 차이 — C14

증거 수준은 `DIAGNOSTIC`이다. 공식 점수와 원본 응답은 변경하지 않았다.

E+PD DP2의 Natural QPS2 독립 3회 중 S2에서 Mantis 점수는 48%였다.
Reference와 S1/S3의 점수는 52%였다. 분모는 각 Start의 Mantis 25문항이다.
차이는 `TIGER-Lab__Mantis-Eval__dj_6` 1문항에서 발생했다.

두 원본 응답의 마지막 줄은 모두 `(A) The first image`다. Reference Answer도 A다.
Reference 응답은 `the answer is`를 사용했다. S2 응답은 `the correct answer is`를 사용했다.
고정한 공식 Parser는 첫 표현을 인식한다. 두 번째 표현은 인식하지 않는다.
따라서 S2의 Parser 입력에는 앞부분의 설명문이 남는다. `get_option`은 첫 알파벳 T를 선택한다.
T는 선택지 범위 밖이다. 이어지는 원문 비교도 실패한다. 공식 점수는 0이다.

이 1문항의 점수 차이를 서로 다른 최종 선택지로 설명할 수는 없다.
그러나 마지막 줄을 별도로 추출한 새 점수로 공식 점수를 대체하지 않는다.
기존 공식 점수 48%와 Reference 대비 −4 percentage points를 그대로 보존한다.
전체 Task Quality Gate는 여전히 미확정이다. MMLongBench-Doc 공식 답변 추출도 미실행이다.

재현 명령은 다음과 같다. 명령은 새 진단 Artifact가 없는 상태에서 1회 실행한다.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.mllm_epd_prefill_heavy_token_split.quality_parser_diagnostic
```

원본 점수는 `analysis/C14_Agg_EPD2_N3_quality_v0.json`에 있다.
진단과 Source Hash는 `analysis/C14_Mantis_parser_diagnostic_v0.json`에 있다.
두 파일은 `mllm_epd_prefill_heavy_token_split_20261124` Namespace에 속한다.
Pinned Source는 `references/quality_sources/Mantis_mantis_benchmark_eval.py.json`이다.
