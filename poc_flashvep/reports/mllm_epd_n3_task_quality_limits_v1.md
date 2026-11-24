# N3 Natural 출력의 Task 품질 감사

원래 Answer와 고정한 Task Scorer를 그대로 사용했다. Prompt·EOS·출력을 수정하지 않았다.
Natural EOS만 평가했다. Equal-output Replay는 품질 동등성 검사로 사용하지 않았다.
전체 품질 판정은 `HOLD`다. Runtime 결과의 실패와 이 품질 미확정을 분리한다.

## Task별 원래 점수

| Subset | Source | Method | Request IDs | Contexts | Starts | Raw Score | Strict R0 Outputs/Observations |
|---|---|---|---:|---:|---:|---:|---:|
| Main100 | TIGER-Lab/Mantis-Eval | NativeStrong | 25 | 23 | 3 | 0.506667 | 53/75 |
| Main100 | TIGER-Lab/Mantis-Eval | NativeCommon | 25 | 23 | 3 | 0.520000 | 57/75 |
| Main100 | TIGER-Lab/Mantis-Eval | Static10 | 25 | 23 | 3 | 0.480000 | 57/75 |
| Main100 | TIGER-Lab/Mantis-Eval | Idle10 | 25 | 23 | 3 | 0.506667 | 54/75 |
| Main100 | lmms-lab/DocVQA | NativeStrong | 13 | 12 | 3 | 0.000000 | 28/39 |
| Main100 | lmms-lab/DocVQA | NativeCommon | 13 | 12 | 3 | 0.000000 | 29/39 |
| Main100 | lmms-lab/DocVQA | Static10 | 13 | 12 | 3 | 0.000000 | 30/39 |
| Main100 | lmms-lab/DocVQA | Idle10 | 13 | 12 | 3 | 0.000000 | 29/39 |
| Main100 | lmms-lab/MP-DocVQA | NativeStrong | 25 | 24 | 3 | 0.042867 | 68/75 |
| Main100 | lmms-lab/MP-DocVQA | NativeCommon | 25 | 24 | 3 | 0.042867 | 66/75 |
| Main100 | lmms-lab/MP-DocVQA | Static10 | 25 | 24 | 3 | 0.042867 | 72/75 |
| Main100 | lmms-lab/MP-DocVQA | Idle10 | 25 | 24 | 3 | 0.042867 | 68/75 |
| Main100 | yubo2333/MMLongBench-Doc | NativeStrong | 37 | 32 | 3 | 0.027027 | 89/111 |
| Main100 | yubo2333/MMLongBench-Doc | NativeCommon | 37 | 32 | 3 | 0.027027 | 91/111 |
| Main100 | yubo2333/MMLongBench-Doc | Static10 | 37 | 32 | 3 | 0.027027 | 78/111 |
| Main100 | yubo2333/MMLongBench-Doc | Idle10 | 37 | 32 | 3 | 0.027027 | 86/111 |
| Heldout50 | TIGER-Lab/Mantis-Eval | NativeStrong | 22 | 21 | 3 | 0.454545 | 43/66 |
| Heldout50 | TIGER-Lab/Mantis-Eval | NativeCommon | 22 | 21 | 3 | 0.454545 | 43/66 |
| Heldout50 | TIGER-Lab/Mantis-Eval | Static10 | 22 | 21 | 3 | 0.454545 | 48/66 |
| Heldout50 | TIGER-Lab/Mantis-Eval | Idle10 | 22 | 21 | 3 | 0.469697 | 45/66 |
| Heldout50 | yubo2333/MMLongBench-Doc | NativeStrong | 28 | 25 | 3 | 0.035714 | 69/84 |
| Heldout50 | yubo2333/MMLongBench-Doc | NativeCommon | 28 | 25 | 3 | 0.035714 | 70/84 |
| Heldout50 | yubo2333/MMLongBench-Doc | Static10 | 28 | 25 | 3 | 0.035714 | 66/84 |
| Heldout50 | yubo2333/MMLongBench-Doc | Idle10 | 28 | 25 | 3 | 0.035714 | 70/84 |

## CI와 해석

동일 Start Block과 이미지·문서 Context를 함께 Resampling했다. 같은 Context의 질문을 분리하지 않았다.
독립 서버 시작은 3회다. Request 또는 Token 수를 독립 Start 수로 취급하지 않았다.
Paired 차이는 pp다. 고정 품질 Margin은 CI 하한≥−2pp다. Source별 결과는 대응 CSV에 있다.
N3와 작은 Context 수는 CI의 한계다. 유리한 Task만 남기거나 Scorer를 변경하지 않았다.
Mantis의 원래 Parser 결과도 보존했다. 같은 선택지를 다른 문장으로 출력하면 Parser 점수가 달라질 수 있다.
DocVQA는 Native와 후보 모두 Raw ANLS Floor가 남았다. 이 값으로 의미적 품질 동등성을 증명하지 않는다.
MP-DocVQA의 낮은 Raw Score도 그대로 보존했다.
MMLongBench-Doc의 공식 GPT 기반 Answer Extraction은 미실행이다. Raw 점수는 완전한 공식 품질 지표가 아니다.
승인된 외부 API 또는 평가 Key 없이 유료 추출을 실행하지 않았다.
BI0는 서로 다른 Batch Geometry에서 완전히 같은 Greedy 출력을 보장하지 않는다.
그러나 이 사실을 구현 오류나 Task 품질 저하를 허용하는 근거로 사용하지 않는다.
원래 E 출력이 Geometry에 따라 달라지는 진단과 KV 정합성 검사는 별도 보고서에 있다.
동일 E·LLM Geometry의 Strict 상태 검사도 모든 Task 품질을 대신하지 않는다.

## 재현 자료

모델 Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.
Main100 SHA256은 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1다. Main100과 기존 Heldout50을 각각 평가했다.
GPU4/5/6, Original BF16, BI0, B8192/S1024, KV42GiB, Block128을 유지했다.
Scorer Source는 `references/quality_sources/`의 고정 Revision과 Hash를 사용했다.
원시 점수는 `analysis/token_confirmation_C37_raw_quality_v0.csv`다.
집계와 CI는 `analysis/task_quality_C37_*_v0`에 있다.
