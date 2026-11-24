# Whole-request Deflection — Same-E 정합성 검사

증거 수준은 `CORRECTNESS_ONLY`다. 실제 Online E 출력을 Full P와 Full D가 공유했다.
N1+N8의 9개 실행 사례가 통과했다. 고유 Request ID는 8개다. 독립 E/P/D 초기화는 1회다.
같은 E 출력을 조건으로 한 검사다. 서로 다른 Encoder Batch의 출력 동등성이나 전체 Task 품질을 증명하지 않는다.

## 실제 경로

원래 E가 실제 이미지 전체를 처리했다. 전체 Prompt는 P를 거치지 않고 D에서 Full28 Prefill과 Decode를 수행했다.
이 요청은 P Prefix나 P→D Prefix KV를 사용하지 않았다. Layer/Token Split이 아니다.
실제 D 요청이 끝난 뒤, 같은 E Export를 Full P Teacher에 전달했다.
두 실제 소비자의 Clone ACK가 끝난 뒤에만 E 소유권을 반환했다.
첫 Greedy 답변이나 KV를 고쳐서 Reference에 맞추지 않았다.
D Full Prefill은 이전 D Stream이 없을 때만 시작했다. Capture의 CPU Copy 때문에 모든 Timing은 성능 판정에서 제외했다.

## 모든 실행 사례

| Phase | Request | Prompt/MRoPE/Delta Strict | KV Layers Strict | Prepared Input Strict | Full Output Strict | Output Tokens |
|---|---|---|---:|---|---|---:|
| N1 | lmms-lab__DocVQA__7621 | True | 28/28 | True | True | 16 |
| N8 | lmms-lab__MP-DocVQA__49187 | True | 28/28 | True | True | 38 |
| N8 | lmms-lab__DocVQA__7621 | True | 28/28 | True | True | 16 |
| N8 | yubo2333__MMLongBench-Doc__688 | True | 28/28 | True | True | 60 |
| N8 | lmms-lab__DocVQA__40799 | True | 28/28 | True | True | 20 |
| N8 | yubo2333__MMLongBench-Doc__975 | True | 28/28 | True | True | 275 |
| N8 | TIGER-Lab__Mantis-Eval__wc_25 | True | 28/28 | True | True | 74 |
| N8 | TIGER-Lab__Mantis-Eval__cw_9 | True | 28/28 | True | True | 17 |
| N8 | lmms-lab__MP-DocVQA__8048 | True | 28/28 | True | True | 9 |

## 계약과 한계

Original BF16 Qwen2.5-VL-7B, BI0, 원래 Vision/Processor/Attention을 유지했다.
모델 Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. GPU4/5/6은 E/P/D다.
B8192/S1024, KV42GiB, Block128, Prefix Cache Off와 공통 Compiled Reduction 설정이다.
단일 이미지·여러 이미지·3페이지·28페이지·45페이지를 포함했다.
D의 실제 Full-request 실행은 정합성 정책에 따라 직렬화했다. C3 Main의 Actual Batch를 1로 가정하지 않는다.
N8의 E는 동시 요청을 받았다. Teacher는 각 요청의 같은 실제 E 출력을 사용했다.
이 조건부 증거는 이전 Encoder Geometry 차이나 Token25%의 KV 실패를 지우지 않는다.
Main100과 동결한 Supplemental50의 Natural/Replay C3 결과는 별도로 측정한다.
Whole-request 선택은 Kairos의 공식 재현 또는 새로운 Modality-aware Method로 명명하지 않는다.
현재 전체 Task Quality는 PENDING이다. 이 보고서만으로 Online Serving GO를 부여하지 않는다.

원시 Tensor Manifest와 해시는 대응 Audit JSON에 있다.
실행별 CSV는 `analysis/whole_DEFLECTION_C43R4_whole_correctness_v0.csv`다.
공통 경로는 `mllm_epd_prefill_heavy_token_split_20261124/`다.
