# Image-only MLLM400 Dataset Profile

입력은 실제 Processor로 측정했다. 이 표는 EPD 성능 결과가 아니다.
공식 단일 GPU Natural EOS 출력은 현재 400/400개다.
미측정 출력은 0으로 채우지 않았다. Encoder Output Shape는 아직 구조적 예상값이다.

## 원문과 선정

Main400 SHA256은 `7ac7d28ae1678499bee35ec20e317760555944cac7b4ca29eb46aeb3957b9592`다.
Model·Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Seed는 `2026112401`이다. 성능과 출력 길이를 보기 전에 요청을 선정했다.
Calibration32·Discovery200·Heldout200은 이미지·문서 연결 Component로 분리했다.
같은 문서의 질문이 Discovery와 Heldout 양쪽에 들어가지 않는다.

| Source | Requests | Mean Images | Min–Max Images | Mean Input Tokens | Mean Output Tokens |
|---|---:|---:|---:|---:|---:|
| lmms-lab/DocVQA | 50 | 1.00 | 1–1 | 4699.50 | 32.28 |
| TIGER-Lab/Mantis-Eval | 100 | 2.54 | 2–5 | 2784.10 | 82.50 |
| lmms-lab/MP-DocVQA | 100 | 3.12 | 2–7 | 14587.50 | 24.46 |
| yubo2333/MMLongBench-Doc | 150 | 29.25 | 15–76 | 16243.23 | 58.17 |

Source는 고정 HF Revision을 사용했다. MP-DocVQA와 DocVQA는 명시한 공개 Mirror다.
원문 ID·정답·이미지 순서·Media Hash는 Manifest에 보존했다.
PDF는 모든 페이지를 72 DPI RGB로 Rendering했다. Crop과 페이지 삭제는 하지 않았다.
선택한 요청의 Truncation 비율은 0/400이다. Input+Output Context 상한은 32,768 Tokens다.
Output Reserve는 2,048 Tokens다. 이는 실제 Natural Output Length가 아니다.
첫 Option Template의 중복 Label을 수정했다. 요청·이미지·정답·Split은 바꾸지 않았다.
이 수정은 Main400 성능 실험 전에 완료했다. 기존 Manifest와 Trace는 그대로 보존했다.

## 실제 Token 통계

| Statistic | Total Input | Visual | Text | Special | Natural Output |
|---|---:|---:|---:|---:|---:|
| mean | 11021.55 | 10951.72 | 39.81 | 30.02 | 52.59 |
| median | 9803.50 | 9769.50 | 31.50 | 11.00 | 25.00 |
| std_population | 7873.24 | 7862.30 | 21.11 | 29.29 | 117.78 |
| min | 105.00 | 40.00 | 19.00 | 7.00 | 2.00 |
| p25 | 4556.00 | 4523.75 | 25.00 | 9.00 | 15.00 |
| p75 | 15669.00 | 15627.50 | 50.00 | 51.00 | 55.25 |
| p90 | 23360.20 | 23247.00 | 69.10 | 77.40 | 117.00 |
| p95 | 25672.70 | 25551.25 | 82.00 | 87.00 | 159.20 |
| max | 30648.00 | 30608.00 | 183.00 | 157.00 | 2048.00 |

Input Token 합계는 4,408,620개다. Visual Token 합계는 4,380,689개다.
Token 가중 Visual 비율은 0.993664이다. Text·Special·Visual 합계는 Total Input과 일치한다.
Image Reference는 5,004개다. 고유 Byte Hash는 2,731개다.
고유 Pixel Hash는 2,731개다. 요청 ID는 400/400개가 고유하다.
원문 Dataset의 동일 이미지·문서 재사용은 보존했다. 새로운 이미지 복제는 하지 않았다.

요청당 이미지 수는 평균 12.51장, 중앙값 3장이다.
p95는 41장이다. 최소 1장, 최대 76장이다.
문서의 각 페이지 이미지를 포함한 수다. 이미지 해상도가 다르므로 장수만으로 Token 작업량을 비교하지 않는다.
출력 상한 2,048 Tokens에 도달한 요청은 측정 400개 중 1개다.
Cap Hit 출력은 EOS 종료와 구분한다. 상한에 도달한 요청도 통계에서 제외하지 않았다.

## 제외와 한계

초기 Template 적합성 검사 분모는 736개 요청이다.
그중 부적격 요청은 269개다.
이 분모는 전체 후보 Pool이 아니다. 목표 수 확보 후 남은 후보는 처리하지 않았다.
정확한 제외 사유는 `feasibility_audit.json`에 있다.
Pinned LFS 크기와 응답이 다른 `mi_phone.pdf`는 원문 검증 실패로 기록했다.
해당 응답을 고정 Revision의 정상 원문으로 대체하지 않았다.
Task Quality는 아직 미확정이다. 위 출력 길이는 정확도 점수가 아니다.
이 Dataset은 Image-only 장문 QA 실험이다. 전체 Production Traffic의 대표 분포가 아니다.

CSV는 `analysis/dataset_requests.csv`다. 집계 JSON은 `analysis/dataset_profile.json`이다.
Natural Output은 부분 측정치와 최종 400개 통계를 구분한다.
