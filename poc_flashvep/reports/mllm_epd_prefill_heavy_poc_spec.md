# Image-only MLLM EPD·Exact Token Split PoC 실행 계약

상태는 `PLAN_FROZEN`이다. 아직 Dataset·성능 결과가 아니다.
시작은2026-11-24 05:08:59 UTC다. 최대 연구 예산은20시간이다.
마지막2시간은 검증·보고·Archive·자원 복원·Push에 우선 배정한다.

## 목적과 범위

Qwen2.5-VL-7B-Instruct의 실제 Image-only E/P/D를 검증한다.
Vision Encoder·Processor·Weight·Attention을 변경하지 않는다.
Video·Pruning·Token Merge 변경·Layer/Hybrid Split·학습 정책은 수행하지 않는다.
최신 [Notion 결정](https://app.notion.com/p/4220ee8277e48153a020e374317777f8)을 과거 설계보다 우선한다.
과거4×4 Quota와 Video 계획은 이번 실행 계약이 아니다.

기준 Commit은 `762265301f43517db94e4b20796b4676c23ad562`다.
새 Source Namespace는 `poc_flashvep/mllm_epd_prefill_heavy_token_split/`다.
Evidence Namespace는 `mllm_epd_prefill_heavy_token_split_20261124/`다.
기존 User Checkout·Index와 Dense Source·보고서는 보존한다.

## 모델과 초기 Runtime 계약

| 항목 | 계약 |
|---|---|
| 모델 | Qwen/Qwen2.5-VL-7B-Instruct |
| Model·Processor Revision | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| 수치 경로 | Original BF16, BI0, Exact Causal Attention |
| 생성 | Greedy, Natural EOS, 초기 공통 Cap2048 |
| Decode | 해당 모델의 전체 LLM Layer. Dense Qwen3의36을 상속하지 않음. |
| Position | 원래 MRoPE·Grid·Placeholder·Token Order 유지 |
| 기본 Scheduler | B8192/S1024를 명시. Actual Batch는 별도 측정. |
| 초기 Context 상한 | Input+Output32768. 실제 Config·GPU Feasibility로 검증. |
| Prefix Cache | 모든 Baseline에서 비활성화 |
| 기본 자원 | 같은 H100 SXM80GB3개. 실제 UUID·Topology 기록. |
| Primary | 실제 Online Encoding을 포함한 C3 Continuous Serving |

설치 Runtime의 공식 EPD 지원을 먼저 검사한다.
공식 기능으로 가능한 부분을 새로 구현하지 않는다.
지원하지 않는 Connector나 Numerical 경로를 지원한다고 가정하지 않는다.
Official Debug File Handoff와 최적화된 GPU Transport를 구분한다.
원래 모델을 임의 구현한 Sequential Script로 Serving Baseline을 대체하지 않는다.

## Dataset 선정 규칙

초기 목표는 Single-image50·Multi-image100·Multi-page100·Long-document150, 총400개다.
원본 질문·정답·이미지·페이지 순서를 보존한다.
이미지를 복제하거나 정답 페이지를 선택해서 남기지 않는다.
자연 출력 길이로 샘플을 선별하지 않는다. Input/Output 균등 Quota도 사용하지 않는다.

Sampling Seed는 `2026112401`이다.
Source별 고정 Revision과 원래 ID의 결정적 순열을 사용한다.
먼저 원문·Media 접근과 Processor Context Feasibility를 검사한다.
실행 불가·Context 초과·중복은 별도 Exclusion Manifest에 남긴다.
실제 Serving 성능을 보기 전에 최종 Manifest를 동결한다.

공식 PDF 원문이 Image로 제공되지 않으면 모든 페이지를72DPI RGB로 Rendering한다.
Crop·정답 페이지 제거·Filler를 사용하지 않는다.
이 고정 Rendering 뒤 원래 Processor를 적용한다. Processor Pixel Budget을 Method별로 바꾸지 않는다.
원래 이미지 파일이 있으면 해당 파일을 사용한다.
큰 문서는 일부 페이지만 남기지 않는다. Context를 초과하면 요청 전체를 제외하고 이유를 기록한다.

Source가 부족하면 성능을 보지 않고 다음 인접 Source로 재배분한다.
Single-image는 DocVQA→ChartQA→InfographicVQA 순서다.
Multi-image는 Mantis-Eval→MuirBench 순서다.
Multi-page는 MP-DocVQA→DUDE 순서다.
Long-document는 원래 MMLongBench-Doc을 우선한다.
Stratum 자체가 부족하면 남는 Document Stratum, Multi-image, Single-image 순서로 보완한다.
원래 목표와 실제 Count를 함께 보고한다.

Calibration32는 Main과 이미지·문서 Context가 겹치지 않게 분리한다.
Main400의 Discovery200·Heldout200은 Media 연결 Component 단위로 나눈다.
같은 이미지나 문서의 질문을 두 집합에 나누지 않는다.
Component 크기 때문에 정확한200/200이 불가능하면 실제 수를 기록한다.
400개 자체가 불가능하면 확보한 실제 수와 미달 이유를 명시한다.

## 비교군과 공정성

1. Agg DP3: Full MLLM Replica3개, Round-robin·Continuous Batching·Chunked Prefill.
2. E+PD DP2: 원래 E1개와 Colocated PD Replica2개.
3. Native EPD: E1개·Full Prefill P1개·Full Decode D1개.
4. Whole-request Deflection: 구현·정합성 통과 시 별도 비교.
5. Generic Static Token Split: Motivation Gate 이후 구현·검증.
6. Simple Idle-aware Split: 안전 근거와 실제 C3 실행이 확보된 경우만 비교.

같은 모델·Processor·Media·Arrival·출력 계약·3GPU 예산을 유지한다.
Native EPD를 불필요한 D Cap으로 약화시키지 않는다.
Best-of-family와 Common-policy를 구분한다.
Source Credit·D Admission·KV Capacity·Embedding Cache·Actual Batch를 기록한다.
공통 Cache Policy는 Baseline Pilot 전에 동결한다.

초기 Offered QPS 후보는0.25·0.5·1·1.5·2·3·4다.
작은 Pilot에서 E/P/D Capacity Knee를 찾은 뒤 공통 Grid를 동결한다.
Discovery는 N1, 핵심 Confirmation은 같은 Trace의 독립 Start N3 이상이다.
유리한 Method만 반복하지 않는다.

## Correctness와 판정

N1→N8→Variable-image·Multi-page·Mixed Batch 순서로 검증한다.
Embedding·MRoPE·Grid·KV·Hidden·Logit·Greedy·Task 품질을 구분한다.
같은 실행 Geometry의 Strict 검사와 다른 Batch Geometry의 Numerical Tolerance를 구분한다.
Task Macro 품질 차이의 Context-cluster95% CI 하한은 기준 대비−2pp 이상이어야 한다.
검증하지 않은 구현 오류를 BI0의 정상 수치 차이로 단정하지 않는다.

초기 SLO는 TTFT≤5000ms·모든 Client ITL≤34ms·TPOT≤33ms/token이다.
MLLM Calibration에서 변경이 필요하면 Candidate 성능 이전에 별도로 동결한다.
Native와 Common-policy Native 대비 ITL p99·TPOT p95의 paired CI 상한은1.05 이하다.
Absolute Decode Attainment는95% 이상이어야 한다.
TTFT p50의 비열등성 CI 상한도1.05로 동결한다.

P-heavy는 P Kernel Busy≥80%·Service Proximity≥85%·P Queue와 E 비병목을 함께 요구한다.
Service Proximity는 독립 Capacity Reference를 측정한 후 정의한다.
D Idle Fraction을 Decode-safe Window와 같은 값으로 부르지 않는다.
Profiler 간섭이5%를 넘으면 해당 Capture는 Diagnostic으로만 사용한다.

GO는 실제 E를 포함한 C3에서 Native 대비 TTFT p95 또는 Goodput≥5% 개선을 요구한다.
Paired CI·TTFT p50·Decode·Task 품질 Gate도 함께 통과해야 한다.
C0·Embedding-ready Replay·Offline Oracle만으로 GO를 부여하지 않는다.
실패·미실행·미계측을 그대로 보존한다.
