# LongBench400 관련 연구와 비교 범위

조회일: 2026-11-21.
아래 내용은 외부 연구 설명이다.
이 캠페인의 측정 결과가 아니다.

## Layered Prefill

[From Tokens to Layers, v1 §4](https://arxiv.org/html/2510.08055v1#S4)는 Layer Group별 Prefill 진행을 설명한다.
§4.3은 Token Chunk와 Layer 분할의 결합도 설명한다.
따라서 두 축을 결합한다는 아이디어 자체를 신규 기여로 주장하지 않는다.
해당 평가의 주요 대상은 Colocated MoE Serving이다.
MoE Expert Weight 재사용 효과를 Qwen3-8B Dense PD의 실측 이득으로 옮기지 않는다.

이번 연구는 1P1D의 계산 소유권을 두 축에서 나눈다.
Imported Prefix KV와 Raw Hidden의 Lifetime이 추가로 필요하다.
이 차이만으로 논문 수준 신규성이 성립한다고 판정하지 않는다.
강한 Pure-axis 대조군과 Continuous Serving 이득을 별도로 검증한다.

## Prefill Deflection

[Towards Load-Aware Prefill Deflection, v1 §3–5](https://arxiv.org/html/2607.02043v1)는 P 대기와 D-side 실행 가능성을 함께 평가한다.
Whole-prefill을 D로 보내는 경로는 KV 전송을 피할 수 있다.
이는 Partial KV와 Hidden을 전송하는 본 연구와 다른 비용 구조다.
해당 논문의 안전 Chunk 크기를 이 환경의 안전 기준으로 사용하지 않는다.
GPU, 모델, Runtime, Decode SLO가 다르다.
Queue를 갱신하지 않는 Request별 최소 시간 선택도 Serving 이득으로 취급하지 않는다.

## DuetServe Motivation

[DuetServe §3와 Figure 2](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf)는 2개 Replica와 1P1D를 비교한다.
그 Workload는 Input8000/Output200이다.
본 캠페인의 Balanced400은 실제 가변 길이와 Natural EOS를 사용한다.
따라서 새 Figure는 방법론을 참고한 별도 Benchmark다.
DuetServe 수치의 직접 재현이라고 표시하지 않는다.
해당 논문은 Attention 비용을 Token Budget만으로 설명할 수 없다고 지적한다.
본 연구도 동일 GEMM Rows와 동일 Attention 작업량을 구분한다.

## 공식 vLLM 계약

[공식 Disaggregated Prefilling 문서](https://docs.vllm.ai/en/latest/features/disagg_prefill/)는 PD를 실험적 기능으로 표시한다.
문서는 TTFT/ITL 분리와 Tail ITL 제어를 설명한다.
이 설명을 PD의 자동 Throughput 개선 보장으로 해석하지 않는다.
Latest 문서는 설치된 0.26.0 Runtime의 정확한 Source가 아니다.
실제 실험은 설치 파일 Hash와 Wheel RECORD 검사 결과를 보존한다.
Wheel 내부 일치만으로 Upstream Commit 동등성을 주장하지 않는다.

## 11월22일 원문 재확인과 Safety 정의

위의 두 논문과 공식 문서를 원문에서 다시 확인했다.
Layered Prefill의 §4.3은 두 축의 결합을 명시한다.
§5.1의 모델은 Qwen3-30B-A3B와 GPT-OSS-20B다.
이는 Dense Qwen3-8B의 측정 결과가 아니다.
Expert별 작은 GEMM에서 발생한 Weight 재사용 효과를 현재 Dense PD의 원인으로 대체하지 않는다.
근거: [Layered Prefill §4.3 및 §5.1](https://arxiv.org/html/2510.08055v1#S4.SS3).

Kairos의 §6.1.1은 DeepSeek-V2-Lite와 A100 환경을 사용한다.
해당 SLO는 TTFT p95≤4s와 TBT p95≤70ms다.
현재 실험의 모든 Client ITL≤34ms 및 Native-relative p99/TPOT 5% CI와 다른 정의다.
따라서 그 논문의 Chunk 실행 가능성을 현재 환경의 Safe Window 증거로 사용하지 않는다.
외부 결과와 맞추기 위해 현재 SLO를 변경하지 않았다.
근거: [Prefill Deflection §6.1.1](https://arxiv.org/html/2607.02043v1#S6.SS1.SSS1).

공식 Latest 문서는 PD의 별도 TTFT/ITL 제어 목적과 Throughput 제한을 설명한다.
이를 이 서버의 모든 Throughput 순위를 예측하는 정리로 취급하지 않는다.
Latest 문서와 설치한0.26.0의 Source를 구분한다.
근거: [공식 PD 설명](https://docs.vllm.ai/en/latest/features/disagg_prefill/#why-disaggregated-prefilling).
