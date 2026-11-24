# MLLM EPD 선행연구: 최종 재조회 보조 감사

기존 [선행연구 감사](mllm_epd_related_work_and_novelty_audit.md)를 덮어쓰지 않았다.
이 문서는 새로 확인한 두 원문과 Raincheck 상태를 보완한다.
아래 다른 논문의 환경·방법은 현재 캠페인의 실측이나 직접 재현 결과가 아니다.

## RServe / REDServe

제목은 RServe다. Abstract는 System을 REDServe라고도 부른다.
Embedding readiness를 추적하며 같은 요청의 Encode와 LLM Prefill을 겹친다.
여러 요청의 Schedulable Token을 Micro-batch에 모으는 Inter-request Pipeline도 사용한다.
같은 요청의 P Prefix KV를 D로 이전하는 현재 Generic Token Split과 같다고 단정하지 않는다.

Qwen2.5-VL 7B·32B·72B를 평가한다.
Main 환경은 NVLink H100 80GB×8이다. 추가 환경은 PCIe A100 40GB×4다.
MMMU·SGLang Benchmark와 Poisson Arrival을 사용한다. 이미지 해상도를 변경한 Workload도 포함한다.
구현 기반은 gLLM0.0.4다. 비교 vLLM은 0.10.1.1이다.
주 평가의 Output Length는 1 Token으로 고정했다. TTFT와 Input Throughput을 주로 평가한다.
따라서 이 논문만으로 Natural EOS C3의 Decode Tail 보호를 검증했다고 말할 수 없다.
실행 가능한 공식 Code를 이번 캠페인에서 재현하지 않았다. [원문 v1, §3–4](https://arxiv.org/abs/2509.24381v1).

## A Matter of Stages

동일한 3GPU 예산에서 Fused Replica3개와 E1P1D1을 비교한다.
Testbed는 PCIe L40S 48GB×4다. Serving에는 그중 3개를 사용한다.
vLLM0.30.0·NIXL/UCX·Eager·Prefix Cache Off를 사용한다.
모델은 Qwen3-VL-8B·InternVL3-9B·Nemotron-Nano-VL-8B·LLaVA-1.5-7B다.
VisionArena-Chat과 Chart-to-text를 사용한다.
Output Throughput·Mean TTFT·TPOT·GPU Energy를 평가한다.
DVFS의 기준 SLO는 TTFT p99 2s와 Mean TPOT 50ms다.

논문은 비슷한 Output Throughput에서도 EPD의 Mean TTFT가 증가할 수 있다고 보고한다.
이는 현재 분석 질문과 관련 있지만, H100·Qwen2.5-VL·Compiled 경로의 재현은 아니다.
GPU 주파수나 SLO를 이 논문의 값으로 바꾸지 않았다.
공식 Artifact를 이번 캠페인에서 실행하지 않았다. [원문 v1, §3–4](https://arxiv.org/abs/2611.21165v1).

## Raincheck 상태 재확인

2026-11-25 조회에서도 최신 v2는 저자 요청의 Withdrawn 상태다.
표시된 사유는 회사 Disclosure·Publication Approval 절차다.
이를 기술적 반증이나 현재 캠페인의 실험 실패로 해석하지 않는다.
역사적 v1의 Deferred Visual Prefill은 모든 Prompt Token을 첫 Output 전에 완료하는 Exact Split과 다르다.
비교 범위와 철회 상태를 함께 명시한다. [현재 상태](https://arxiv.org/abs/2610.39562), [역사적 v1](https://arxiv.org/abs/2610.39562v1).

## 보존 자료와 결론

PDF와 추출 Text는 `references/primary_papers/`에 있다.
RServe PDF SHA256은 `97ef30c4b9191c88f43b035f40c6238fda946e918583726e26e4185a3ed6595b`다.
A Matter of Stages PDF SHA256은 `672d403ea6e9c8b7a32b7fb908974ba2c0a7a465df253d5f67141a232d88482b`다.
각 JSON에는 원문 URL·Revision·Text SHA256과 다운로드 상태를 보존했다.
E Idle 활용·Embedding readiness·Intra/Inter-request Pipeline 자체의 최초성을 주장하지 않는다.
새 Modality-aware Contribution은 실제 Strong Baseline 대비 추가 이득과 별도 Novelty 검증이 필요하다.
