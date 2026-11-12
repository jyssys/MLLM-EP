# Dense AR Layer-wise PD: 선행연구 및 Runtime 검토

## 범위

이 문서는 설계와 구현 계약을 검토한다.
외부 논문의 성능 수치는 이번 GPU 실측이 아니다.
선행 구현이 보이지 않는다는 이유로 Novelty를 주장하지 않는다.

## Kairos

Kairos는 요청 전체 Prefill을 D에 배정한다.
이미 P에서 계산한 Layer의 잔여 계산을 이전하는 방식은 아니다.
논문은 vLLM 0.18.1 기반 구현을 설명한다.
평가의 2P2D A100 구성과 DeepSeek-V2-Lite는 이번 Dense Qwen3-8B 1P1D와 다르다.
Company X Trace의 짧은 응답 조건도 이번 Output16과 구분한다.
TBT 예측에는 Decode Batch, KV 점유와 Prefill Chunk를 사용한다.
Prefill 위치 선택은 전체 예상 완료시간도 비교한다.
따라서 D가 한순간 비었다는 조건만 재현하면 Kairos와 동등한 정책이 아니다.

논문에 표시된 공식 Repository는 `sudokara/Kairos`다.
이번 접근에서 GitHub와 `git ls-remote`는 Repository Not Found를 반환했다.
공식 코드를 확보하거나 실행했다고 주장하지 않는다.
[논문 및 공식 코드 링크의 출처](https://arxiv.org/html/2607.02043v1).

## FluidPD / FluidToken

FluidToken은 Token Prefix의 KV를 전달하고 Token Suffix를 D에서 계산한다.
PPI는 Queue의 예상 Prefill 완료시간과 남은 TTFT 예산을 비교한다.
DPI는 Active Decode와 Active/Reserved KV를 구분한다.
Suffix 완료 예상시간은 P의 Prefill Deadline에서도 차감한다.
구현은 SGLang 0.5.4의 KV 사전 할당과 비동기 전송을 유지한다.

논문의 Suffix 예산은 예측 TPOT Slack에 계수 eta를 적용한다.
eta가 1보다 크면 여러 Decode Iteration의 여유를 빌릴 수 있다.
이 정책은 Native-relative ITL p99 5% 비열등성을 보장하지 않는다.
이번 PoC는 TTFT 이득만으로 Decode 보호를 통과시키지 않는다.
확인한 논문에서 공식 배포 Repository는 찾지 못했다.
유사 기능을 구현하면 `style reproduction`으로 구분한다.
[FluidPD 설계와 평가](https://arxiv.org/html/2610.06917v1).

## Layered Prefill

Layered Prefill은 Layer Group 사이에 Hidden/Residual을 유지한다.
동일 GPU에서 Prefill Group과 전체 Decode를 함께 진행한다.
MoE에서는 작은 Token Chunk가 유발하는 Expert Weight 재로딩을 줄인다.

Dense Ablation은 단일 H100의 Qwen3-8B, arXiv 1.5 requests/s다.
Layered/Chunked의 Mean TTFT는 1.45/0.955s다.
Mean TBT는 20.3/16.2ms다.
저자는 Dense에서의 Utilization 감소와 MoE 재사용 이득 부재를 설명한다.
이 결과는 Partial Layer Forward가 불가능하다는 증거가 아니다.
이번 Cross-GPU P→D Continuation의 성능 증거도 아니다.
[논문 §5.13](https://arxiv.org/html/2510.08055v2).

공식 코드는 `scale-snu/layered-prefill`이다.
검토한 Commit은 `053f80e5201a7c0ab56e468e3d578907a2ca9cc3`다.
Scheduler는 Decode 요청을 먼저 선택한 뒤 Prefill Stage Queue를 진행한다.
ModelRunner는 요청별 Hidden, Residual, Input IDs와 Positions를 보관한다.
모델은 선택한 Layer Group의 Prefill과 나머지 Layer의 Decode를 구분한다.
검토한 코드는 nano-vLLM 기반이다.
이번 vLLM/NIXL Runtime에 그대로 설치하지 않았다.
전송, Source Lifetime과 ACK는 별도 검증 대상이다.
[공식 코드](https://github.com/scale-snu/layered-prefill).

## vLLM RFC와 설치된 Runtime

RFC #62689는 MRV2와 기존 PP 모델의 `IntermediateTensors` 계약을 사용한다.
각 Group에 맞춰 Layer 범위와 PP First/Last 의미를 바꾼다.
해당 초안은 동기 Scheduling을 요구하고 `torch.compile`을 끈다.
KV/EC Connector도 지원 범위에서 제외한다.
따라서 완성된 P→D Compute Continuation으로 바로 사용할 수 없다.
RFC에 기록된 사후 선택 SLO를 이번 실험에 적용하지 않는다.
[공식 RFC](https://github.com/vllm-project/vllm/issues/62689).

이번 Runtime은 실제 설치된 vLLM 0.26.0+cu129와 NIXL 1.4.1이다.
Native NIXL Worker와 Scheduler는 KV 예약, READ와 ACK를 유지한다.
Dense 확장은 별도 Flag로 Static Layer Span을 실행한다.
모델의 기존 Qwen Decoder Layer, Attention과 Norm 연산을 사용한다.
중간 Hidden과 Residual을 분리해서 전달한다.
양쪽 GPU는 전체 모델 Weight를 보유한다.
설치된 Runtime 파일은 수정하지 않는다.
원본 Hash와 실행별 확장 Source Archive를 보존한다.
[vLLM Disaggregated Prefill 문서](https://docs.vllm.ai/en/latest/features/disagg_prefill/).

## PP2 및 관련 시스템과의 경계

일반 PP2는 Weight를 Stage별로 나눌 수 있다.
이번 경로에서 D는 모든 Layer의 Decode를 실행한다.
P의 일부 Prefill만 D의 잔여 Layer 실행으로 연결한다.
이 차이만으로 성능 우위나 Novelty를 주장하지 않는다.

DuetServe는 SM 분할로 Prefill/Decode 간 간섭을 제어한다.
Bullet도 Spatial/Temporal 자원 공유를 사용한다.
두 연구는 Kernel Inactivity만 찾는 접근의 한계를 보여 주는 참고 설계다.
이번 PoC가 SM 격리나 Kernel 중간 선점을 구현했다고 주장하지 않는다.
[DuetServe](https://proceedings.mlr.press/v306/gao26m.html), [Bullet](https://arxiv.org/abs/2504.19516).

Libra는 Token 경계의 Micro-request와 Chunked KV 전송을 사용한다.
BanaServe는 Layer Weight와 KV의 동적 재배치를 설명한다.
따라서 Layer를 사용한다는 사실만으로 새 방법이라고 할 수 없다.
이번 구현과 기존 시스템의 전체 실행 동등성은 검증하지 않았다.
[Libra](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra), [BanaServe](https://arxiv.org/html/2510.13223v1).

## 로컬 Looped Reference

`ref_loopserve`에는 Device Controller, Depth KV와 Recurrent Scheduling 코드가 있다.
문서는 초기 UT0 공유 KV의 정확도 문제와 이후 Depth-indexed KV 복구를 구분한다.
그 Early-exit/Shared-core 경로를 Dense Qwen3에 적용하지 않는다.
이번 PoC는 Dense vLLM 모델과 검증된 Native NIXL 경로를 우선한다.
정확한 State Frontier와 Buffer Lifetime을 명시하는 원칙만 참고한다.

## 증거 파일

- [Pinned 공식 Layered Prefill 소스 Manifest](dense_ar_layerwise_pd_feasibility_20261112/references/layered_prefill/manifest.json)
- [설치 Runtime Source Hash](dense_ar_layerwise_pd_feasibility_20261112/state/installed_runtime_sources.json)
- [Runtime Package Version](dense_ar_layerwise_pd_feasibility_20261112/state/package_versions.json)
- [실험 Spec](dense_ar_prefill_heavy_layerwise_pd_feasibility_spec.md)

외부 코드의 검토와 실제 실행은 별개다.
최종 보고서는 실행하지 않은 Baseline을 `NOT_RUN`으로 남긴다.
