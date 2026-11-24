# E/P/D Elastic Prefill: 후속 구조의 실행 가능성

증거 수준은 STRUCTURAL_ACCOUNTING이다. 아래 네 구조의 Online Speedup은 측정하지 않았다.
이 문서는 이미 측정한 자원 조건과 Causal Dependency를 바탕으로 후속 구현의 요구사항을 구분한다.
EAServe의 E Local Prefill이나 HydraInfer의 Stage 배치를 새로운 기여로 주장하지 않는다.
선행연구의 모델·하드웨어·구현 범위는 [별도 감사](mllm_epd_related_work_and_novelty_audit.md)에 있다.

## 사용할 수 있는 실제 근거

적격 C25 저부하에서 E/D의 동시 Kernel Idle을 관측했다.
중앙 Arrival 구간 전체의 74.237585%다. P Busy 중 교집합은 P Busy 시간의 71.043991%다.
그러나 P Kernel Busy는 14.068244%였다. 이 조건은 Prefill-heavy가 아니다.
고부하에서 E 비병목·P 포화·실행 가능한 E/D 여유의 동시 조건은 아직 미확정이다.

C16에서 원본 Full Model을 E에 추가로 적재했다. 추가 Helper KV는 8GiB였다.
실제 Encoder 입력 처리 후 Device Free는 43,844,042,752 bytes였다.
이 측정은 Weight 적재와 제한된 공존 입력의 Memory Fit이다.
최대 Encoder Batch Capacity나 Online Prefill 안전성을 증명하지 않았다.

Generic Token Split의 보수적 Static10 C3에서는 D 입장 대기가 증가했다.
Main100 Replay의 NativeCommon 대비 P HTTP Enclosure 차이는 평균 −305.329ms였다.
Coordinator D 대기 차이는 평균 +971.956ms였다.
평균 TTFT 차이는 +933.063ms였다. Paired Start-level 95% CI는 [817.498, 1020.626]ms다.
이는 같은 요청·Arrival의 독립 N3 Host 경계 분석이다. GPU-exclusive 비용이나 p95의 가법 분해가 아니다.
따라서 새 구조도 계산량 감소만으로 Serving 이득을 예측하면 안 된다.

## 구조별 비교

| 후보 | 실행 단위 | 필요한 선행 상태 | 통신·동기화 | E/D 원래 업무의 위험 | 현재 증거 |
|---|---|---|---|---|---|
| A. Request-level Elastic Offloading | 서로 다른 요청의 Full Prefill | 해당 요청의 실제 E 완료, LLM Weight·KV 여유 | E 측 Prefill은 완료 KV를 D에 전달해야 함. D 전체 Deflection은 Prefix KV Handoff가 없음 | 새 Encode 또는 Decode가 비선점 Prefill을 기다릴 수 있음 | Whole-request D 대조군과 E Memory Fit은 별도 측정. E/P/D 통합 Offload는 NOT_RUN |
| B. Inter-request Token Pipeline | 서로 다른 요청의 Contiguous Chunk | 같은 요청의 이전 Chunk KV와 현재 Embedding·MRoPE | Chunk 사이 KV·Embedding Handoff와 Buffer ACK | Stage 간 Queue와 Credit이 Pipeline을 막을 수 있음 | Exact P/D Token Continuation은 일부 조건에서 정합성 통과. E 포함 3단 Token Pipeline은 NOT_RUN |
| C. Layer-wise Wavefront | Token Chunk의 Layer별 진행 | 같은 Layer에서 필요한 모든 앞 Chunk KV, 현재 Chunk의 이전 Layer Hidden | Layer별 KV Ready 통지·전송, 반복되는 의존성 경계 | 짧은 E/D 여유가 Layer Barrier 때문에 유효 계산으로 이어지지 않을 수 있음 | Computational Dependency 분석만 수행. 실제 Wavefront는 NOT_RUN |
| D. Elastic Context Parallelism | 같은 Layer의 Token 연산을 여러 GPU가 분담 | 참여 GPU의 Weight·KV·공통 Attention Metadata | Layer별 필요한 KV 교환과 최종 D KV 배치 | 참여·이탈의 Barrier와 Attention 통신이 Encode/Decode Deadline을 침해할 수 있음 | 실제 동적 Context Parallel 실행과 비용은 NOT_RUN |

Window가 겹친다고 같은 요청을 독립 계산할 수 있는 것은 아니다.
각 후보는 필요한 Weight·KV·Hidden·Position State와 통신 완료를 모두 만족해야 한다.
추가 Weight 적재가 가능하다는 사실도 해당 GPU가 업무를 중단 없이 수행한다는 뜻은 아니다.

## Causal 순서

Contiguous Token Continuation은 앞 Token의 KV를 사용한다.
모든 Prompt Token은 첫 Output 전에 필요한 전체 LLM Layer를 완료해야 한다.
Visual Span 내부 Cut도 Token 순서와 원래 MRoPE·Image Grid 대응을 보존해야 한다.

Wavefront에서 뒤 Chunk가 Layer i를 실행하려면 앞 Chunk들의 Layer i KV가 필요하다.
앞 Chunk의 모든 Layer 완료를 기다리는 것과는 다른 의존성이다.
하지만 현재 Chunk의 Layer i 입력 Hidden은 Layer i−1의 결과를 필요로 한다.
이 구조가 Causal Attention의 필요한 Prefix KV를 생략할 수 있다는 뜻은 아니다.

Context Parallelism은 필요한 Attention KV 교환을 생략할 수 없다.
E/D의 참여 기간, 통신 완료, 최종 D Cache 소유권을 같은 Deadline 계약에서 검사해야 한다.
현재 Kernel Gap만으로 그 Layer별 통신과 Barrier의 비용을 상쇄한다고 가정하지 않는다.

## 권장 검증 순서

1. 실제 Workload에서 E 비병목·P Queue·E/D 안전 여유의 동시 조건을 먼저 검증한다.
2. 동일한 강한 Native와 Runtime Policy에서 Request-level Offloading을 검사한다.
3. 실제 요청 완료에 기여하는 E 측 최소 Prefill을 구현한 뒤 Decode와 Encoder 보호를 함께 검사한다.
4. 추가 Handoff와 Queue가 줄지 않으면 3단 Token Pipeline으로 확장하지 않는다.
5. Layer별 통신 비용과 충분한 Window가 확인된 뒤에만 Wavefront나 Context Parallelism을 검토한다.

이 순서는 복잡도가 낮은 검증부터 수행하자는 제안이다. A의 성능 우위를 입증한 결론은 아니다.
EAServe-inspired 또는 Kairos-inspired라는 표현은 공식 구현 재현과 구분해야 한다.
Probe가 실제 Client 요청을 완료하지 않으면 Throughput 개선으로 보고하지 않는다.
Future Idle Duration이나 실제 EOS를 사용한 선택은 OFFLINE_ORACLE로 분리한다.

## 우선 남은 측정

- 실제 E 추가 Prefill의 Encoder Service·Queue·E2E 영향.
- 실제 D 추가 Prefill의 ITL·TPOT·입장 대기 영향.
- Imported Prefix KV가 있는 Suffix와 없는 Prefix의 별도 비용.
- 같은 Kernel Backend·Token Geometry에서 E/D의 실행 비용.
- 실제 통신 완료와 Buffer Lifetime을 포함한 Tail Cost.
- 고부하의 적격 Kernel/Queue 계측과 P Service Proximity.

독립적인 Encoder-exclusive·Decode Protection Gate 없이 ENCODE_SAFE_HEADROOM_CONFIRMED를 부여하지 않는다.
현재 구조적 분석만으로 GO_DUAL_SIDE_EXPLORATION이나 실제 3GPU Serving 이득을 선언하지 않는다.
후속 최종 판정은 Online Intervention과 전체 Task Quality의 한계를 함께 반영해야 한다.

## 관련 실측

- [Prefill 병목과 Idle의 증거 수준](mllm_epd_prefill_bottleneck_and_decode_idle.md)
- [E와 추가 Full LLM의 Memory Fit](mllm_epd_encoder_llm_memory_diagnostic_v1.md)
- [보호형 Token의 독립 N3](mllm_epd_protected_token_confirmation_v1.md)
- [요청별 D 입장 대기와 TTFT](mllm_epd_protected_token_critical_path_v1.md)
