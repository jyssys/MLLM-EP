# H200 Dynamic Token Split 연구 계약

기준 Commit은 `a53d707e8882c3bd5f770a02708876384c785747`다. 이전 캠페인의 7/8 Greedy 결과와 Shadow 0건을 그대로 보존한다. 이번 브랜치에서 정합성 복구, 실제 Cut에 정렬된 D 여유와 Predictor, 같은 Cut의 전송 효율을 각각 검증한다. [직전 종합 보고서](mllm_epd_h200_modality_aware_dynamic_split_final_summary.md) · [Notion 연구 계획](https://app.notion.com/p/4220ee8277e48153a020e374317777f8)

## 실험 조건

| 항목 | 동결 조건 |
|---|---|
| Hardware | H200 3장, E4/P5/D6 |
| 모델 | Qwen2.5-VL-7B-Instruct, `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16, Full28, 원본 MRoPE/Vision Encoder |
| P / D | B8192/S1024, KV42GiB/GPU, Block128, prefix cache off |
| E | B114688, Renderer2, MM Cache0, sync scheduling |
| Decode | B8 primary, Free strong baseline, B1 및 별도 커널 설정은 수치 진단 |
| Workload | Main100 Natural EOS cap2048, Equal-output Replay 5,269 tokens, Calibration32 정합성 및 비용 보정 |
| Load | QPS5 primary. QPS4/6 확장은 QPS5에서 정합성과 성능 확인 뒤 수행 |
| Metric | TTFT ≤5 s, 모든 Client ITL ≤34 ms, TPOT ≤33 ms/token |
| Transport | 공통 inline-redirect 유지. Whole-image와 slice-only 비교는 동일 요청·Cut·KV를 사용 |

정합성은 원본 Vision Embedding, Token Order, MRoPE, 실제 Query Coverage, Full28 prefix/import, Final KV tolerance 및 exact Greedy를 확인한다. 동일 Scheduler의 Native-only 동작도 비교한다. 제어된 geometry 증거와 일반 mixed 경로의 strict gate는 각각 보존한다. Active C3는 사용할 수치 설정, Cut 방식, Ratio 및 Transport에서 정합성을 입증한 후보로 수행한다.

## 실험 순서와 판정

1. Serial, Batch1, P/D 같은 Geometry 및 Continuous Batching에서 A/A를 비교한다. 이전 실패 시점까지 같은 출력 prefix를 주고 최초 divergent Argmax를 추적한다. BI1이나 eager 같은 옵션은 명시적인 별도 조건으로 기록한다.
2. Stock P Chunk 완료와 실제 D 상태를 같은 시계로 기록한다. 정책은 현재까지 완료된 관측만 읽는다. Cut 직후의 실제 idle 길이는 offline audit에서만 사용한다. 기존 80K tok/s, 160 ms와 새 calibration을 같은 관측으로 비교한다.
3. `0/1/2.5/5/10/20/30%` 및 `64/128/256/512/1024/2048/4096` 후보는 실제 P progress에서 실행 가능한 시점에만 평가한다. D에 active/pending decode가 없고 필요한 KV가 있으며 예측된 점유 시간과 margin이 예산 이내이고 순이익이 양수일 때만 예약한다. Native 완료 요청은 일반 continuous batching 경로를 사용한다.
4. 동일한 Cut으로 D의 whole-image clone과 slice-only clone을 비교한다. P가 이미 whole-image를 가져온 비용을 그대로 포함한다. CUDA IPC의 producer sync, consumer clone 완료, ACK 순서를 지킨다. Logical bytes, 실제 link counter, KV 전송을 각각 기록한다.
5. Generic과 modality-aware의 정책을 Whole과 Slice에서 각각 비교한다. G-Slice와 M-Slice의 공통 scheduler, candidate search와 runtime을 고정한다. Slice로 비용이 사라지면 전송 최적화 효과의 범위까지만 결론을 내린다.
6. 통과 후보를 B8 QPS5 Natural EOS로 스크리닝한다. 성공 후보는 최소 독립 Server Start N3로 B8 및 Free Native와 비교하고 5,269-token Replay를 따로 확인한다. 각 짝의 GPU, arrival trace, request IDs, warm-up과 output 계약을 맞춘다. Discovery에서만 parameter를 조정한다.

Generic의 목표는 가장 강한 Native 대비 TTFT p95 ≥5% 감소와 개선 방향의 paired 95% CI, TTFT p50 비열등이다. ITL p99와 TPOT p95 각 ratio의 paired CI 상한은 1.05 이하다. 요청별 absolute Decode SLO ≥95%, 품질 비열등, Continuous Batching 유지 및 joint SLO/goodput 보호가 필요하다. M-Slice는 G-Slice 대비 직접 측정된 비용 감소와 C3의 추가 개선을 보여야 한다. Goodput만 5% 개선된 결과는 보조 목표로 기록한다.

Profiler의 5% Clean/Profile/Clean 기준은 유지한다. 실패한 Trace는 `DIAGNOSTIC`에 남긴다. Correctness, online 성능, trace qualification, 품질은 독립 근거를 사용한다. Raw tensor는 로컬 실행 디렉터리에 두고 재현용 digest, 통계, source와 표를 이 브랜치에 보존한다.
