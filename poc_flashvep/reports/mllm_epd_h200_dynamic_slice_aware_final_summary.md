# H200 Dynamic Correctness와 Slice-aware Token Split

**Generic Dynamic은 `HOLD`다. 전송 최적화는 `TRANSPORT_OPTIMIZATION_ONLY`, 독립적인 Modality-aware Scheduling 효과는 `MODALITY_MECHANISM_ONLY`다.** QPS5 Main100의 B-new는 실제 한 요청에서 1,482개 토큰을 이전했으며 TTFT p95는 20.317 s, absolute Decode SLO는 75%였다. 동일 BI1·Preissue의 Native-Matched Stock Observer 값은 17.215 s와 75%였다. P/D BI1은 기존 혼합 정합성을 복구했고, Slice-only와 Descriptor 사전 발급은 반복 Clone payload와 Live Export RPC를 줄였다. [Active Online 결과](mllm_epd_h200_dynamic_split_active_online_serving.md) · [동일 Cut 전송 검증](mllm_epd_h200_same_cut_embedding_transfer_ablation.md)

## 필수 질문

| 질문 | 실측과 판정 |
|---|---|
| 1. 이전 Exact Greedy 7/8은 해결됐는가 | 두 기존 Mixed N8이 대칭 BI1에서 각각 8/8이 됐다. D Tensor와 Greedy가 일치했다. P 단독 BI1, D 단독 Split4, Eager의 일부 혼합 조건은 각각 남은 실패를 확인했다. |
| 2. Native A/A 86–88%의 원인은 무엇인가 | Batch, Query Geometry 및 Peer KV Context가 바뀌면 수치 재현성이 달라졌다. P 단독 BI1에서 Native-only Follower의 Index 23 차이를 같은 teacher prefix로 재현했다. Eager는 CUDA Graph 없이 Interior Index 13과 경쟁 Token ID를 재현했다. 단일 GPU Kernel의 인과 분해는 남아 있다. |
| 3. 같은 Geometry에서 Split 추가 오류는 있는가 | 대칭 BI1의 검증한 Cut에서 Full28 Import, Final State와 Greedy가 일치했다. 사전 발급을 포함한 10/20% N1/N8도 통과했다. Eager에서 30% Mixed 및 새 사전 발급 Mixed의 실패는 그대로 보존했다. |
| 4. 기존 Shadow는 왜 0건이었는가 | 콜백 자체가 제한됐고 QPS5 Natural B-old의 주요 거절은 작은 예측 Budget이었다. 이번 Free BI0 Stock의 전체 43개 후보 Cut은 모두 다음 P Row가 이미 발행되어 즉시 이전할 수 없었다. B-new는 실제 retirement hold와 비용·가용성 검사로 첫 선택을 실행했다. |
| 5. Cut-aligned D 여유가 있는가 | Free BI0 QPS5 중앙 후보 19개 중 D HTTP 및 Engine Queue가 동시에 비어 있던 Cut이 8개였다. 사전 발급 BI1은 QPS5 중앙에서 1개, QPS4에서는 0개였다. B-new의 Main100 Natural 여섯 Cohort에서 실제 이전 12개 중 중앙 구간은 1개였다. |
| 6. 160 ms와 80K tok/s는 지나치게 제한적인가 | 같은 시작의 Request-disjoint Free BI0에서 새 예측은 MAE를 258.10→102.44 ms로 줄였다. BI1의 QPS4 Heldout도 308.50→109.13 ms였다. 실제 QPS3 B-new에서 보수적 Budget을 넘은 1건이 있어, 상한 완화만으로 안전성을 확보할 수 없다. |
| 7. 개선한 B-new는 실제 얼마나 이전했는가 | 첫 Main100 Natural에서 QPS5 1/100 또는 1,482 tokens, QPS4 및 3은 각각 2/100 또는 3,234 tokens다. 별도 Counter 세 Cohort를 합하면 12/600 요청이다. |
| 8. 온라인 Decode Batching을 유지했는가 | QPS5 Counter의 Prepared Decode Batch 평균은 Native 2.648, B-new 2.640이었다. QPS4/3에서 Split Decode 52 step 중 Native와 공유한 것은 29/31 step이었다. QPS5 Counter의 실제 Split Decode 10 step은 단독 실행이었다. 별도 Counter의 `DIAGNOSTIC`이다. |
| 9. 같은 Cut의 Slice는 얼마나 절감했는가 | Crossing 38쌍에서 E NVLink TX 절감 중앙값 32.81 MiB였다. L7,823/n19는 D Clone 7,790→1 Row와 E TX 약 53.25 MiB→3–5 KiB를 기록했다. NIXL Bytes는 같았다. |
| 10. Slice 후에도 Boundary 장점이 남는가 | 인접 Cut 18쌍에서 Slice Interior의 추가 E TX는 중앙값 7 KiB였고, 추가 Clone과 Descriptor는 각 1개였다. 기존 Live Export RPC는 6개 실측에서 최대 2.47 s였고, 새 사전 발급 경로에서는 A와 B의 실제 이전에서 0 ms였다. |
| 11. 유용한 Workload 특성은 무엇인가 | 큰 Crossing Image에서 Slice의 물리 Payload 절감이 컸다. 7,823-token 단일 이미지 사례가 대표적이다. Eager A의 실제 이전 6건은 Long-document였다. Workload Stratum별 온라인 성능 우위는 미검증이다. |
| 12. Native보다 TTFT가 낮은가 | 같은 BI1/Preissue Native-Matched 대비 B-new TTFT p95는 QPS5 +18.0%, QPS4 −4.6%, QPS3 −0.2%였다. 5,269-token Replay의 B-new는 18.035 s, NativeStrong-Free Clean Before/After는 15.849/12.332 s였다. |
| 13. ITL·TPOT·Quality가 보호됐는가 | B-new Natural absolute Decode SLO가 QPS5/4/3에서 75/80/88%였다. QPS5 로컬 품질 점수 차이는 Mantis/DocVQA/MP-DocVQA에서 각각 0 pp, 결과 Token ID는 Native와 99/100 일치했다. 품질 비열등 CI와 MMLongBench 공식 채점은 미실행이다. |
| 14. G-Slice 대비 Modality 독립 효과가 있는가 | QPS4 Hard는 D가 HTTP idle일 때 B-new가 실제 사용한 같은 Request/Cut 두 개를 Image 내부라는 이유로 제외했다. Hard의 실제 이전은 0건이었다. Soft Feature는 예측 채택 기준에 미달했다. 독립적인 온라인 이득은 확보하지 못했다. |
| 15. 주장할 수 있는 Contribution은 무엇인가 | 같은 Cut의 물리 Clone Payload 절감, Live E Descriptor RPC 병목의 제거, strict BI1의 실제 D Suffix와 Native Decode 공동 실행, Cut-aligned 여유와 낮은 실행률을 분리해 확인했다. 전체 TTFT와 Decode를 동시에 개선한 정책은 이번 시험에서 확정하지 못했다. |

## 수치 재현성과 실제 D Suffix

- 기존 Interior의 Token Index 13은 BI1에서 후보 ID 1447과 13의 Logit이 각각 33.75로 같았다. 실제 D Prepared Batch 1과 2에서 전체 Logit 및 Final Hidden이 Byte-identical이고 두 실행이 ID 13을 선택했다.
- 기존 Boundary의 Native-only Follower는 P에만 BI1을 적용했을 때 같은 teacher-forced prefix에서 Index 23의 Greedy 차이가 재현됐다. ID는 32719와 9666, Top1–Top2 margin은 0.125와 0이었다. D도 BI1인 실행에서는 두 Follower 경로가 ID 32719로 일치했다.
- P/D BI1과 Slice를 결합했을 때 명목상 5/10/20/30%의 네 N1 및 Mixed N8 fixture가 기존 Exact 기준을 통과했다. 실현된 Suffix 비율은 7.22/12.81/24.84/32.03%였고, Native와 함께 Decode한 step은 60/49/51/32개였다.
- 같은 P의 Prefix를 D에 가져오는 N1에서 P에만 BI1을 켰을 때 Import Full28은 Byte-identical이었다. D Suffix 완료 뒤 P teacher와 Final KV가 같은 Layer는 0/28이어서 strict gate에 미달했다. [원인별 대조 및 첫 Divergence](mllm_epd_h200_native_numerical_variability.md)

## Transport 최적화와 Modality 비용

7,823-token 입력의 19-token Suffix에서는 D 이미지 Clone이 7,790개 Row에서 1개 Row로 줄었다. E NVLink TX는 약 53.25 MiB에서 3–5 KiB로, NIXL KV 전송은 두 경로 모두 447,741,952 bytes였다. 같은 모델과 Cut에서 이 개선을 측정했다. Slice 이후에도 인접 Cut 대조에서 Image Interior는 한 Clone과 한 Descriptor를 추가했다. 그 영향의 온라인 순이익은 별도 Transport × Partition 비교에서 다룬다. [전송 구현과 수명 계약](mllm_epd_h200_slice_aware_embedding_transfer.md) · [Transport × Partition](mllm_epd_h200_modality_transport_partition_ablation.md)

## 실제 Stock Cut의 D 여유와 병목

QPS5 BI0 Native Free의 중앙 Arrival 구간에서 P 내부 완료 70개 가운데 19개가 Dynamic 후보 범위에 있었다. 이때 D HTTP와 실제 Engine Queue가 동시에 비어 있던 Cut은 8개였다. 그중 6개는 Source 자신을 제외한 다음 P-ready까지 225 ms 이상 관측됐다. 이미 발생한 미래를 사용한 `OFFLINE_ORACLE`이며 후보를 실제 이전하면 다른 P 요청의 완료 시간이 달라질 수 있다. Stock 실행 전체의 후보 43개에서 P의 다음 Row가 이미 발행되어, 즉시 사용할 수 있는 no-hold Cut은 0개였다. [Cut의 P·D 상태](mllm_epd_h200_cut_aligned_actionable_slack.md)

이전 80K tok/s 휴리스틱은 같은 시작의 Request-disjoint 대조에서 다음 P-ready 예측 MAE 258.10 ms였다. 실제 P 진행·동시 Batch·Attention Work와 Queue를 사용한 대조는 102.44 ms였다. Native Free의 사후 Trace 및 BI0 isolated 비용을 결합한 대조에서는 160/225/300 ms 각 상한에서 선택이 0건이었다. 500 ms 또는 Adaptive와 point-gain으로 Discovery와 같은 시작의 Heldout에서 각각 1건이 fit했다. 모두 retirement hold가 필요한 Cut이었다. [Budget 및 비용 민감도](mllm_epd_h200_dynamic_budget_predictor_recovery.md)

## Figure와 재현 자료

| Figure | 실측 범위 |
|---|---|
| K1 | [수치 정합성](../mllm_epd_h200_dynamic_correctness_slice_aware/results/figures/k1_correctness.pdf) · [Native A/A](../mllm_epd_h200_dynamic_correctness_slice_aware/results/native_aa_geometry.csv) · [최초 Logit 차이](../mllm_epd_h200_dynamic_correctness_slice_aware/results/teacher_divergence.csv) |
| K2 | [실제 Cut과 채택 Funnel](../mllm_epd_h200_dynamic_correctness_slice_aware/results/figures/k2_cut_aligned_slack.pdf) |
| K3 | [Causal P-ready 예측](../mllm_epd_h200_dynamic_correctness_slice_aware/results/figures/k3_budget_prediction.pdf) |
| K4 | [Image Boundary 주변 물리 전송](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/figures/same_request_cut_sweep.png) |
| K5 | [동일 Cut Whole/Slice](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/figures/same_cut_transfer_ablation.png) |
| K6 | [Transport × Partition의 측정한 두 Cell](../mllm_epd_h200_dynamic_correctness_slice_aware/results/figures/k6_partition_transport.pdf) |
| K7 | [QPS5 Natural TTFT와 Token ITL CDF](../mllm_epd_h200_dynamic_correctness_slice_aware/results/figures/k7_online_serving.pdf) · [실제 Prepared Batch](../mllm_epd_h200_dynamic_correctness_slice_aware/results/active_decode_geometry.json) |
| K8 | [Stock Cut 시점의 실제 Queue](../mllm_epd_h200_dynamic_correctness_slice_aware/results/figures/k8_request_timeline.pdf) · [실제 Export RPC 경로](../mllm_epd_h200_dynamic_correctness_slice_aware/results/figures/k8_active_export_timeline.pdf) |

동결 Gate, Model 및 데이터 계약은 PoC Spec에 있다. 실제 명령, Source/Runtime Hash, GPU UUID, Topology, Arrival Seed와 Output Counts는 실행 Receipt와 Raw 디렉터리에 남겼다. 공개 Commit의 Figure는 포함한 경량 CSV/JSON에서 재생성할 수 있다. Full28 Tensor, 원본 모델과 이미지는 서버에 보존했다. [PoC Spec](mllm_epd_h200_dynamic_correctness_slice_aware_poc_spec.md) · [실행 Receipt](../mllm_epd_h200_dynamic_correctness_slice_aware/results/campaign_execution.json) · [재현 명령](../mllm_epd_h200_dynamic_correctness_slice_aware/README.md)

독립 N3 및 새 Main400 Heldout은 미실행이다. 실제 Observed Counter와 Stock Scheduler 타임라인은 `DIAGNOSTIC`, 이전량과 고객 지연은 검증된 Native/Candidate 온라인 Cohort, 미래 D Idle 추정은 `OFFLINE_ORACLE`로 보존했다.

QPS5 Replay의 NativeStrong-Free CUPTI Collector는 데이터를 수집했고 기존 5% Observer 기준은 실패했다. E/P/D Kernel Busy 90.79/94.21/60.26%는 `DIAGNOSTIC`이다. 같은 출력 계약에서 NativeStrong-Free의 absolute Decode SLO는 두 Clean Cohort에서 98/95%, B-new 첫 Cohort는 73%였다. [강한 Native 대조와 Collector Gate](../mllm_epd_h200_dynamic_correctness_slice_aware/results/native_strong_replay_profile.json)

## 실행 범위와 종료 상태

H200 GPU4/5/6을 E/P/D로 사용했다. 기존 Python 3.14.6, vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129 및 원본 Qwen2.5-VL-7B-Instruct를 재사용했다. GPU 사용 중 바깥 다섯 GPU의 기존 Burner를 유지했고 공유 자원 간섭을 독립 비교하지 않았다. 이번 실험의 Worker를 정리한 뒤 GPU0–7 모두 소유권을 확인한 `train_lm.py` Burner가 실행 중이었다. 최종 CPU 회귀 검사는 59개 통과했다. [GPU 복원 증거](../mllm_epd_h200_dynamic_correctness_slice_aware/results/burner_restore.json)

초기 Stock Hook의 Scheduler 호출 시그니처 문제와 종료 시 Trace Flush 누락은 각각 별도 기록을 남기고 수정했다. 첫 Eager 비용 런처는 Overlay 초기화 전 모듈 Import 때문에 실패했고, 새 시작에서 335개 실제 Trial을 완료했다. 최초 BI0 Slice-only 전송의 별도 Natural 검증은 Stage 간 EOS 계약 불일치로 중단됐다. 동일 Cut의 BI0 전송 60쌍과 이후 BI1의 Full28 및 Greedy 표를 각각 보존했다. Whole 두 Arm의 Active C3, QPS6, 독립 N3 및 공식 전체 Task Quality는 미실행이다.
