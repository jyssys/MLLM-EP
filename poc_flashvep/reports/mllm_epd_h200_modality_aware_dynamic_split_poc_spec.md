# H200 MLLM EPD Dynamic Split PoC

이 캠페인은 두 가설을 독립적으로 검사한다. 첫째, P의 실제 prefill 경계에서 일부 작업을 D로 이전하면 Native 수준의 continuous batching과 Decode SLO를 유지하면서 TTFT를 줄일 수 있다. 둘째, 공통 실행기에서 이미지 정보를 활용하면 가장 강한 Generic Dynamic보다 추가 시스템 비용 또는 온라인 지연을 줄일 수 있다.

## 동결한 실험 설계

| 항목 | 계약 |
| --- | --- |
| 베이스라인 | NativeStrong-B8, NativeCommon-B8, NativeStrong-Free, NativeCommon-Free, 기존 Idle10. Static10은 이전 실패 및 admission 인과분석에 사용 |
| 방법 A | 기존 P forward가 끝난 실제 경계에서 약 10% 잔여 작업의 분할 여부를 판단. 예약을 거절한 요청은 같은 P 요청과 KV로 계속 계산 |
| 방법 B | 같은 실행기에서 0/5/10/20/30%와 실행 가능한 잔여 토큰을 검사. 필요량에는 KV/Embedding handoff, D 실행 및 관측된 제어 비용을 한 번씩 반영 |
| 방법 C | B와 같은 탐색 및 학습 예산, D admission, 예약, fallback, 안전 여유 및 compiler 사용. 우선 같은 candidate set에 feature-off, Hard, Soft를 적용. 추가 이미지 경계 후보 자체의 효과는 별도 비교 |
| 워크로드 | 동결 Main100 이미지 질문. Natural EOS Output Cap2048과 요청별 Equal-output Replay 5,269 tokens를 별도 실행. Calibration32와 이전 Discovery50/Heldout50 분리를 유지 |
| GPU 및 설정 | 동일 1E+1P+1D H200. Primary B8 QPS5, 강한 Free, 후속 QPS4/6. P/D B8192·S1024·KV42GiB·Block128. E B114688·Renderer2·MM Cache0·Sync |
| 모델 | Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5. BF16, BI0, Full28, 원본 Vision/Processor, Greedy, 원본 MRoPE, exact causal attention, prefix cache off |
| 예약 | D active, pending, suffix 예약, KV 수용량, 이전 진행 상태를 한 절차에서 검사. Suffix 시작 전 Native 우선권을 유지하고 active decode 중 suffix를 삽입하지 않음 |
| 검증 순서 | 후보별 정확성 및 실패 정리 → N1 clean screening → 유망 조건에서 독립 paired N3 이상과 고정 정책 Heldout |
| 비교량 | 실제 D batch 분포와 weight 기준, token/step/time-weighted mean, TTFT, ITL, TPOT, Absolute/Joint Decode SLO, SLO-goodput, 정확성 및 태스크 품질 |

### 확정 판정 기준

- 기존 SLO는 TTFT ≤5,000 ms, 전체 Client ITL ≤34 ms, TPOT ≤33 ms/token이다. Absolute Decode SLO는 ITL 및 TPOT를 함께 충족한 요청 비율로 계산한다.
- Generic의 성공에는 강한 Native 대비 TTFT p95 5% 이상 감소와 개선 방향의 paired 95% CI, p50 비열등, ITL p99 및 TPOT p95 비율의 paired CI 상한 ≤1.05, Absolute Decode SLO ≥95%, batching·품질·goodput 보존이 필요하다. N3의 통계 단위는 서버 시작이다.
- Modality의 추가 성공에는 B 대비 직접 실측한 시스템 비용 감소와 TTFT p95 5% 이상 추가 감소 또는 SLO-goodput 5% 이상 추가 증가, 개선 방향의 paired 95% CI 및 동일 보호 조건이 필요하다. Goodput 단독 이득은 secondary로 남긴다.
- Motivation은 Clean/Profile/Clean 5% 관찰 조건과 P Busy ≥80%, 기존 P Capacity 근접도 ≥85%, 실제 P backlog, E starvation 배제, 160 ms 이상 D quiet window 두 개 및 P busy 대비 교집합 ≥10%, 32,768-token D KV 여유, Absolute Decode SLO ≥95%로 평가한다. 불충족 trace는 진단 수준을 유지한다. Clean 온라인 파일럿은 별도 증거다.
- A/B는 같은 경계에서 10% 목표와 동적 후보의 효과를 분리한다. C의 modality 효과는 feature-off, work matching, 동일 cut transfer/layout, block-alignment, fake-boundary에서 원인을 확인한다. Soft penalty는 측정된 추가비용이 있을 때만 적용한다.
- 실제 suffix 실행이 없거나, batch1 붕괴와 Admission stall이 재발하거나, 정확성이 깨지면 해당 조건의 결과와 미실행 이유를 보존한다. CPU 분석이나 compiler job은 clean latency 구간과 분리한다.

실제 D suffix와 Native follower가 co-decode한 N8 두 조건의 전체 Greedy는 모두 7/8이었다. 그래서 active A/B/C의 timed C3는 보류했다. 별도 QPS5 shadow 실행은 실제 P/D 상태와 후보 비용을 검사하고 예약 가능성만 기록한다. 이 실행에서는 P가 전 구간 prefill을 완료한다. Shadow의 TTFT 및 Decode는 실제 full-prefill control의 계측이고, suffix가 작동한 경우의 수치는 제공하지 않는다.

## 데이터와 계보

원본 Main100 SHA256은 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1이다. 복원 시 이미지 절대 경로가 달라지므로 실행 manifest hash와 원본 hash를 각각 기록한다. 기존 Heldout50은 이전 연구에 나타난 cohort다. 이 캠페인의 새 context-disjoint Main400 결과가 있을 때 별도로 보고한다. [Main400 dataset 설계](mllm_prefill_heavy400_dataset_profile.md)

시작점은 55870ccafb210d32dc956320d02226d26f2efee6이며 브랜치는 flashvep/mllm-epd-h200-modality-aware-dynamic-split이다. 이전 H200 e1a31767ef3c9653e7a61b5a9d8da52ad0e4baee, H100 38e67f91f0099dc546d470c83ff03af638f7af16 및 인수인계 449c3e1873c65206435a6bb9411a889a62e55be7을 역사적 증거로 유지한다. [직전 H200 보고서](mllm_epd_h200_wide_decode_batch_final_summary.md) · [직전 H100 보고서](mllm_epd_prefill_heavy_token_split_final_summary.md)

2026-11-26 Notion 승인 내용은 B8/Free, late-bound idle-only, 0/5/10/20/30%, 동일 실행기의 모달리티 feature ablation 및 추가 5% 문턱을 명시한다. [연구 의사결정](https://app.notion.com/p/4220ee8277e48153a020e374317777f8) · [MLLM-EP 실험 아카이브](https://app.notion.com/p/1062a9a81d8641f895f0bdf6c0702489)

## 증거와 Figure 계약

실제 온라인, 독립 GPU 측정, 적격 trace, 진단, 오라클, 정확성 통과, 미측정, 미실행을 각각 MEASURED_ONLINE, ISOLATED_GPU_MEASUREMENT, TRACE_ANALYSIS, DIAGNOSTIC, OFFLINE_ORACLE, CORRECTNESS_PASS, NOT_MEASURED, NOT_RUN으로 기록한다. 과거 수치와 새 실행은 같은 표에서도 각각의 run과 fingerprint를 표시한다.

| Figure | 등록한 관측 |
| --- | --- |
| 1 | E/P/D kernel busy, P backlog, D quiet. Trace 적격 여부와 각 시간 분모 명시 |
| 2 | 실제 prepared D batch 분포. 이전 Static10과 Native를 표시하고 새 A/B/C는 실행 증거가 있을 때 추가 |
| 3 | 실제 suffix cost를 suffix tokens 및 context별로 표시. Host와 GPU-exclusive boundary 구분 |
| 4 | Intra-image와 boundary의 work, bytes/copies/descriptors/lifetime 및 suffix-ready 시간 |
| 5 | 같은 학습 split을 사용한 Generic/Modality heldout 오차와 false-safe |
| 6 | Single/Few/Multi/Long-document 길이 분포와 strata 내 정책 차이 |
| 7 | A/B/C의 실제 offload ratio와 0% |
| 8 | Hard/Soft의 safe-fit, fallback, 비용과 TTFT |
| 9 | Native/Dynamic/Modality의 TTFT CDF 및 Decode 지표 |
| 10 | 동일 실행기 B/C feature ablation |
| 11 | 양쪽에서 관측된 같은 요청의 image span, cut, 예약, 실제 시간 구간 |
| 12 | 실제 결정 및 실패 원인 분해 |

[종합 증거 현황](mllm_epd_h200_modality_aware_dynamic_split_final_summary.md)

## 결과 재현

저장소 루트에서 RAW_RUNS를 실행 디렉터리로 지정한다. RESULTS는 새 결과 디렉터리다. Figure 6의 REPLAY는 앞 캠페인의 동결 5,269-token 표다. 이 표를 byte 그대로 보존했다. Figure 3–6은 저장한 results를 --analysis로 설정해서도 재생성할 수 있다. 원본 tensor, 모델, 이미지와 사용자 자격증명은 결과 묶음에 포함하지 않는다. [Frozen replay 표](../mllm_epd_h200_modality_aware_dynamic_split/results/frozen_main100_replay_reference.csv)

```bash
package=poc_flashvep.mllm_epd_h200_modality_aware_dynamic_split
wide=poc_flashvep/mllm_epd_h200_wide_decode_batch_token_split/results
python -m "$package.execution_receipt" --runs "$RAW_RUNS" --download-receipt "$DOWNLOAD_RECEIPT" \
  --asset-repo "$ASSET_REPO" --output "$RESULTS/campaign_execution.json"
python -m "$package.online_evidence" --runs "$RAW_RUNS" --output "$RESULTS"
python -m "$package.correctness_evidence" --runs "$RAW_RUNS/correctness_late_v1" \
  --output "$RESULTS/dynamic_correctness.json"
python -m "$package.correctness_evidence" --runs "$RAW_RUNS/natural_nohold_correctness" \
  "$RAW_RUNS/natural_hold_correctness" "$RAW_RUNS/natural_hold_inline_correctness" \
  "$RAW_RUNS/natural_nohold_inline_observed" "$RAW_RUNS/natural_hold_inline_mixed" "$RAW_RUNS/natural_hold_lead_control" \
  --output "$RESULTS/dynamic_natural_correctness.json"
python -m "$package.calibration_figures" --analysis "$RAW_RUNS/suffix_calibration/analysis" \
  --replay "$REPLAY" --output "$RESULTS/figures"
python -m "$package.calibration_figures" --analysis "$RESULTS/inline_calibration" \
  --inventory "$RESULTS/modality_inventory.json" --output "$RESULTS/inline_calibration/figures"
python -m "$package.historical_figures" --wide "$wide" --motivation "$RESULTS/dynamic_motivation.json" \
  --correctness "$RESULTS/dynamic_correctness.json" --natural-correctness "$RESULTS/dynamic_natural_correctness.json" \
  --output "$RESULTS/figures"
python -m "$package.online_figures" --online-analysis "$RESULTS" --output "$RESULTS/figures"
python -m "$package.quality" --runs "$RAW_RUNS" --output "$RESULTS"
```

Motivation JSON은 이전 H200 isolated P capacity 영수증을 지정해 기존 motivation module로 B8/Free raw trace에서 만든다. Figure의 현재 결과와 과거 reference는 다른 시작 및 서로 다른 증거 scope로 기록한다. 출처 hash는 Figure data JSON에 있다.

실행 기록은 실측 start/end/return code, runtime, 명령, GPU UUID 및 topology, 모델 revision과 원본/relocated manifest의 별도 재검증을 저장한다. 프로세스의 정상 종료와 실험 gate 결과는 각각의 receipt에서 확인한다. [캠페인 실행 기록](../mllm_epd_h200_modality_aware_dynamic_split/results/campaign_execution.json)
