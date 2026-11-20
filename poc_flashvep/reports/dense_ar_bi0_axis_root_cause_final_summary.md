# BI0 Layer/Token Root-cause Campaign

상태: `CLOSED / HOLD`. 원인 분해와 사전 Confirmation을 마쳤다.
Current BI0의 반복 가능한 Layer-specific Serving 우위는 확보하지 못했다.
GPU 작업은 종료했다. 미실행 범위와 적격성 실패를 아래에 구분한다.
Base Commit은 `12e0f370a26ec544af81a90577874527ce8d4c5b`다.
새 Branch는 `flashvep/dense-ar-bi0-axis-rootcause-20261120`다.
기존 사용자 Checkout, Index와 외부 Runtime을 변경하지 않는다.

## 핵심 원인

Layer의 D 경로가 짧아도 Online TTFT가 반드시 짧지는 않다.
Current LongBench Last4에서 추가 P Queue가 D 경로 절감을 초과했다.
기존 Natural Start의 Mean P Queue 차이는 +2711.089ms다.
그중 +2706.135ms는 이전 Source 체인의 누적이다.
직접 Credit Block 차이는 −1.041ms였다.
이 Source 체인을 전부 GPU 계산시간으로 표현하지 않는다.

Source 체인을 순서대로 다시 풀면 P Host Service 누적은 +2876.210ms다.
그중 Dispatch→Prepare 누적은 +538.177ms다. Prepare→Worker Ready 누적은 +2339.260ms다.
Credit Extension 누적은 −62.619ms다. 실행 가능한 P의 Host Gap 누적은 +4.718ms다.
Arrival Gap 등의 상쇄를 포함하면 실제 Queue 차이2711.089ms와 정확하게 일치한다.
이 조건은 Credit 거절이 주원인이라는 설명보다 P Service 누적이라는 설명을 지지한다.

별도 적격 Fixed-output Profile에서 Layer의 P Kernel은12.764ms 더 길었다.
식별한 Attention Kernel은14.724ms 더 길었다.
P Attention 차이는 실제 Source Service 비용의 중요한 근거다.
하지만 이 Profile을 과거 Natural Queue의 정확한 인과 기여율로 바꾸지 않는다.

Historical Input8000/Output16·Last8의3개 Paired Start도 다시 감사했다.
Mean TTFT 차이−494.968ms와 Ready→D 차이−824.402ms가 일치했다.
그 대기의 대부분은 이전 D 요청의 준비·Decode·Delivery Guard와 겹쳤다.
이 중343.801ms는 Layer의 P Ready가 늦어져 대기가 짧아진 부분이다.
실제 D Submit은480.601ms 빨랐다.
정확한 FIFO 체인에서 이전 D 준비 구간의 누적 차이는 −537.264ms였다.
이전 Decode Residency의 누적 차이는 오히려 +43.140ms였다.
따라서 Guard 겹침을 실제 Decode 계산의 인과 효과로 해석하지 않는다.
Historical Runtime과 Current Runtime의 같은 Workload Bridge도 완료했다.
BI0/BI1 결과는 같은 Primary 성능 표에 합치지 않는다.
같은 Current Input8000/Output16 진단의 후속 Worker Interval은 BI0 약7.2ms, BI1 약23.0–23.2ms였다.
각 조건은1회다. Actual Engine Option·Query Shape·핵심 Source는 일치했다.
Dispatcher 적격성 Metadata의 차이도 별도로 보존했다. GPU-exclusive Decode 비용은 아니다.
BI1의 큰 D 부담을 BI0에도 그대로 가정할 수 없다는 근거다.

Historical Layer는 Token보다 빨랐지만 같은 Native보다 TTFT p95가33.1% 느렸다.
Layer/Native TTFT 비율의95% CI는 [1.315488, 1.346452]다.
당시 Native도 D Host Demand가 P보다 길었다.
따라서 Historical Winner는 두 Offloading Axis 사이의 승리다.
Native의 Prefill-heavy 병목을 개선한 사례로 사용하지 않는다.

같은 Input8000/Output16·Last8의 Current BI0/KV128 독립3회도 완료했다.
Layer/Token TTFT p95 비율은1.160249다.95% CI는 [1.088786, 1.249619]다.
SLO-goodput 비율은0.962092다. CI는 [0.952115, 0.975201]다.
두 Axis 모두 Current Native 대비 TTFT를 줄였고 Decode5% Gate를 통과했다.
그러나 Token이 더 빨랐다. Historical Layer 우위를 Current BI0 결과로 재사용할 수 없다.

## Full200의 독립3회 Confirmation

Full200, QPS2.35, BI0, 실제 Batch1/1/1, Credit4, Hidden Slots4와 KV128을 사용했다.
독립 Arrival Seed2071/2072/2073으로 각 Axis를3회 실행했다.
Last2/4/8/12를 모두 같은3개 Seed에서 반복했다.
다음 비율은 Layer/Token이다. 각 CI는27개 Paired Start Bootstrap 재표본을 사용한다.
독립 Start3회의 Tail 정밀도는 제한적이다. Request나 Token을 독립 Start로 계산하지 않았다.

| Cut | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | 판정 |
|---|---|---|---|
| Last2 | 1.154544 [1.073808, 1.240517] | 0.966456 [0.930078, 0.996872] | Token 우세 |
| Last4 | 1.259430 [1.148548, 1.412117] | 0.186642 [0.090714, 0.285802] | Token 우세 |
| Last8 | 1.235989 [1.139479, 1.312967] | 0.785769 [0.743145, 0.874830] | Token 우세 |
| Last12 | 0.959758 [0.884393, 1.068346] | 0.966035 [0.831489, 1.043561] | 불확실 |

두 Axis 모두 모든 Cut의 Native-relative Decode5% Gate를 통과했다.
ITL p99 CI 상한의 최댓값은1.019911이다. TPOT p95는1.011740이다.
Full200 Task-quality CI와 실제 Batch1/1/1 검사도 통과했다.
그러나 Layer의 추가 TTFT·Goodput 우위는 이 결과로 확정하지 못했다.
모든 Method의 Joint SLO 평균은95% 미만이었다.

같은 Start들의 Layer D First Host Service는 모든 Cut에서 Token보다 짧았다.
반대로 P Host Service와 P Queue는 모든 Cut에서 길었다.
Last8에서 P Service는 Layer430.540ms, Token394.101ms였다.
D First Service는162.289ms와180.053ms였다.
P Queue 차이+1203.598ms는 Ready→D Prepare 절감61.767ms보다 컸다.
이 Host 경계는 Exclusive GPU Forward가 아니다.
원래 Last4/Seed2061의2711ms와 새 Last8 평균을 하나의 합계로 사용하지 않는다.

## 추가 연구 질문의 직접적인 답

**Q1. Historical Layer 승리의 원인은 무엇인가?**

관측된 주된 항은 이전 D 준비 Host 비용의 누적 차이−537.264ms였다.
Whole-request Guard 아래에서 이 차이가 D Submit 시각에 누적됐다.
D GPU GEMM만의 효과인지, Guard를 제거하면 얼마가 남는지는 미식별이다.
같은 Historical Native보다 Layer TTFT가33.1% 느렸다는 제한도 있다.

**Q2. Current BI0 Token 승리는 P Attention과 Queue 때문인가?**

추가 Source Host Service의 Queue 누적은 정확하게 확인했다.
별도 적격 Fixed-output Profile은 P Attention의14.724ms 차이를 확인했다.
하지만 Natural Generation의 GPU 인과 기여율은 확정하지 못했다.
Launch, Chunk와 Host 준비 비용도 구분해야 한다.

**Q3. 공정한 Current BI0에서 재현 가능한 Layer Sweet Spot이 있는가?**

이번 Full200/QPS2.35 확인에서는 확보하지 못했다.
원본 Length Bucket의 짧은 입력에는 Layer가 빠른 N8 Point가 있었다.
이는 독립1회 진단이다. Task·Output 차이도 있어 성공 Gate로 사용하지 않는다.
Historical Workload의 Current KV16 첫 Layer 승리도 Held-out3회에서 확정되지 않았다.

**Q4. Decode SLO를 보호하는가?**

C0의 두 Axis는 Native-relative5% Gate를 통과했다.
이는 Active Decode 사이에 추가 Prefill을 안전하게 넣는 Window 검증과 다르다.
Historical Layer의 Native-relative ITL CI 상한1.055447은1.05 Gate를 실패했다.

**Q5. Continuous Serving으로 일반화할 수 있는가?**

현재 C3 Layer Adapter는 Partial Prefill/Full Decode Mixed-row Assertion에서 차단됐다.
정합성 Assertion을 제거하지 않았다. 적격 C3 Layer 성능은 `NOT_RUN`이다.
따라서 C0의 결과를 Strong Continuous Serving 우위로 일반화하지 않는다.

## 반증 결과와 미실행 범위

Credit1/4/8/16/32의 Last8 One-factor Control에서는 Token TTFT 우위가 유지됐다.
단, Credit의 영향은 Cut에 따라 달랐다.
새 Last12에서 Token의 추가 Credit Extension 누적은2424.919ms였다.
P/D Host Demand의 우세한 단계도 Last8의 P에서 Last12의 D로 바뀌었다.
이는 설명 후보다. 검증된 Single-knob Threshold는 아니다.

BI1 Coalescing Flag의3회 진단에서는 TTFT 비율 차이가 관측됐다.
하지만 실제 Descriptor는 입력1764개, 제출1764개로 줄지 않았다.
물리적 Coalescing Intervention은 `NOT_REALIZED`다.
이를 Historical DMA 병합 이득의 인과 재현으로 사용하지 않는다.

N50 Discovery는 QPS0.4/1.2/2.35의27개 Cell을 완료했다.
모두 사전 Task-quality CI Gate를 통과하지 못해 `DIAGNOSTIC`으로 남겼다.
나머지36개 N50 Method Cell은 `NOT_RUN`이다.
Full200은 QPS2.35의 Axis8개 Cell×독립3회와 Native3회를 확보했다.
다른 QPS의 Full200 Sweep, Controlled Decode Batch2/4와 Exact-rows 보정 Intervention은 `NOT_RUN`이다.
Best-vs-Best 전역 튜닝도 완료한 것으로 표시하지 않는다.

Natural Source와 D Suffix의 적격 Kernel Breakdown은 확보하지 못했다.
Observer 실패와 Worker 종료 실패의 원시 기록을 보존했다.
SM/HBM Counter와 순수 DMA 시간은 `NOT_MEASURED`다.
새 Arrival Seed의 Replay는 Layer Goodput 오차15.388%로 사전10% Gate를 실패했다.
검증된 추가5% Dynamic Oracle Headroom과 최소 Knob 일반화도 확보하지 못했다.
따라서 Online Dynamic Size/Axis Controller를 구현하지 않았다.

## 공통 계약과 제한

Qwen3-8B Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
GPU0/1의 H100 SXM80GB 두 대를 사용한다. 각 GPU의 KV Budget은42GiB다.
Primary Numerical Mode는 BI0다. Original BF16과 Full36 Decode를 유지한다.
B8192/S1024는 Scheduler 상한이다. 실제 Batch와 다르다.
Primary Dataset은 LongBench-ShortAnswer200-v2다. Natural EOS Cap은2048이다.
Dataset SHA256은 `8e58b2f75bd0c0ef274d6e53d9b4044e6aecf3a09a787133ee128b21f702e6d6`다.
Source Dataset Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`다.
Task별40개인 HotpotQA/MuSiQue/NarrativeQA/QMSum/Qasper를 그대로 사용했다.
Aggregate GEMM Rows는 near-matched다. Alignment 오차를 Exact Work라고 표시하지 않았다.
Frozen SLO는 TTFT5000ms, 모든 ITL34ms와 TPOT33ms다.
Primary Latency는 예정 Arrival→Coordinator Delivery다. HTTP 전송 비용은 포함하지 않는다.
모든 결과는 Finite Cohort다. Sustainable Capacity를 입증하지 않는다.

GPU0 UUID는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`다.
GPU1 UUID는 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`다.
Compile0, FULL_DECODE_ONLY와 Async Scheduling을 사용했다.
실제 Worker의 B/S, Weight dtype, KV 용량과 UUID를 별도로 보존했다.
공통 Coordinator SHA256은 `36a9c5bfeba591a5f4b5d202ee155af0bc5af06ced94469351b5fdabd79f3856`다.
각 실행의 전체 Source ZIP·SHA256, Prompt Hash와 Arrival Trace도 보존했다.

## 보고서

- [Credit와 Serial/Online 인과 분해](dense_ar_bi0_credit_queue_root_cause.md)
- [전체 Cut·QPS 비교](dense_ar_bi0_all_cuts_qps_sweep.md)
- [Input Length Control](dense_ar_bi0_input_length_axis_crossover.md)
- [Credit·KV·Batch Sensitivity](dense_ar_bi0_batch_kv_credit_sensitivity.md)
- [최소 Knob과 Replay 한계](dense_ar_bi0_layer_token_sweetspot_knob_final.md)
- [Continuous Serving Bridge](dense_ar_bi0_continuous_serving_bridge.md)
- [Historical Winner Bridge](dense_ar_bi0_historical_layer_winner_reproduction.md)
- [Ready→D Queue 분석](dense_ar_bi0_ready_to_d_queue_root_cause.md)
- [Attention·GEMM 분석](dense_ar_bi0_attention_gemm_axis_crossover.md)
- [PNG/PDF Figure 목록](dense_ar_bi0_axis_rootcause_20261120/figures/README.md)

## 보존과 출판

기존 사용자 Checkout과 Staged Index는 보존한다. 출판은 별도 Git Index를 사용한다.
기존 Source120개와 보고서5개의 SHA256을 검사한다.
설치된 vLLM Python 파일2234개도 Wheel RECORD와 비교한다.
모델과 설치 Runtime을 고쳐 성능을 맞추지 않았다.
실패한 실행과 유리하지 않은 Start는 삭제하지 않았다.

GPU0–7의 승인 Burner를 복원했다. 소유권과 사용률을 검사했다.
초기 모델 로딩 중 사용률 검사는 실패했다. 로딩 후 재검사는 통과했다.
Raw Evidence는 무손실 Zstandard Archive와 SHA256으로 보존한다.
선택한 원시 자료2,982,256,068 Bytes를471,326,765 Bytes로 압축했다.
Member별 SHA256을 모두 검사했다.48MiB 이하의10개 Part로 나눴다.
미인증 Debug Tensor와 중복 Source의 제외 이유·Hash도 보존한다. 원본은 로컬에 남긴다.
GPU 최종 Release까지 Wall-clock은8.7165시간이었다.
Dispatcher Command218개 중214개가 정상 종료했고4개는 실패했다.
정합성·진단 실행도 포함한다. 독립 Serving Confirmation214회라는 뜻은 아니다.
Elapsed×할당 GPU 수의 합은16.2606 GPU-hours다. Exclusive GPU Compute Time이 아니다.
최종 Remote SHA는 Push 뒤 `git ls-remote`와 비교한다.
자기참조를 피하기 위해 최종 SHA는 Local Publication Receipt와 최종 답변에서 기록한다.

## CPU 재현

Archive 복원 방법과 Member별 SHA256은 `archives/large_evidence_manifest.json`에 있다.
복원한 Evidence에서 다음 분석은 추가 GPU 실험 없이 실행할 수 있다.

```bash
python -m poc_flashvep.dense_ar_bi0_axis_rootcause.statistics
python -m poc_flashvep.dense_ar_bi0_axis_rootcause.confirmation_service
python -m poc_flashvep.dense_ar_bi0_axis_rootcause.figures
python -m poc_flashvep.dense_ar_bi0_axis_rootcause.historical_figures
python -m poc_flashvep.dense_ar_bi0_axis_rootcause.final_validation --final
```

원래 Python 경로는 `/home/esjung/.venvs/looped-pd-abc/bin/python`이다.
새 GPU 재실행은 별도 Evidence Namespace와 승인 Lease를 사용해야 한다.
종료한 Namespace의 실패 기록을 지우거나 `CAMPAIGN_CLOSED` Guard를 우회하지 않는다.

판정은 `HOLD`다. Queue 원인은 부분 식별했지만 Layer-specific Serving 성공 Gate는 통과하지 못했다.
이 결론은 이번 Model·Dataset·Numerical Mode·Runtime과 실행 범위에 한정한다.
