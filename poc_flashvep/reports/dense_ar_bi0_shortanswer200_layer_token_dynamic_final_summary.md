# BI0 ShortAnswer200: Layer/Token Offloading 및 Decode-safe Scheduling

## 문서 상태

캠페인의 GPU 실행과 CPU 분석을 종료했다. 최종 연구 판정은 `HOLD_LAYER_AWARE_DECODE_SAFE_DYNAMIC`이다. BI0 기준선, Dataset, 고립 비용과 Quiescent-only Static 개선은 확보했다. Layer 고유의 Online 이득과 Active-decode Dynamic 이득은 확보하지 못했다.

이 판정은 이번 Model, Dataset, GPU와 Runtime 계약에 한정한다. Layer 또는 Dynamic의 보편적인 불가능성을 뜻하지 않는다. 실행하지 않은 Phase는 `NOT_RUN`으로 남겼다.

## 연구 계약

모델은 Qwen/Qwen3-8B다. Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다. GPU0/1의 H100 SXM80GB를 사용한다. Precision은 Original BF16이다. Attention은 Exact Causal Attention이다. Greedy, Thinking Disabled, Natural EOS와 Output Cap2048을 적용한다.

모든 Primary Worker는 B8192/S1024와 GPU당 KV42GiB를 명시한다. Prefix Cache는 비활성화한다. BI0와 BI1의 성능·출력을 같은 계약으로 취급하지 않는다. vLLM은0.26.0+cu129다. PyTorch는2.11.0+cu129다. NIXL은1.4.1이다.

BI0는 `VLLM_BATCH_INVARIANT=0`이다. BI1은1이다. 두 모드의 Weight Dtype이 BF16이어도 Numerical Path와 Greedy Output이 모두 같지는 않다.

실험은 세 종류다. Ordinary Continuous Agg/Native는 C3 실행 경로다. Layer/Token Mechanism Control은 C0, Async Scheduling, Full Decode Graph와 Actual Batch1/1/1을 사용한다. 고립 Forward는 Queue·Transport·Sampling을 제외한다. 이 세 결과를 같은 실행 계약으로 합치지 않는다.

공통 Frozen SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, TPOT≤33ms다. Native-relative5% 비열등성은 별도 판정이다. 각 Request의 실제 출력 Token 수를 사용한다. Finite-cohort 처리량은 Arrival→Drain으로 계산한다. Sustainable Capacity를 측정했다고 주장하지 않는다.

## Dataset: 확정

`LongBench-ShortAnswer200-v2`는 HotpotQA, MuSiQue, NarrativeQA, QMSum, Qasper를 각각40개 포함한다. GovReport 포함 기존 v1은 보존한다. 기존 네 Task의160개 ID·원문·Tokenization 계약을 유지했다. Qasper는 결과를 보기 전에 고정 Seed로 선정했다.

Dataset Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`다. Dataset SHA256은 `8e58b2f75bd0c0ef274d6e53d9b4044e6aecf3a09a787133ee128b21f702e6d6`다. Context Limit은40960이다. Output Reserve는2048이다. NarrativeQA10개에 명시적인 Head/Tail Truncation을 적용했다. 제외 요청은0개다.

Input은 총3,111,075Tokens다. 평균은15,555.375Tokens다. p50은15,661이다. p95는38,253.65다. Min/Max는1,910/38,912다.

BI0 Single-GPU C1 Reference의 실제 출력은 총4,723Tokens다. 평균은23.615Tokens다. p50은7이다. p95는102.1이다. Max는152다. Cap Hit는0/200이다. Qasper 평균/Max는13.8/44Tokens다. 전체12/200개가100Tokens 이상이다. 모두 QMSum이다. 모든 요청이100Tokens 미만이라고 주장하지 않는다.

상세 Manifest와 분포는 [Dataset 계약](dense_ar_shortanswer200_bi0_dataset_contract.md)에 있다.

## Native BI0 재현

`MEASURED_ONLINE`. 동일 Calibration20, C0/Async/Decode Graph/KV128/Batch1에서 Single-GPU Mean TBT는7.779ms였다. Native PD는7.544ms였다. 각1개 독립 Start다. CI는 계산하지 않는다. GPU 수가 다르므로 이 결과를 동일 자원의 Throughput 비교로 사용하지 않는다.

Greedy Output은18/20개가 일치했다. Native D는 마지막 Prompt Query를 다시 계산한다. BI0의 다른 실행 Geometry가 같은 전체 Greedy Sequence를 보장하지 않는다. Cross-geometry 비교에는 Task Quality와 State 검사를 별도로 적용한다.

기존 BI1의22–25ms/Step을 현재 BI0에서 재현하지 않았다. BI0로 전환한 결과를 BI1 출력을 보존한 Runtime 수정이라고 표현하지 않는다.

Actual Batch1 Native N32의 Async A/B에서 Mean TBT는10.944→7.380ms였다. Greedy Output은32/32개가 일치했다. 이 Screen은 각1회다. 이후 모든 Layer/Token Cut에 새로운 Async State Qualification을 적용했다. Primary Controlled 성능에는 이 검증된 경로만 사용한다.

## 정합성 범위

`CORRECTNESS_ONLY`. Last1/2/4/6/8/12의 Layer와 Token 경로는 N8 State Gate를 통과했다. Token은 각 Partition과 같은 Single-GPU Chunk Teacher도 통과했다. Native와 Layer/Token은 Valid KV, Hidden, Position, Query Coverage, Full36 Decode, ACK와 Lifetime을 검사했다.

Min Input1910, B8192 경계 근처, Max Input38912의 N3 Context Control도 통과했다. Performance 실행에는 CPU Tensor Capture를 적용하지 않는다. Layer Producer의 비인증 Debug Tensor는 Local에 Hash와 함께 보존한다. 이를 유효한 Full36 KV 증거로 사용하지 않는다.

Full200 Cross-geometry Task Quality는 별도 검사한다. Task별2.0Point 비열등성 Margin은 연구용 사전 계약이다. Production Standard가 아니다. 실패한 Cell은 삭제하지 않는다. Primary 적격 결과와 분리한다.

## Prefill-heavy와 Safe Window

`MEASURED_ONLINE` 및 `TRACE_ANALYSIS`. Continuous Native/KV128/QPS2.35의 P-only Profiler는 Clean/Profile/Clean5% Observer Gate를 통과했다. 동일86.1447초 ROI에서 P Kernel Busy는99.6264%였다. P Waiting Queue 평균/최대는28.39/43이었다. D Decode-ready Waiting은0이었다.

D의 현재 Stream 완료와 Scheduler/Engine Empty를 함께 확인한 구간은121개였다. 합계는52.6953초였다. p50/p95/p99는317.93/1290.03/1566.88ms였다. 이는 Host-observed Quiescence다. All-stream Physical Idle이나 미래 Decode-safe Budget으로 바꾸어 해석하지 않는다.

두 GPU를 동시에 계측한 CUPTI 실행은 Observer Gate에 실패했다. Mean TBT는10.24%, TPOT p95는46.03% 증가했다. 해당 D Kernel Busy27.66%는 `DIAGNOSTIC`이다. Clean Primary D Kernel Busy로 사용하지 않는다.

따라서 P-prefill 병목의 근거는 확보했다. Active Decode 사이에 삽입 가능한 미래 Safe Window의 독립 검증은 아직 `HOLD`다.

## Layer/Token 비용

`ISOLATED_KERNEL_MEASUREMENT`. Input4608/8064, Batch1, Last1/2/4/6/8/12의 Exact Matched GEMM Rows Grid를 검사했다. 각 Cell은 독립 Start3회다. 이 Grid의 Full Forward는 Layer가 더 빨랐다. 큰 Cut의 Token 우위로 역전됐다고 주장하지 않는다.

Input8064에서 Last1의 Token/Layer Forward 비용비는2.389였다. Last4는1.254였다. Last12는1.142였다. 작은 Token M에서 Projection 비용 차이가 컸다. 큰 Cut에서는 Attention 배치도 중요했다. 같은 GEMM Rows가 같은 Attention FLOPs를 뜻하지 않는다.

고립 Forward의 KV Block은16이다. 신규 Controlled Online은128이다. Microbenchmark 비율을 Online 비용으로 직접 대체하지 않는다. 실제 요청 Serial Service와 Online TTFT는 별도 측정한다.

Full200은16Token Prefix Alignment를 사용한다. 따라서 Aggregate GEMM Rows가 완전히 일치하지 않는 Cut이 있다. 실제 오차를 보존한다. Exact Matched Work 주장은 고립 Grid에 한정한다.

## Online 결과와 판정

QPS2.35의 독립 Held-out3Start를 완료했다. Seed는2026112051/52/53이다. Discovery Seed2026112061과 분리했다. 다만 같은200개 Prompt를 사용했다. 새로운 Prompt 집합의 일반화 검증은 아니다.

아래 p95/p99는 각 Start 통계의 평균이다. 처리량도 Start별 값의 평균이다. 모든 행은 같은 KV Block128, B8192/S1024, GPU0/1과 Natural EOS를 사용했다. 각 Start의 Task-quality CI를 모두 통과했다.

| Method | Actual Batch 계약 | TTFT p95 | ITL p99 | TPOT p95 | Raw Output Tokens/s | SLO-qualified Tokens/s | Joint SLO |
|---|---|---:|---:|---:|---:|---:|---:|
|Continuous Native PD|C3, 기본 Continuous|20.068s|16.443ms|8.093ms|46.251|0.943|5.67%|
|Agg DP2|C3, 기본 Continuous/RR|4.209s|372.783ms|283.808ms|55.036|16.085|42.50%|
|Controlled Native|C0,1/1/1, Source Credit4|24.680s|8.969ms|8.196ms|42.826|0.750|5.33%|
|Static Layer Last8|C0,1/1/1, Source Credit4|12.006s|9.040ms|8.247ms|52.323|17.347|33.17%|
|Static Token Suffix≈L×8/36|C0,1/1/1, Source Credit4|9.603s|8.974ms|8.208ms|53.547|23.393|44.67%|

Source의 핵심 Runtime8개 파일 Hash가 일치했다. Dataset과 실제 Arrival Trace도 일치했다. 실제 Worker GPU UUID 쌍도 일치했다. C3 External과 C0 Control의 Compile/Batch/Admission 차이는 여전히 존재한다. 이 표 전체를 Actual Batch-matched 비교라고 부르지 않는다.

Layer/Token의 TTFT p95 비율은1.2502다. Paired-start95% CI는[1.1871,1.3448]이다. SLO-goodput 비율은0.7416다. CI는[0.5128,0.8233]이다. 이 조건에서는 Layer가 강한 Token 고정 후보보다 유리하지 않았다.

Layer/Continuous Native의 TTFT p95 비율은0.5982[0.5508,0.6437]다. Token/Continuous Native는0.4785[0.4100,0.5422]다. Native 대비 개선은 확인했다. 그러나 이를 Layer 고유의 이득이나 Dynamic 이득으로 해석하지 않는다.

Layer와 Token 모두 Controlled Native 및 Continuous Native 대비 Decode5% 비열등성을 통과했다. Continuous Native 대비 ITL p99 CI 상한은 각각0.6043/0.6005다. TPOT p95 CI 상한은1.0230/1.0162다. Absolute Decode SLO와 이 Relative 판정은 다르다. Controlled Native와 두 Static의 모든-ITL Request 통과율은 각각99.5%였다. 모든 요청의 모든 Interval이34ms 이하였다는 뜻은 아니다.

Agg 대비 SLO-goodput의 Layer 비율은1.0785[0.4744,1.5241]다. Token은1.4543[0.8851,1.9147]다. 두 CI 모두1을 포함한다. Agg를 이겼다는 반복 가능한 SLO-goodput 결론은 미확정이다. Agg의 Raw 처리량이 가장 높았다. 이 Agg는 정상 Continuous B8192/S1024다. 새 Dataset의 모든 SLO별 Best-tuned Frontier를 전수 선정한 결과는 아니다.

추가 QPS1.65는 각 Arm1회 완료했다. Layer8/Token8의 TTFT p95는7.154/5.700초였다. Goodput은31.112/33.969Tokens/s였다. 두 Arm의 Full200 Task-quality Gate도 통과했다. 이1회를 통계적으로 재현된 다른 QPS의 우위로 표현하지 않는다.

동일 Output Control은 Seed2026112062/QPS2.35에서 각1회 실행했다. 세 Arm 모두 C1의 요청별 길이대로4723Tokens를 생성했다. Layer8/Token8의 TTFT p95는5.785/4.774초였다. Natural Output 수를 같게 한 진단에서도 Token의 TTFT가 낮았다. 이 Control은 Natural Generation의 Task 품질 비교를 대체하지 않는다.

고부하 Native의 낮은 Goodput을 분모로 한 큰 배율을 Sustainable Capacity 개선으로 표현하지 않는다. CI는3개 Start의 경험적 Bootstrap이다. 가능한27개 조합을 사용했다.3개 표본의 불확실성 범위는 제한적이다. 실행 순서는 무작위화하지 않았다.

Static 후보는 Last1/2/4/6/8/12, Migration100%, Source Credit4와 Actual Batch1/1/1 범위다. 이 범위의 Best Tested Static이다. 모든 Fraction·Batch·Streaming Token Pipeline을 포함한 전역 Best-tuned 결과가 아니다.

## Dynamic과 Oracle의 제한

현재 적격 Online Offloading은 D가 완전히 Quiescent인 경계에서만 실행한다. 다른 Decode를 실행 중인 GPU에 Suffix를 삽입하지 않는다. 이 Safety 계약은 Active-decode Window Scheduler의 성공이 아니다.

State-updating Replay는 P/D Queue와 Source Credit4를 갱신한다. Model v2는 Serial Source Submit→Ready 비용을 사용한다. Oracle은 미래 Arrival/EOS를 사용한다. Finite Beam24는 전역 최적 상한이 아니다.

Replay의 성과 지표는 TTFT-qualified Token Proxy다. Client ITL Tail과 Action별 Task Quality를 모델링하지 않는다. Full Joint SLO-goodput이 아니다. 따라서 Proxy의5% Headroom만으로 Online Axis Controller를 허용하지 않는다.

Dynamic Size와 Dynamic Axis의 실제 Online 이득은 `NOT_RUN`이다. Active-decode Safe Window가 검증되지 않았으므로 위험한 삽입을 실행하지 않았다.

Conditional Queue Validation의 TTFT MAE는284.33–1897.95ms였다. TTFT p95를6.21–40.26% 낮게 예측했다. Layer8 D Service의 Validation Upper Coverage는11/15였다. 안전한 Admission용 Tail Model로 사용하지 않았다.

유한 Oracle 탐색의 Mixed-axis 추가 Proxy는 Best Specialized-axis 대비9개 Cell 모두0%였다. Best-per-cell Hindsight Fixed 대비 추가 Proxy는 최대0.0248%였다. 전역 최적 Headroom이0이라는 증명은 아니다. 두 상태 Rule도 Held-out에서 Metadata-only 대비0.96174–1.00366배였다. 검증된5% 추가 이득을 확보하지 못했다.

따라서 복잡한 Online Axis Controller를 구현하지 않았다. Native 대비 Static 개선, 고립 Layer 효율과 Dynamic 독립 기여를 분리한다.

## 무엇이 차이를 설명하는가

`STRUCTURAL_ACCOUNTING`. 같은 GEMM Rows에서 Token은 D에 더 많은 Attention을 배치한다. 그만큼 P Attention을 더 줄인다. 전체 P+D의 유효 Causal Attention Pair는 같다. Layer는 추가 Hidden/Residual도 전송한다. 따라서 Layer의 짧은 D Forward가 전체 TTFT의 충분조건은 아니다.

`ISOLATED_SERIAL_SERVICE_MEASUREMENT`. 같은 Calibration20에서 Layer8의 P Prepare→Ready/Ready→First Worker 평균은397.80/144.43ms였다. Token8은370.24/197.34ms였다. Layer의 D 경로는 짧았다. Token의 P 경로는 짧았다. 이 값은 Host Envelope다. Exclusive GPU Kernel Time이 아니다.

`TRACE_ANALYSIS`. Discovery Last4의 Layer−Token Mean TTFT 차이는2673.86ms였다. P Queue 차이는2711.09ms였다. D의 두 구간은 합계59.61ms 짧았다. 작은 D 비용 이득을 P Queue 차이가 상쇄했다. Mean 경계의 분해를 p95의 Additive 분해로 표현하지 않는다.

Actual Batch1/1/1을 일치시켰으므로 과거 Decode Batch1대2–3 차이로 이번 순위를 설명할 수는 없다. Query Length, Layer 수, B8192 Chunk Count와 Attention 배치는 Axis의 구조적 차이로 남았다. Source Credit4의 Backpressure도 일부 P Queue를 만들었다. 모든 P Queue를 GPU 포화로 해석하지 않는다.

정확한 Online Operator별 GPU 비중은 확보하지 못했다. C0 P-only Profile은 Mean TTFT 간섭5.2908%로5% 기준을 실패했다. C3의 적격 P-only Profile과 이를 섞지 않았다. 최종 원인 판정은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`이다.

## 최종 Gate

| 연구 항목 | 판정 | 근거와 제한 |
|---|---|---|
|ShortAnswer200-v2|PASS|원문200개, 고정 Revision/Hash, Natural Output4723Tokens의 Reference|
|BI0 Native Reference|SCOPED_PASS|동일 Batch1 Single-GPU와 Decode 범위가 비슷함. Cross-geometry 전체 Strict Parity는 아님|
|P-prefill 병목|MEASURED_ONLINE|C3 고부하의 적격 P Busy99.6264%와 같은 ROI Queue|
|미래 Active-decode Safe Budget|HOLD|Host Quiescence는 확인. Clean D all-stream Gap과 예측 Deadline Budget은 미검증|
|Layer Compute 효율|ISOLATED_KERNEL_MEASUREMENT|정확한 Rows의 두 Input Grid에서 Layer가 빠름. Online TTFT와 다름|
|Native 대비 Quiescent Static|CONDITIONAL_SERVING_GAIN|Held-out3Start TTFT 개선과 Decode5% 보호. Finite Cohort이며 실행 Family가 다름|
|Layer-specific Online 이득|NOT_ESTABLISHED|Best Tested Cut8에서 Token 우위. Cut12 Discovery는 각1회이며 반복은 NOT_RUN|
|Dynamic Size / Axis|HOLD / NOT_RUN|안전 Budget 미검증. 유한 Replay에서도 검증된5% 추가 Headroom 없음|
|강한 Agg 대비 전체 승리|NOT_ESTABLISHED|Raw는 Agg가 가장 높음. Static/Agg Goodput CI는1 포함|

다음 연구는 먼저 낮은 간섭의 D all-stream 계측과 미래 Ready/Deadline 예측을 확보해야 한다. 그 뒤에도 Whole-prefill Deflection과 Window-aware Fixed Token을 강한 대조군으로 유지해야 한다. 현재 결과만으로 복잡한 Layer×Token Controller 개발을 권하지 않는다.

## 재현과 자원 종료

Config, Source ZIP SHA256, Dataset Hash, Arrival Seed, GPU UUID, Actual Query Geometry와 실패 로그를 별도 Namespace에 보존한다. Main Table의 각 행은1개 독립 Server Start다. Request나 Token을 독립 Start로 계산하지 않는다.

2026-11-20 11:57KST에 Resource Closeout을 완료했다. 모든 캠페인 GPU Worker와 Controller가 종료됐다. 승인 Burner8개를 복원했다. GPU0–7 모두 실행 중이며 복원 검사에서 Utilization100%를 확인했다. 기존 Checkout, Staged Index, 과거 보고서와 외부 Runtime을 보존했다. 설치된 vLLM2234개 Python 파일은 Wheel RECORD와 일치했다.

실험은03:07:24KST에 시작했다. 최종1시간의 예산을 지키기 위해 Cut12 반복과 추가 Low-load Control은 실행하지 않았다. 실패 로그와 Local 원본은 삭제하지 않았다. Source와 유효 대형 Evidence는 SHA256으로 확인한 Lossless Archive에 보존했다. 비인증 Debug Tensor는 Local에 남기고 게시 제외 사유와 Hash를 기록했다.

게시 Branch는 `flashvep/dense-ar-bi0-shortanswer-axis-dynamic-20261120`이다. Base는 `3a97151dd4e32b3c4e421de36363931f7ad62621`이다. Private Index로 게시한다. 사용자 Checkout으로 Branch를 전환하지 않는다. 최종 Commit SHA는 최종 응답에 기록한다. Local `state/publication_receipt.json`은 Push 뒤 Remote SHA 일치를 검사한 영수증이다. 자기 Commit을 포함하는 순환 Hash를 피하기 위해 이 영수증은 Git Payload에서 제외한다.

## 근거와 재현

- [Online Table와 CI](dense_ar_bi0_online_serving_qps_results.md)
- [Source·Config·GPU UUID별 Main Table](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_main_table_with_provenance.json)
- [Paired Held-out 통계](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_shared_KV128_confirmation_statistics_v2.json)
- [실제 Batch와 Queue 원인 분석](dense_ar_bi0_layer_token_sweetspot_knob_analysis.md)
- [Oracle 제한과 검증](dense_ar_bi0_axis_selector_oracle_analysis.md)
- [Figure 목록](dense_ar_bi0_shortanswer200_main_figures.md)
- [최종 Resource Closeout](dense_ar_bi0_shortanswer200_20261120/state/CAMPAIGN_CLOSED.json)

CPU Test59개를 통과했다. 원시 Client Event에서46개 Cell의 주요 Metric을 다시 계산했다. 저장된 값과의 최대 절대 차이는0이었다.70개 완료 Controlled/Serial 실행의 Query Coverage도 다시 검사했다. 이는 새로운 독립 Start를 추가한 검증이 아니다.

CPU 재분석은 `python -m poc_flashvep.dense_ar_bi0_shortanswer_pd.final_analysis`다. CPU Test는 `python -m unittest discover -s poc_flashvep/dense_ar_bi0_shortanswer_pd -t . -p 'test_*.py'`다. 종료 검사는 `python -m poc_flashvep.dense_ar_bi0_shortanswer_pd.final_validation`이다. 실제 실험 환경의 Python 경로는 `/home/esjung/.venvs/looped-pd-abc/bin/python`이다. 원시 Evidence 복원은 Archive Manifest의 명령과 파일별 SHA256을 사용한다.
