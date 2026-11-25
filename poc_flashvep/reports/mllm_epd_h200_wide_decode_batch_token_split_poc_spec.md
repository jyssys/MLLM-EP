# H200 Wide Decode Batch와 Exact Token Split PoC

H200에서 Decode Admission을 넓히면 이전 Batch1 통제에서 감소했던 D Capacity와 Idle이 회복되는지 검사한다. 확인하려는 Serving 이득은 같은 GPU·Admission·도착 계약에서의 TTFT 개선과 ITL·TPOT 보호다. 2026-11-26 Notion 결정에서 B4·B8·B16·B32·B64·Free를 탐색 대상으로 삼았고, B1·B2를 통제와 보조로 유지했다. 여기서 B는 D에서 동시에 살아 있는 요청의 상한이며 실제 Decode Batch는 Scheduler와 Forward에서 측정한다. [최신 연구 계획](https://app.notion.com/p/4220ee8277e48153a020e374317777f8) · [실험 아카이브](https://app.notion.com/p/1062a9a81d8641f895f0bdf6c0702489)

## 이전 H200에서 이어받은 근거

독립 E/P/D-B1/D-Free의 유한 Cohort 처리율은 각각 3.92/3.18/3.32/7.19 req/s였다. Phase B QPS2 Equal-output Clean 전후 평균에서 NativeCommon Free/B1/Static10-B1/Idle10-B1의 TTFT p95는 4.47/5.25/5.84/4.96초, ITL p99는 12.19/5.97/5.98/6.21ms였다. Actual Prepared 최대 요청은 Counter 측정에서 4/1/1/1이었고 Counter 간섭 기준은 실패했다. NativeCommon-B1 대비 Static10-B1의 평균 P HTTP 변화 −311ms와 D 대기 +347ms를 평균 TTFT +219ms와 함께 본다. Batch1에서 Static10의 ITL 격차가 좁아졌다는 방향성은 유효하나, 독립 N3·품질·실용 Free 우위는 미확정이다. [이전 H200 종합 보고서](mllm_epd_h200_prefill_heavy_token_split_final_summary.md)

NativeCommon Free QPS5의 두 `DIAGNOSTIC` Trace에서 E/P/D Busy는 87.05/92.95/61.16%와 88.16/93.86/60.25%였다. P Backlog와 겹친 160ms 이상 D Quiet의 P Busy 대비 비율은 27.22%, 26.22%였고 Decode Attainment는 각각 95%였다. 두 경우 모두 Trace Qualification이 실패했다. B1 QPS5/6에서는 높은 P Busy와 달리 이 겹침이 4.72%, 7.99%로 제한됐다. 그래서 Free QPS5를 적격 재측정의 첫 지점으로 삼는다. [이전 H200 Discovery](mllm_epd_h200_prefill_heavy_discovery.md) · [독립 Capacity](mllm_epd_h200_independent_stage_capacity.md)

## 외부 비교의 범위

| 결과 | 이번 연구에 적용하는 질문 |
| --- | --- |
| DuetServe Fig. 2는 Qwen3-8B, 2×H100, 8000 Input·200 Output의 텍스트 요청에서 Agg-vLLM과 Disagg-Dynamo를 비교한다. 분리형은 QPS가 4를 넘으며 TTFT가 급증했고 TBT는 상대적으로 안정적이었다. | H200에서 동일한 방향의 TTFT·TBT·Throughput 변화가 있는지 본다. E·Vision·입출력 분포·GPU 수가 달라 절대 수치 재현 기준은 다르다. [DuetServe](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf) |
| DistServe는 TTFT·TPOT를 함께 만족하는 Goodput에 대해 P/D 자원과 배치를 조정한다. | 늘어난 D Capacity를 단순 총 Throughput이나 낮은 평균 ITL만으로 성공 판정하지 않는다. [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) |
| Kairos는 D의 진행 중 Decode와 기존 요청의 Prefill Chunk를 병행해 전체 Prefill을 Deflect한다. FluidPD는 순간적 부담 차이를 Token 일부 이전으로, 지속적 차이를 Role 변경으로 다룬다. | D 자원 활용은 선행연구가 있다. 이번 범위는 원래 MLLM E를 유지하며 Decode가 없는 D에서 exact suffix를 실행하고 실제 Wide Batch 보존을 검사하는 것이다. [Kairos](https://arxiv.org/abs/2607.02043) · [FluidPD](https://arxiv.org/abs/2610.06917) |
| Singh et al.은 멀티모달 요청에서 Encode를 Prefill 및 Decode와 분리하고 Encoding·자원 조정을 연구한다. | E에 실제 Vision 요청을 공급하고 Image-heavy 자체를 P-heavy로 해석하지 않는다. [EPD Disaggregation](https://proceedings.mlr.press/v267/singh25d.html) |
| 공식 Recipe는 Qwen2.5-VL의 BF16 실행과 VisionArena·Synthetic Bench를 보여준다. | Stock Single-H200와 Native EPD의 같은 Calibration32 절대 지연을 이 서버에서 비교한다. 공식 예시는 성능 단위와 외부 범위를 확인하는 데 사용한다. [공식 vLLM Recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen2.5-VL.html) |

## 동결 계약과 실행 순서

- 새 Branch `flashvep/mllm-epd-h200-wide-decode-batch-token-split`, H200 기준 SHA `e1a31767ef3c9653e7a61b5a9d8da52ad0e4baee`에서 출발한다. 기존 Worktree·Raw·Manifest를 보존한다.
- `Qwen/Qwen2.5-VL-7B-Instruct` Revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16·BI0·원래 Vision/Processor·Full28·정확 Causal·MRoPE, Greedy, Prefix Cache Off. P/D B8192·S1024·KV42GiB·Block128을 고정하고 E B114688·Renderer2·MM Cache0·Sync Scheduling·Custom Ops None을 유지한다.
- Main100 Natural EOS Cap2048을 주 집합으로 유지하며 Equal-output Replay 5,269 tokens를 별도 비교한다. Calibration32는 같은 요청을 쓰는 Sanity와 정책 회귀에 사용한다. Arrivals·요청 순서·Seed·출력 계약·독립 Start와 Actual Work를 기록한다.
- 세 실험 GPU를 같은 E/P/D 배치로 유지한다. 버너 상태는 실행 환경의 일부로 기록하며 Free QPS5에서 나머지 GPU의 Burner On/Off를 구분한다. 승인 GPU에 있는 확인된 Burner와 이번 캠페인 소유 프로세스만 제어한다.

| 단계 | 설계 |
| --- | --- |
| A. 절대 Latency Sanity | 같은 H200, Runtime, 모델·Processor, Calibration32 요청, 출력 계약에서 Stock Single GPU와 Native EPD를 비교한다. 요청별 E/P/D HTTP·Client TTFT/ITL·TPOT·출력 길이와 실행된 P/D 분리를 기록한다. GPU 수 차이를 포함한 진단이며 동일 예산의 Serving 경쟁으로 판정하지 않는다. |
| B. 독립 D Capacity | 실제 준비한 NIXL KV와 동일 출력 계약에서 B1/2/4/8/16/32/64/Free를 각각 측정한다. Offered cap과 Actual Scheduled·Prepared·Graph Padding, cap binding, D Queue, 실제 KV Headroom을 기록한다. B32/B64가 형성되지 않는 런은 실제 도달한 Batch로 보고한다. |
| C. Native 온라인 Grid | 먼저 Native Free QPS5를 확인한다. B4/B8/B16/B32/B64/Free 각각에 QPS 2, 3, 4, 5, 6, 8을 적용하고 B1/B2에 QPS 2, 4, 5, 6을 적용한다. 완주하지 못한 셀은 생략 이유와 함께 기록한다. 독립 N1을 탐색으로 두고 유망 조건만 독립 Start로 확인한다. |
| D. Motivation 재검증 | 적격 CUPTI Kernel Trace의 중앙 Arrival p20–p80에서 E/P/D kernel busy, 실제 scheduler queue, D Quiet Window, P Busy·P Backlog 동시 효과를 교차 검증한다. Free QPS5의 Burner On/Off를 별도 측정한다. 이전 5% Clean/Profile/Clean Gate, P Busy≥80%·Service Proximity≥85%·실제 Backlog·Starvation 부재·D KV/Buffer·Decode Attainment≥95% 및 160ms 기준을 유지한다. |
| E–G. Static과 Idle | 원래 Decode가 없을 때만 D에 exact suffix를 보내며 실제 P/D KV·Embedding Lifetime, Suffix 이후의 Batch 재형성, 다른 요청의 Admission Wait를 관측한다. Static10부터 확인하고 같은 cap Native와 강한 Free Native를 각각 비교한다. 정합성 회귀가 완료된 Ratio만 성능 실험에 넣는다. Idle 계열은 Static 결과 이후 기존 현재 상태 정책을 검증하고 적용/미적용 건과 실제 이전 Token을 센다. |

P Busy와 최근 P Backlog에 동시에 겹치는 160ms 이상 D No-active/pending Window를 두 개 이상, P Busy 대비 10% 이상 관측할 때 초기 D Idle 조건을 평가한다. 225ms와 128–2048 Token별 비용은 민감도 근거다. Qualification만 반복적으로 실패하고 독립 Capacity·P Queue·D Quiet·SLO 근거가 강하게 일치하면 최대 두 조건에서 `CONDITIONAL_ONLINE_PILOT`을 수행할 수 있다. 증거 범위는 Pilot과 진단이며, `GO`나 적격 P-heavy 판정을 부여하지 않는다.

## 비교 및 판정

Decode Cap 설정과 실제 Batch를 나눠 보고한다. P/E와 전체 Client의 동시성을 유지하고 D Suffix의 다수 토큰 행을 다수 Decode 요청으로 세지 않는다. Same-cap Native 대비 Scheduled/Prepared 히스토그램, 시간 가중 D Queue, Cap Binding, 실제 KV 길이, 첫 Token 이전 진행과 Decode Tick을 함께 보관한다. Candidate가 넓은 Batch를 실제로 유지하는지 성능과 별도로 판단하고 Batch1에 가까워진 경우 이를 원인 후보로 기록한다.

TTFT ≤5,000ms, 모든 Client ITL ≤34ms, TPOT ≤33ms/token을 그대로 적용한다. 같은 cap Native 대비 C3 TTFT p95가 5% 이상 감소하고 Paired CI가 개선 방향을 지지해야 한다. TTFT p50 비열등, ITL p99·TPOT p95 ratio의 Paired 95% CI 상한 ≤1.05, Absolute Decode ≥95%, 품질 유지와 Joint/Goodput 보호도 확인한다. 실제 Free 비교와 모든 통제는 같은 GPU 그룹을 사용한다. 독립 Start를 반복 단위로 삼고 Natural·Replay와 H100·H200의 결과를 따로 보고한다.

최종 보고는 Runtime Sanity, Capability·Actual geometry, QPS trade-off, Qualification, E Busy 조건, Static·Idle·Critical Path, 같은 cap 및 Free 대비 두 효과, 품질과 향후 Modality-aware 근거에 답한다. 미실행·실패는 수치가 있는 관측과 함께 보존한다.

## 재현 자료

각 Start의 실제 인자, 입력 Hash, Runtime 및 GPU 배치는 실행 인벤토리에 있다. 원본 Timestamp·요청별 출력·Worker Counter·CUPTI Trace는 인벤토리의 `raw_root` 아래에 있다. 저장소의 결과 CSV와 PDF는 해당 원격 GPU를 사용하지 않고 재생성할 수 있다. [실행 인벤토리](../mllm_epd_h200_wide_decode_batch_token_split/results/execution_metadata.json)

저장소 루트에서 `RAW_RUNS`에 기존 실행 경로, `RESULTS`에 새 출력 경로를 지정하고 Matplotlib이 설치된 Python으로 실행한다.

```bash
package=poc_flashvep.mllm_epd_h200_wide_decode_batch_token_split
python -m "$package.analysis" --runs "$RAW_RUNS" --output "$RESULTS"
python -m "$package.figures" --runs "$RAW_RUNS" --results "$RESULTS"
python -m "$package.quality" --runs "$RAW_RUNS" --output "$RESULTS"
python -m "$package.report" --runs "$RAW_RUNS" --docs poc_flashvep/reports \
  --analysis "$RESULTS" --extra-output "$RESULTS"
```
