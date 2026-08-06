# FlashVEP DeepEP overlap revalidation PoC

## 결론

**FINAL STATUS: HOLD**

DeepEP 기반 scheduler-free layer-24 replay에서는 실제 kernel overlap과 최대
**1.3100x** wall speedup을 확인했다. 따라서 기존 AllGather/ReduceScatter
wavefront의 1.034x 결과를 모든 communication substrate에 일반화할 수는 없다.
다만 raw-best B_eq=128/8-SM에서 D/E/C slowdown이 각각
**1.2717x/1.0579x/1.1284x**였고, stock vLLM의 DeepEP DBO-on은 correctness를
유지하지 못했다. 또한 이 workload의 stock DeepEP DBO-off는 stock AG/RS보다
1.41–1.75x 느렸다. 그러므로 mechanism 가능성은 남지만 최종 FlashVEP
adaptation을 구현할 GO 근거는 부족하다.

DeepEP, stock DBO, 또는 DeepEP 자체의 성능은 FlashVEP novelty로 주장하지
않는다. 이 PoC의 기여 범위는 substrate별 contention과 integration gap을
분리해 측정한 것이다.

## 범위와 환경

- 물리 GPU: 4,5,6,7만 노출한 H100 80GB 4장
- 모델/topology: Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1
- vLLM 0.20.0, PyTorch 2.11.0+cu129, NCCL 2.28.9
- 별도 환경: `/home/esjung/.venvs/flashvep-deepep-v020`
- DeepEP: 정확히 `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`
- NVSHMEM 3.3.24, SM90 build; 기존 `flashvep-poc` 환경은 변경하지 않음
- capture: layer 24, 799 tokens 중 vision 784, hidden 2048, top-k 8,
  experts 128, SHA256 `208e789c...3eda0`
- B_eq는 같은 실제 request capture의 controlled replication이며 실제 serving
  batch로 해석하지 않는다.

원래 요청된 `docs/spec/flashvep_deepep_overlap_revalidation_poc_spec.md`는 작업
시작 시 archive에 없었다. 내용이 일치하는 685-line
`docs/flashvep_deepep_overlap_revalidation_poc_spec_ko.md`와 요청 prompt 전체를
source of truth로 사용했으며, publication branch에는 요청된 정확한 경로에도
spec을 보존한다.

## 구현 및 backend proof

4-rank smoke는 BF16 hidden 2048/top-k 8/experts 128에서 실제 DeepEP
dispatch/combine을 실행했다. 모든 rank에서 expected 512 tokens를 수신했고,
route/weight/order 보존과 combine error 0을 확인했다.

실제 vLLM 객체 proof는 다음과 같다.

| 구성 | all-to-all manager | prepare/finalize | expert |
|---|---|---|---|
| AG/RS, DBO off | `AgRsAll2AllManager` | `MoEPrepareAndFinalizeNaiveDPEPModular` | `TritonExperts` |
| DeepEP HT, DBO off | `DeepEPHTAll2AllManager` | `DeepEPHTPrepareAndFinalize` | `TritonExperts` |
| DeepEP HT, DBO on | `DeepEPHTAll2AllManager` | `DeepEPHTPrepareAndFinalize` + `UBatchWrapper` (2 ubatches) | `TritonExperts` |

Operator replay는 위 모델에 실제 로드된 layer-24 Triton expert weight/backend와
DeepEP `Buffer.dispatch/combine`을 재사용한다. scheduler 영향을 제거한 K2
wavefront는 D(next)∥E(current), C(previous)∥E(current)를 실행한다. vLLM
workspace가 다음 expert call에 storage를 재사용하므로 reduced expert output을
microbatch별로 double-buffering했다. 이 copy는 serial과 overlap의 E 양쪽에
동일하게 포함했다. 최초 shape JIT는 warmup에서만 lockstep으로 priming했고
측정 반복에는 barrier를 넣지 않았다.

## Stock end-to-end 비교

896×896 vision request, warmup 5, 측정 20회의 critical DP-rank median이다.

| Global requests | AG/RS off (ms) | DeepEP HT off (ms) | AG/RS/DeepEP | DeepEP slowdown |
|---:|---:|---:|---:|---:|
| 1 | 2363.587 | 4080.818 | 0.579x | 1.727x |
| 4 | 2413.875 | 4219.627 | 0.572x | 1.748x |
| 8 | 2520.405 | 3549.768 | 0.710x | 1.408x |
| 16 | 2714.765 | 4609.748 | 0.589x | 1.698x |

두 DBO-off 구성은 모두 output token 1986으로 correctness PASS였다. 이 결과는
DeepEP communication kernel만의 비교가 아니라 vision encoder를 포함한 stock
request wall이므로, DeepEP가 느린 원인을 통신 kernel 하나로 단정하지 않는다.

DeepEP DBO-on은 실제 `UBatchWrapper`, 2 ubatches, 20 communication SM까지
확인했지만 유효한 speedup을 산출하지 못했다.

- request 8 재측정: 동일 prompt의 output token이 51292/1986/198/7516 등으로
  달라져 correctness FAIL. median 약 4.190 s는 성능값으로 채택하지 않음.
- request 16: `batch_size must be equal to batch_size_k` FlashAttention shape
  오류로 종료. 이전 full run의 hang은 해당 오류 뒤 process teardown 과정이었다.

따라서 stock DBO speedup은 `null`이며, 성공으로 보고하지 않는다.

## DeepEP operator replay matrix

모든 값은 네 rank의 median 중 critical(max) rank를 사용한다. `Speedup`은
full-serial K1 / overlap K2, slowdown은 공정한 micro-serial K2 대비이다.

| B_eq | SM | Full serial (ms) | Overlap (ms) | Speedup | D slow | E slow | C slow | min-rank overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20 | 4.268 | 3.407 | 1.253x | 1.426x | 1.054x | 1.503x | 0.600 |
| 32 | 16 | 3.999 | 3.528 | 1.134x | 1.437x | 1.053x | 1.496x | 0.601 |
| 32 | 12 | 4.398 | 3.758 | 1.170x | 1.349x | 1.049x | 1.486x | 0.649 |
| 32 | 8 | 5.277 | 4.214 | 1.252x | 1.225x | 1.059x | 1.232x | 0.817 |
| 32 | 4 | 7.867 | 6.414 | 1.227x | 1.125x | 1.047x | 1.005x | 1.000 |
| 64 | 20 | 7.136 | 6.318 | 1.130x | 1.244x | 1.044x | 1.466x | 0.545 |
| 64 | 16 | 7.665 | 6.576 | 1.166x | 1.252x | 1.044x | 1.474x | 0.574 |
| 64 | 12 | 8.562 | 7.060 | 1.213x | 1.274x | 1.052x | 1.476x | 0.637 |
| 64 | 8 | 10.316 | 7.992 | 1.291x | 1.252x | 1.059x | 1.139x | 0.850 |
| 64 | 4 | 15.527 | 12.711 | 1.222x | 1.126x | 1.041x | 1.008x | 1.000 |
| 128 | 20 | 14.250 | 12.231 | 1.165x | 1.306x | 1.044x | 1.515x | 0.573 |
| 128 | 16 | 15.287 | 12.750 | 1.199x | 1.321x | 1.044x | 1.496x | 0.610 |
| 128 | 12 | 17.111 | 13.748 | 1.245x | 1.320x | 1.053x | 1.456x | 0.706 |
| 128 | 8 | 20.604 | 15.728 | **1.310x** | **1.272x** | **1.058x** | **1.128x** | 0.876 |
| 128 | 4 | 31.067 | 25.427 | 1.222x | 1.156x | 1.037x | 1.006x | 1.000 |

24 SM은 vLLM manager가 20 SM 최대 budget으로 buffer를 초기화하므로
`UNSUPPORTED`다. 20/16/12/8/4만 DeepEP API로 설정했다. K4는 K2 필수
비교가 이미 integration 결정을 드러냈고 fragmentation/working-set 위험이
추가되므로 실행하지 않았다.

### Raw-best stage breakdown와 oracle

B_eq=128/8-SM의 critical-rank CUDA-event breakdown:

| Variant | D (ms) | E (ms) | C (ms) | Wall (ms) |
|---|---:|---:|---:|---:|
| micro-serial K2 | 6.696 | 9.283 | 6.463 | 20.553 |
| overlap K2 | 8.516 | 9.821 | 7.293 | 15.728 |

- full-serial K1 wall: 20.604 ms
- achieved: 1.3100x vs full serial, 1.3068x vs micro-serial
- stage ideal lower bound: `max(E, D+C) = 13.160 ms`
- stage-sum oracle ceiling: `(D+E+C)/max(E,D+C) = 1.7054x`
- achieved/oracle은 구분한다. stage interval 합에는 launch/dependency 경계가
  포함되므로 oracle은 end-to-end wall 예측값이 아니라 단순 ceiling이다.

8-SM raw-best는 D/E gate를 조금 넘는다. 반면 B64/4-SM과 B128/4-SM은 각각
약 1.222x speedup을 유지하면서 D/E slowdown이 1.126/1.041x와
1.156/1.037x로 limit 안에 들어온다. 즉 SM budget에 따른 adaptive signal은
있지만 stock DBO gate를 대체하지는 않는다.

## 기존 AllGather/ReduceScatter wavefront와 비교

같은 layer-24 capture를 사용한 이전 일반 `torch.distributed`
AllGather/ReduceScatter replay의 최고점은 B_eq=64/K2에서 full serial 대비
**1.0343x**였다. micro-serial 대비 slowdown은 D **3.585x**(+258.5%),
E **1.050x**(+5.0%), C **1.315x**(+31.5%)였다. 이번 DeepEP raw-best는
full serial 대비 **1.3100x**이고 D/E/C slowdown은 **1.272/1.058/1.128x**다.

따라서 기존 1.034x의 주된 한계가 모든 EP substrate에 내재한 것은 아니며,
일반 collective의 dispatch contention이 중요한 요인이었다는 가설을
지지한다. 다만 이전 replay와 이번 replay는 통신 layout과 harness가 다르므로
이 비교만으로 원인 기여도를 정량 분해하지는 않는다. 또한 이 개선은 DeepEP
substrate/stock DBO 자체의 성과이며 FlashVEP novelty로 주장하지 않는다.

## Correctness

- DeepEP smoke: all ranks PASS, max/mean error 0
- operator 15 configurations × 4 ranks: 모두 PASS
- reference: 각 설정의 DeepEP full serial
- `rtol=1e-2`, `atol=1e-2`
- 전체 operator max absolute error: **0.0**
- 최소 cosine similarity: **0.9999997896**
- route identity, top-k weight identity, source token order restoration: PASS
- stock AG/RS off와 DeepEP off: PASS
- stock DeepEP DBO on: FAIL (별도 표시)

초기 harness 시도에서 발견한 API/stream/JIT/workspace 문제는
`attempts/`에 보존했다. 특히 double-buffer 전 overlap만 correctness가
실패했던 결과를 성공값으로 사용하지 않았다.

## Nsight Systems actual overlap

raw-best B_eq=128/8-SM, K2, warmup 1, 측정 3회를 Nsight Systems 2024.6.2로
수집했다. 대형 32 MiB `.nsys-rep`와 72 MiB SQLite는 로컬
`large_local_artifacts/`에만 두고 commit하지 않는다.

판정은 **PASS**다.

- 모든 네 worker에서 DeepEP intranode dispatch와 `fused_moe_kernel` overlap
  4쌍, DeepEP combine과 expert overlap 4쌍을 관측했다.
- communication stream은 22, expert streams는 75/79로 서로 다르다.
- rank별 GPU interval intersection 합: 8.996–12.116 ms
- 최대 단일 intersection: 1.271–1.869 ms
- Nsight NVTX-attributed kernel-sum 최악 slowdown: D 1.3224x, E 1.0850x
- asynchronous combine launch는 host NVTX attribution으로 C kernel-sum을
  완전하게 귀속할 수 없어 C는 corrected CUDA-event 1.1284x를 사용한다.
- 실제 concurrent residency는 확정하지만 HBM/L2 contention 원인은 이
  trace만으로 직접 확정하지 않고 추정으로 남긴다.

## Gate

| Gate | 결과 |
|---|---|
| DeepEP correctness | PASS |
| actual DeepEP backend proof | PASS |
| Nsight actual kernel overlap | PASS |
| raw-best E slowdown ≤1.05x | FAIL (event 1.058x; Nsight max 1.085x) |
| raw-best D slowdown ≤1.25x | FAIL (event 1.272x; Nsight max 1.322x) |
| B64 또는 B128 ≥1.15x | PASS |
| 나머지 B64/B128 ≥1.10x | PASS |
| stock DBO end-to-end benefit | FAIL / invalid correctness |
| adaptive signal | PASS (SM별 trade-off) |

최종 **HOLD**의 핵심은 operator mechanism은 강하지만 stock integration에서
correctness와 end-to-end benefit이 검증되지 않았다는 점이다. 이는 환경 문제로
core 실험을 못 한 BLOCKED도 아니며, optimized operator gain이 없는 NO-GO도
아니다.

## 수행하지 않은 작업

- K4 replay: K2 뒤 fragmentation gate로 실행하지 않음
- 24 communication SM: API/buffer 최대 20이라 UNSUPPORTED
- stock DBO speedup 수치: correctness/runtime failure 때문에 산출하지 않음
- 최종 FlashVEP adaptation, Attention/Router 확장: GO 이후 작업이므로 미실행
- HBM/L2 contention 원인 확정: Nsight interval만으로는 미확정
- `.nsys-rep`/SQLite commit: 대형 artifact 정책에 따라 미실행

## 단 하나의 권장 다음 작업

**Qwen3-VL + DeepEP HT의 stock vLLM 0.20 DBO-on에서 request별 token 혼선과
FlashAttention batch shape 오류를 먼저 최소 재현·수정하고, 동일 deterministic
stock DBO on/off matrix를 다시 실행하라.** 이 correctness가 해결되기 전에는
operator-level 1.31x를 최종 FlashVEP integration 근거로 승격하지 않는다.

## 산출물

- main report: `poc_flashvep/reports/deepep_overlap_revalidation_report.md`
- gate: `poc_flashvep/deepep_revalidation/results/deepep_revalidation_20260806_164431/gate.json`
- operator matrix: 같은 디렉터리의 `operator_matrix.json`
- backend matrix: 같은 디렉터리의 `backend_matrix.json`
- correctness: 같은 디렉터리의 `correctness.json`
- Nsight evidence: `nsight_summary.json`, `nsight_summary.txt`,
  `nsight_overlap_pairs.csv`, `nsight_kernel_summary_cuda_gpu_kern_sum.csv`

## Git publication

- remote: `https://github.com/jyssys/MLLM-EP.git`
- branch: `flashvep/deepep-overlap-revalidation`
- commit message: `poc: revalidate FlashVEP overlap with DeepEP`
- 이 보고서는 자신을 포함하는 commit SHA를 self-record할 수 없으므로 실제
  SHA와 push 성공 여부는 최종 handoff에 기록한다.
