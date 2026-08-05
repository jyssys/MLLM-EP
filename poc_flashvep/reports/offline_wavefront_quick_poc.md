# FlashVEP offline expert-centered wavefront Quick PoC

- 결론: **NO-GO**
- Core 최고 speedup: **1.034x**
- 결과 디렉터리: `/home/esjung/MLLM-EP/poc_flashvep/results/offline_wavefront_quick_poc_20260805_130322`
- 범위: layer 24 Core D/E/C만; vLLM request는 capture 1회에만 사용
- Workload: `synthetic batch scaling from real captured request`
- Backend: 실제 `TritonExperts`, 명시적 NCCL all-gather/reduce-scatter

## Capture

- token: 799 (vision 784)
- hidden/intermediate/top-k: 2048 / 768 / 8
- experts: 128 total, 32 per EP rank
- runtime DPEP chunks: `[400, 400, 1, 1]`

## Routed workload

| B_eq | real/vision tokens | assignments | rank assignments | critical assignments | max local-expert tokens | active local experts/rank |
|---:|---:|---:|---|---:|---:|---|
| 16 | 12784/12544 | 102272 | `[18720, 27600, 24448, 31504]` | 31504 | 5808 | `[31, 28, 23, 28]` |
| 32 | 25568/25088 | 204544 | `[37440, 55200, 48896, 63008]` | 63008 | 11616 | `[31, 28, 23, 28]` |
| 64 | 51136/50176 | 409088 | `[74880, 110400, 97792, 126016]` | 126016 | 23232 | `[31, 28, 23, 28]` |
| 128 | 102272/100352 | 818176 | `[149760, 220800, 195584, 252032]` | 252032 | 46464 | `[31, 28, 23, 28]` |

## O1 — full-batch serial D→E→C

| B_eq | tokens | D@critical ms | E@critical ms | C@critical ms | E max ms | MoE critical ms | expert-max % | M assignments/s | critical rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 12784 | 0.457 | 1.071 | 0.552 | 1.466 | 2.099 | 69.8 | 48.72 | 2 |
| 32 | 25568 | 0.377 | 2.301 | 0.282 | 2.301 | 2.977 | 77.3 | 68.71 | 3 |
| 64 | 51136 | 0.674 | 4.002 | 1.065 | 4.526 | 5.756 | 78.6 | 71.07 | 2 |
| 128 | 102272 | 1.199 | 8.093 | 1.974 | 9.063 | 11.281 | 80.3 | 72.53 | 2 |

`E max`는 rank별 expert duration의 최대값이고, `D/E/C@critical`은 같은
critical-wall rank의 coherent breakdown입니다. 따라서 별도 rank의 stage max를
더해 MoE wall과 비교하지 않습니다.

## O2 — expert-centered wavefront

| B_eq | K | full ms | micro-serial ms | wavefront ms | vs full | vs micro | frag. | repeat coll. | min-rank overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 2.991 | 3.220 | 3.128 | 0.960x | 1.029x | 4.9% | 5.8% | 55.0% |
| 32 | 4 | 2.991 | 3.741 | 3.412 | 0.878x | 1.095x | 15.9% | 22.3% | 71.0% |
| 64 | 2 | 5.755 | 5.914 | 5.567 | 1.034x | 1.063x | 1.7% | 1.9% | 63.0% |
| 64 | 4 | 5.755 | 6.368 | 5.857 | 0.982x | 1.088x | 6.5% | 8.6% | 76.9% |

### Critical-rank stage breakdown

| B_eq/K | variant | D sum ms | E sum ms | C sum ms | wall ms |
|---:|---|---:|---:|---:|---:|
| 32/2 | full serial | 0.385 | 2.302 | 0.284 | 2.988 |
| 32/2 | micro serial | 0.458 | 2.413 | 0.320 | 3.218 |
| 32/2 | wavefront | 1.250 | 2.650 | 0.422 | 3.128 |
| 32/4 | micro serial | 0.625 | 2.668 | 0.396 | 3.741 |
| 32/4 | wavefront | 1.691 | 2.993 | 0.538 | 3.410 |
| 64/2 | full serial | 0.670 | 4.528 | 0.543 | 5.754 |
| 64/2 | micro serial | 0.710 | 4.606 | 0.569 | 5.912 |
| 64/2 | wavefront | 2.544 | 4.837 | 0.748 | 5.565 |
| 64/4 | micro serial | 0.859 | 4.821 | 0.643 | 6.368 |
| 64/4 | wavefront | 3.407 | 5.329 | 0.949 | 5.854 |

Wavefront 행에서 stage 합이 wall보다 큰 것은 측정된 동시 실행 때문입니다.
최적점 B64/K2는 D/E overlap 1.969–2.336 ms, E/C overlap 0.377–0.460 ms,
rank별 overlap fraction 63.0–80.0%였습니다. 그러나 micro-serial 대비
wavefront의 critical-rank D/E/C duration은 각각 258.5%/5.0%/31.5% 늘어,
overlap의 대부분을 NCCL/compute contention이 상쇄했습니다.

## Correctness / overlap / gate

- assert_close 전체 rank/config: `True`
- max abs / max-rank mean abs / min cosine: `0` / `0` / `0.999999914`
- route identity와 token/order restoration: `PASS`
- 실제 CUDA event overlap: `True`
- fragmentation <15%: `False`
- gain > uncertainty: `True`
- rank 균형: `True` (최적점의 rank별 speedup 1.027–1.034x)
- memory 합리성: `True`

O1 최대 allocated peak는 B128에서 25.27 GiB/GPU였고, 최적 B64/K2는
wavefront 19.80 GiB 대 full-batch 20.82 GiB였습니다. Nsight/profiler는
실행하지 않았으므로 별도 profiler overhead는 N/A이며, uncertainty는 30회
embedded CUDA-event 표본으로 평가했습니다(B64/K2 speedup p10 1.0168x,
median 1.0343x, p90 1.0380x).

Attention/Router 확장: **미수행** — not executed because core best speedup did not exceed 1.10x.

최종 판정은 **NO-GO**입니다.

다음 단 하나의 작업: 이 결과를 archive하고 이번 mechanism branch를
중단합니다. Attention/Router 또는 vLLM 통합으로 확장하지 않습니다.
