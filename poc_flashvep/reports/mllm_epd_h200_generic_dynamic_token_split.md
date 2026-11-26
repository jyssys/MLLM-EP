# Method B Generic Dynamic Split

실행과 admission은 Method A와 공유한다. 후보는 0/5/10/20/30%와 실제 가능한 P progress다. P가 이미 지난 위치는 다시 고르지 않는다. D가 사용할 수 있는 시간은 현재 관측한 요청과 queue 상태, P의 다음 completion 예상, transfer 상태와 실제 비용의 보수적 추정으로 제한한다. 안전한 선택이 없으면 P가 원래 요청을 계속 수행한다.

| 비교 | 한 번에 바꾸는 요소 |
| --- | --- |
| Idle10 → A | 예약 및 P continuation을 동반한 결정 시점. 모든 실행 변경을 기록 |
| A → B | 같은 실행기에서 fixed target과 dynamic budget 선택 |
| Early dynamic → B | 해당 counterfactual의 실제 fallback이 동등할 때만 결정 시점 |
| B → C | 동일 후보 · 동일 planner에서 modality 정보만 추가하는 기본 대조. 추가 후보의 독립 arm |

오프라인에서 실제 미래 idle 길이를 사용하는 결과는 OFFLINE_ORACLE이다. 온라인 기록에는 선택 후보, 비용과 여유, 예약 응답과 거절 이유, 실제 실행 및 추가 Native 지연을 남긴다.

## 동결한 구현 정책

초기 forced-cut 구현의 공통 후보는 5/10/20/30% nominal cut과 각 목표 ±256 tokens 내 가장 가까운 image edge였다. B/Hard/Soft가 같은 후보를 검사한다. 별도의 pure Generic은 같은 최대 후보 예산에서 정수 위치와 Block128-aligned 후보를 사용한다. Placebo는 요청별 Hard와 같은 후보 개수를 고정한다. Forced-cut 후보의 동시성 정합성은 미충족이었다.

현재 P의 다른 요청에서 아직 처리하지 않은 토큰을 optimistic 80,000 tok/s로 나눈 최소를 다음 P completion의 lead로 삼는다. Budget은 최대 160 ms이며 최소 여유 10 ms를 비용에 반영한다. 이미 알고 있는 P work에 관한 heuristic이라 미래 Arrival과 실제 idle 길이를 보장하지 않는다. 비용은 isolated import-to-first-token과 train-side p95 residual을 포함한다. Native-Matched는 late cut 이후 D에서 필요한 embedding을 받을 수 있도록 같은 on-demand E descriptor path를 사용한다. NativeStrong과 Free의 빠른 실용적 기준은 이전 inline 경로를 유지한다.

후속 natural-hold 비교에서 B와 Hard는 stock P가 발행한 동일한 후보 기회를 사용한다. QPS5 shadow에는 명목 10/20%를 적용했고 실제 ratio band는 7.5–15% 및 15–25%였다. Method A는 5–15% band에서 late idle을 검사한다. Dynamic에서는 실제 입력/prefix/suffix가 같은 inline transport의 Calibration support 범위를 벗어나면 거절한다. Shadow는 실제 coordinator와 worker 상태를 조회한 뒤 예약 가능성만 기록하고 P 전체 prefill을 계속한다.

두 staged correctness fixture에서 실제 D suffix와 Native follower가 한 배치에 합류했고 Full28 tolerance는 각각 8/8이었다. 전체 Greedy가 각각 7/8에 그쳐 active A/B/C의 timed C3는 NOT_RUN이다. [Mixed 정합성](mllm_epd_h200_late_bound_batch_preserving_split.md) · [Shadow 실험과 온라인 증거](mllm_epd_h200_generic_vs_modality_aware_online_serving.md)
