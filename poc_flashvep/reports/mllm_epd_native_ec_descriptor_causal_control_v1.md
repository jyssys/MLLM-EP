# Native EC Descriptor Policy — 한 요소 인과 검사

Native Full28 E/P/D에서 E Descriptor Policy 하나만 변경했다. Token Split과 Whole-request Deflection은 없다.
각 독립 초기화에서 ABBA 또는 BAAB를 실행했다. 각 Policy의 두 Period 평균을 먼저 계산했다.
독립 Start는 3회다. 24개 Cohort를 24회 독립 Start로 취급하지 않았다.
Natural EOS와 Equal-output Replay를 별도로 평가했다. 전체 Task Quality는 PENDING이다.

| Output | Metric | Inline/On-demand | Paired 95% CI |
|---|---|---:|---|
| natural | TTFT_mean_ms | 0.537815 | [0.5292905594679604, 0.5453788682895915] |
| natural | TTFT_p50_ms | 0.977910 | [0.9692434890990451, 0.9889113049118259] |
| natural | TTFT_p95_ms | 0.482137 | [0.4730606013773049, 0.4931211660070255] |
| natural | ITL_mean_ms | 1.004410 | [1.0005981410943388, 1.008566704551216] |
| natural | ITL_p99_ms | 1.044801 | [1.0104953828157677, 1.0868788988297] |
| natural | ITL_max_ms | 1.569249 | [1.3458501174736988, 1.711660742452059] |
| natural | TPOT_p95_ms | 1.027418 | [1.0115312681275832, 1.040975404642252] |
| natural | raw_output_tokens_per_second | 1.092687 | [1.0782638625949008, 1.1120652286587298] |
| natural | total_input_output_tokens_per_second | 1.097397 | [1.0953889715888754, 1.0995809109771637] |
| natural | completed_requests_per_second | 1.097420 | [1.0954294260285216, 1.0996331542313542] |
| natural | SLO_goodput | 1.593137 | [1.4587953590241007, 1.707588254008973] |
| natural | joint_percent | 1.634441 | [1.580877371273713, 1.6743607379834922] |
| natural | decode_pass_percent | 0.966044 | [0.958825561312608, 0.9753554583508103] |
| replay | TTFT_mean_ms | 0.529575 | [0.49935414686630436, 0.5665972026804267] |
| replay | TTFT_p50_ms | 0.980159 | [0.9471094042236741, 1.0154253004161546] |
| replay | TTFT_p95_ms | 0.462210 | [0.42854409190548076, 0.4937165583931533] |
| replay | ITL_mean_ms | 1.003349 | [1.0028646383375126, 1.0037882556642046] |
| replay | ITL_p99_ms | 1.085194 | [1.0671450783154408, 1.1054148800026584] |
| replay | ITL_max_ms | 1.660367 | [1.3927111979976459, 2.0061521040210497] |
| replay | TPOT_p95_ms | 1.017044 | [1.0059224475797293, 1.0267219510552228] |
| replay | raw_output_tokens_per_second | 1.100826 | [1.0884993281229645, 1.112622291232198] |
| replay | total_input_output_tokens_per_second | 1.100826 | [1.0884993281229642, 1.1126222912321981] |
| replay | completed_requests_per_second | 1.100826 | [1.0884993281229647, 1.1126222912321984] |
| replay | SLO_goodput | 1.710421 | [1.6952748858829898, 1.7311465338670087] |
| replay | joint_percent | 1.661721 | [1.6447110423116615, 1.6794599989210763] |
| replay | decode_pass_percent | 0.961409 | [0.9520652173913042, 0.9714646464646465] |

## 해석 범위

두 Policy 모두 원본 E·Full P·NIXL D와 같은 공통 Compiler·Coordinator를 사용했다.
원래 Lifetime Guard가 비어 있는 Image/Descriptor 상태를 확인한 후에만 Policy를 변경했다.
Clone 완료와 모든 ACK를 유지했다. Tensor나 출력값을 수정하지 않았다.
Inline Arm의 모든 실제 P Import에서 Export RPC가 0인지 검사했다. On-demand Import도 원시 기록으로 보존했다.
Import Group에는 여러 이미지나 요청이 포함될 수 있다. Group 시간을 요청별 GPU 비용으로 배분하지 않았다.
Clone/Sync Host Enclosure에는 E Ready Event 대기가 포함될 수 있다. 이를 순수 DMA나 제거 가능한 낭비로 단정하지 않았다.
요청별 다섯 공통 Host 구간의 합은 TTFT와 일치한다. 세부 P 구간을 다시 더하지 않았다.
NativeStrong과 기존 Static Token은 이미 필요한 Descriptor를 미리 발행했다.
이 인과 검사의 개선율을 두 기존 Method의 개선으로 옮기지 않았다.
Independent N3의 27개 Paired Bootstrap 조합을 사용했다. CI 해상도는 제한적이다.
이 결과는 Native 전송 경로의 인과 검사다. Prefill-heavy·Safe Headroom·Online Token Split 성공의 증거는 아니다.

## 재현 정보

Qwen2.5-VL-7B BF16 Revision cc594898137f460bfe9f0759e9844b3ce807cfb5, GPU4/5/6을 사용했다.
Main100, QPS2, B8192/S1024, KV42GiB, Block128, Prefix Cache Off와 원래 SLO를 유지했다.
Config는 `configs/ec_INLINE_CAUSAL_C44.json`이다. 모든 CSV·Audit·Figure는 `ec_INLINE_CAUSAL_C44_*_v0`에 있다.
실제 Clean GPU Batch와 GPU-exclusive Transfer는 NOT_MEASURED다. Finite-cohort 처리량을 Sustainable Capacity로 부르지 않았다.
