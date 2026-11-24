# Compiled Numerical Path 원인 분석 — C21R4 / C24

첫 Layer의 RMSNorm 차이는 Controlled Intervention으로 제거했다. 전체 Prompt KV 차이는 남았다.
증거 수준은 `DIAGNOSTIC`이다. 독립 초기화는 조건별 1회다. 성능과 전체 Task Quality는 미측정이다.

## 실행 계약

원본 Qwen2.5-VL-7B-Instruct BF16, BI0, Greedy, Full28을 유지했다.
모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
GPU4/5/6은 E/P/D다. GPU UUID는 `state/environment_initial.json`과 각 Server Lease에 보존했다.
Calibration Request0은 Input4831 Tokens와 이미지 1개다. Prefix2415 / Suffix2416을 사용했다.
B8192/S1024, KV42GiB, Block128, Prefix Cache Disabled를 유지했다.
Teacher와 Token 모두 같은 Query Chunk 경계를 사용했다. CPU Tensor 복사는 모든 Timing을 무효화한다.
검사는 N1이다. Online C3, Mixed Batch 또는 전체 Dataset 품질 검사로 해석하지 않는다.

## 첫 차이의 위치

C21R4에서 첫 Attention Q/K/V, Attention 출력과 O Projection은 P/D 사이에서 Strict Match였다.
다음 RMSNorm의 Residual과 Weight도 Byte 단위로 같았다.
Norm 출력의 Relative L2는 5.115058311611417e-6이었다. Max Absolute Difference는 0.001953125였다.
실제 P Launcher의 R0_BLOCK은 2048이었다. 실제 D Launcher는 4096이었다.
XBLOCK1, Num Warps16, Num Stages1은 같았다. Cached best_config만으로 실제 Launcher를 판정하지 않았다.
원래 FX Graph와 생성 Kernel Source가 같아도 Runtime Reduction Tile은 다를 수 있었다.

## One-factor Intervention과 반증

C24는 P/D 공통 `inductor_compile_config.dynamic_scale_rblock=false`만 추가했다.
Compiled 경로와 BI0는 유지했다. E 경로, 원본 Weight와 Attention은 변경하지 않았다.
이 설정은 새 Numerical 하위 계약이다. Stock Primary 결과와 섞지 않는다.
이 경우 실제 P/D Norm0 R0_BLOCK은 모두 4096이었다.
첫 Norm과 관측한 네 GEMM의 Input/Output이 모두 Strict Match가 됐다.
하지만 Full KV Strict Match는 1/28에서 2/28로만 증가했다.
따라서 첫 Norm 차이의 원인은 지지된다. 전체 KV 차이의 유일한 원인이라는 가설은 반증됐다.

| Mode | Comparison | Strict KV Layers | Maximum Layer Relative L2 | First Four GEMM Strict |
|---|---|---:|---:|---|
| Stock compiled | P_B_vs_P_A | 28/28 | 0 | True |
| Stock compiled | D_teacher_vs_P_A | 1/28 | 0.0245827469939 | False |
| Stock compiled | D_teacher_B_vs_P_A | 1/28 | 0.0245827469939 | False |
| Stock compiled | Token_D_vs_P_A | 1/28 | 0.0233504360104 | False |
| Common fixed reduction diagnostic | P_B_vs_P_A | 28/28 | 0 | True |
| Common fixed reduction diagnostic | D_teacher_vs_P_A | 2/28 | 0.0222575551594 | True |
| Common fixed reduction diagnostic | D_teacher_B_vs_P_A | 2/28 | 0.0222575551594 | True |
| Common fixed reduction diagnostic | Token_D_vs_P_A | 2/28 | 0.0201394372539 | True |

두 조건에서 관측한 Greedy Output16 Tokens는 모든 Arm에서 같았다.
이 한 요청의 출력 일치는 전체 품질 또는 KV Gate PASS가 아니다.
동일 P 반복은 두 조건 모두 28/28 Strict였다. Token 없는 Full D Teacher도 P와 달랐다.
이 결과는 수치 차이를 Token Split 전송 오류만으로 설명하는 가설을 지지하지 않는다.
뒤쪽 Layer의 첫 미일치 Operator는 아직 식별하지 못했다. 허용값과 SLO는 완화하지 않았다.

## 증거와 남은 Gate

Stock Compiled C3 Token 성능은 `NOT_RUN`이다. 전체 정합성 Gate는 `NOT_PASSED`다.
Adaptive Tile 변경의 성능 영향은 `NOT_MEASURED`다. Speedup으로 표기하지 않는다.
Eager 15/15 Strict 결과는 별도 Correctness 근거다. Compiled Baseline 성능을 대체하지 않는다.
C21 계측 Closure와 Local Lifetime 오류는 실패 로그에 보존했다. 모델 정합성 실패와 구분한다.
원시 비교는 `analysis/token_NUMERICAL_C21R4.json`과 `analysis/token_NUMERICAL_C24.json`이다.
재계산 CSV는 `analysis/compiled_numerical_C21_C24_{summary,operators,layers}_v0.csv`다.
Figure는 `figures/compiled_numerical_C21_C24_KV_v0.{png,pdf}`다.
모든 상대 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.
