# H200 동일 Cut의 Whole-image 및 Slice-aware CUDA IPC

**ISOLATED_GPU_MEASUREMENT.** H200에서 30개 고정 cut을 전체 이미지 복제와 필요한 row 복제로 각각 두 번 실행했다. Crossing-image pair 38개 모두에서 D clone payload, D 구간 E NVLink TX, D NVLink RX가 줄었다. D First-token HTTP는 20/38에서 줄었고 중앙값 절감은 5.20ms였다. 지연의 범위는 −109.12~+89.30ms로 변동이 컸다. [Paired CSV](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/same_cut_pairs.csv) · [수치 및 provenance](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/same_cut_timing_summary.json)

## 실험

같은 요청과 실제 cut 위치에서 원본 E/P/D 경로로 실행했다. Cut마다 입력, suffix query 수, 인과 attention pair 수, prefix KV block 수를 맞췄다. 두 arm의 NIXL byte counter가 60/60 pair에서 일치했다. 두 arm의 첫 greedy token도 60/60 일치했다. D 측 transport flag만 교체했고, 같은 BI0 장치 epoch에서 arm 순서를 교차했다.

GPU 4/5/6을 E/P/D로 사용했다. 승인된 나머지 GPU 다섯 대에서는 burner가 동작했다. NVLink는 P 완료 뒤부터 D의 첫 토큰 응답 직후까지 읽은 장치별 counter다. D HTTP는 KV import, EC, GPU forward, HTTP 응답을 포함한다. [실행 계획](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/same_cut_plan.json) · [120개 실행 원자료](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/same_cut_trials.jsonl)

| Whole − Slice | Crossing 38 pair 중앙값 | 양의 절감 |
| --- | ---: | ---: |
| D clone payload | 32.81 MiB | 38/38 |
| D clone storage 합계 | 32.81 MiB | 38/38 |
| D 구간 E NVLink TX | 32.81 MiB | 38/38 |
| D 구간 D NVLink RX | 32.81 MiB | 38/38 |
| D clone operation 수 | 0 | 0/38 |
| D descriptor operation 수 | 0 | 0/38 |
| D clone host enclosure | 0.57 ms | 24/38 |
| D First-token HTTP | 5.20 ms | 20/38 |
| E backing storage snapshot | 0 MiB | 0/38 |

나머지 22 pair는 D clone payload 변화가 0이었다. 이 대조군에서 E TX의 paired 차이는 −6~+4 KiB였다. Crossing pair의 E TX 절감 범위는 0.927–53.247 MiB였다. Clone storage 합계는 개별 생성 tensor의 관측 storage 크기다. 동시 요청별 peak live memory는 `NOT_MEASURED`다.

## 7,823-token 단일 이미지, 19-token suffix

단일 이미지의 visual span은 `[15, 7805)`이며 고정 prefix는 7,804 tokens다. D가 필요로 한 visual embedding은 마지막 한 row였다. D clone 형상이 `[7790, 3584]`에서 `[1, 3584]`로 바뀌었다. D clone payload는 55,838,720 → 7,168 bytes였고, P는 두 arm 모두 전체 이미지를 clone했다. D 구간 E TX는 whole에서 55,838,720 / 55,836,672 bytes, slice에서 5,120 / 3,072 bytes였다. NIXL payload는 양쪽 모두 447,741,952 bytes였다. 두 transport 모두 clone 한 건과 ACK 한 건이 필요했다. [관측 clone 형상과 counter](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/reference_L7823_n19_clones.json)

이 cut에서 D HTTP는 whole 67.52 / 55.78ms, slice 70.30 / 164.90ms였다. 7,805-token prefix의 인접 경계에서는 D clone과 해당 E TX가 두 arm 모두 0이었다. Slice는 경계의 byte 불연속을 크게 줄였다. 이미지 한 row라도 쓰는 경우 descriptor와 clone 요청이 남았으며, 이 사례의 두 번 반복만으로 boundary preference의 순이익을 확정할 수 없다. [6개 측정 cut의 실제 counter](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/figures/same_request_cut_sweep.png) · [전체 paired 비교](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/figures/same_cut_transfer_ablation.png)

## Slice 이후 남은 이미지 경계 비용

사전에 정한 9개 입력에서 image-edge와 바로 앞 interior cut을 비교했다. 각 transport의 18 pair에서 interior suffix는 한 토큰 더 길고, NIXL counter는 동일했다. [인접 cut 원자료](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slice_transfer/intra_vs_edge_by_transport.csv)

| Interior − Edge | Whole 중앙값 | Slice 중앙값 |
| --- | ---: | ---: |
| D clone payload 증가 | 32.81 MiB | 7 KiB |
| E NVLink TX 증가 | 32.81 MiB | 7 KiB |
| D clone / descriptor 증가 | 1 / 1 | 1 / 1 |
| D clone host enclosure 증가 | 9.98 ms | 10.04 ms |
| D First-token HTTP 증가 | 13.58 ms | 20.23 ms |

Slice의 추가 E TX는 3–10 KiB였다. D clone과 descriptor는 18/18에서 한 건씩 늘었다. Slice의 D HTTP는 15/18에서 증가했으며 차이는 −23.64~+127.97ms였다. 전송량 불연속의 대부분은 제거됐지만 한 row의 CUDA IPC import와 clone에 필요한 host 경로가 남았다. 실제 scheduler에서 이 비용을 경계 preference로 줄일 수 있는지는 입증되지 않았다.

## 정합성 및 해석

120개 timing 요청에서 서로 다른 1,368개 실제 media identifier를 관측했다. 모든 요청이 E의 clone ACK와 buffer 해제를 완료했다. 같은 이미지의 중복 요청은 이 timing arm에서 순차적으로 실행됐다. GPU에서 동시 동일 이미지 요청의 abort/timeout을 강제하는 회귀는 포함하지 않았다.

BI0의 첫 별도 Full28/Natural correctness probe는 D 요청의 `ignore_eos`와 E/P의 값이 달라 processor fingerprint 검사에서 종료됐다. D forward 전에 거부됐으므로 `FAILED_EXECUTION`으로 남긴다. 위 timing pair의 계약과 결과에는 영향을 주지 않았다.

같은 cut의 byte 절감은 transport에 귀속한다. 이번 실험은 강한 Slice-aware Generic에 modality scheduling을 더했을 때의 Active C3 TTFT, ITL, TPOT 또는 Task Quality에 대한 결과를 제공하지 않는다.

두 그림은 commit의 paired CSV를 입력으로 `poc_flashvep.mllm_epd_h200_dynamic_correctness_slice_aware.transfer_figures`를 실행해 재생성할 수 있다.
