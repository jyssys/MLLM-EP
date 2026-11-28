# 원본 이미지의 준비 경계에서 P/D causal prefill 실행

## 실험 계약

이전 H200 실행에서 실제 E 파이프라인의 이미지별 IPC 이벤트에 나타난 선행 구간이 실행 기회인지 검증한다. Qwen2.5-VL-7B-Instruct `cc594898137f460bfe9f0759e9844b3ce807cfb5`, H200의 E/P/D GPU4/5/6, BF16, P/D B8192/S128, 각 KV42GiB, 원래 FA3와 MRoPE, CUDA Graph, CUDA IPC와 NIXL을 유지했다. APC와 요청 간 이미지 임베딩 재사용을 끄고 각 원본 요청에 Fixed16을 적용했다. [이전 이미지 준비 근거](mllm_epd_readiness_aware_recipient_execution.md) · [고정 런타임](mllm_epd_modality_aware_idle_execution_poc_spec.md)

진단에서 25/50 ms 이상의 선행이 관측됐던 원본 요청 5개와 선행이 입증되지 않은 대조 1개를 사전 동결했다. 각 접두사는 원래 `<|vision_end|>`까지 포함하며 1,152–7,739 LM 토큰과 2–15장이다. 7,739토큰의 두 요청은 원본 prefix의 이미지 순서, 토큰 및 MRoPE 위치가 동일하고 질문이 다르다. 코호트 선택은 실행 전의 선행 진단을 활용했다. [고정 코호트 영수증](../mllm_epd_modality_aware_idle_execution_discovery/results/readiness_followup_cases.json)

| 비교군 | 동일 prefix의 실행 시작 시점 | 후속 실행 경로 |
| --- | --- | --- |
| Early P | 원래 prefix의 inline IPC descriptor가 모두 보이면 제출 | 같은 P에서 suffix 실행, 전체 원본 KV를 NIXL로 D에 전달 |
| Late P | 전체 E HTTP 응답 이후 제출 | Early P와 같은 경계 및 P→D 경로 |
| Early D | 같은 원래 prefix의 inline IPC descriptor가 모두 보이면 제출 | 같은 D에서 suffix와 Decode, E→D 임베딩과 로컬 KV 사용 |
| Late D | 전체 E HTTP 응답 이후 제출 | Early D와 같은 경계 및 D 내부 경로 |

각 Arm은 전체 원본 입력을 한 번 접수한다. AsyncScheduler는 원래 이미지 순서와 종료 토큰을 검증하고 첫 단계의 LM 행을 고정 cut까지 제한한다. 후속 이미지 descriptor가 모두 준비되면 같은 요청의 paged KV로 원래 suffix를 수행한다. 기존 E CUDA IPC를 단일 소비자가 claim하고 file ACK를 사용한다. 원본 이미지 누락 시 로컬 Vision fallback은 금지되어 있다. 모든 Arm에 동일 cut을 적용하여 다른 batch/query shape로 인한 처리량 및 수치 차이를 줄였다. [실행기](../mllm_epd_modality_aware_idle_execution_discovery/readiness_followup_runner.py) · [Scheduler](../mllm_epd_modality_aware_idle_execution_discovery/readiness_scheduler.py)

입력 준비는 inline descriptor의 가용성으로 추적한다. 해당 consumer의 CUDA event가 실제 임베딩 의존성을 보장한다. 접두사 진행은 Scheduler에 제출된 토큰 수 대신 `update_from_output`으로 확인한 GPU retirement 행에서 원래 image span과 겹치는 부분으로 계산한다. 먼저 descriptor가 관측된 시각, 마지막 descriptor가 관측된 시각, 전체 E HTTP 응답은 서로 다른 경계다. 이 실험은 CUDA kernel 전체 구간이나 늦은 이미지의 Vision 연산 중첩을 독립적으로 계측하지 않았다.

한 번에 원본 한 요청을 실행하는 2×2 물리 비교다. Arm의 순서를 fixture와 반복 사이에 회전시켰다. D에서 계속 진행해 없어진 전체 P→D handoff 비용과 조기 readiness의 추가 효과를 구분하기 위해 각 recipient 내부 Early/Late 차이와 두 차이의 차이를 각각 구한다. TPOT·ITL은 같은 요청의 관측 SSE token timestamp를 사용한다. 부하가 겹치는 요청의 Decode 보호나 실제 open-loop Goodput은 이 설계의 측정 범위에 포함되지 않는다.

## 실제 원본 요청 6개의 결과

**72/72 완료. Early P/D 36/36의 제출이 E의 전체 HTTP ACK보다 빨랐다. 그중 0/36이 전체 E ACK나 마지막 inline descriptor의 첫 양성 관측 전에 첫 Visual LM 행을 retire했다.** 72/72의 전체 prompt token digest는 원본 processor NPZ와 동일했다. 모든 Arm은 직렬 독립 요청의 절대 Decode SLO를 통과했다. [실측 자료](../mllm_epd_modality_aware_idle_execution_discovery/evidence/readiness_followup/matched.manifest.json) · [짝지은 실측 비교](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/readiness_followup_paired.svg)

| TTFT 차이, 양수는 조기 제출의 절약 | 전체 18쌍 중앙값 | 첫 반복 뒤 12쌍 중앙값 |
| --- | ---: | ---: |
| Late P − Early P | −1.08 ms | −1.08 ms |
| Late D − Early D | +19.42 ms | +10.52 ms |
| 요청별 D 절약 − P 절약 | +5.08 ms | +5.08 ms |

Late D는 Late P보다 중앙값 93.63 ms 빨랐다. Early D와 Early P의 차이는 98.86 ms였다. 이 비교에는 D에서 전체 Prefill을 완료해 P→D KV handoff를 없애는 차이도 들어 있다. 같은 경로에서 early/late만 바꾼 비교의 차이는 5.08 ms였다. 첫 Late P 샘플에는 E ACK만 10.79초 정도 지연된 큰 관측값이 있었다. 위 표는 이를 포함하며 첫 반복 이후만 계산한 민감도도 함께 제공한다.

4,084토큰 입력 접두사가 가장 긴 실제 descriptor 간격을 보였다. Early D의 첫 prefix descriptor 집합과 마지막 descriptor 관측 사이 간격은 434–472 ms였다. 그럼에도 4,084개 행의 실제 retirement는 E ACK보다 82.6–97.7 ms 뒤였다. 첫 반복에서 10장, 29,030,400 B의 원본 E 임베딩을 받는 IPC import·clone·동기화의 host 구간은 D 384.6 ms, P 412.6 ms였다. 두 경로 모두 동기 export RPC는 0 ms였고 ACK는 파일 방식이었다. 이 시간에는 원본 CUDA event를 기다리는 구간이 들어 있다.

같은 recipient 안에서는 Early/Late의 생성 token이 P 18/18, D 18/18 일치했다. 네 Arm이 전부 일치한 것은 15/18이었다. 나머지 셋은 동일 4,084토큰 문서의 P→D와 D-local 경로 사이에서 달랐다. 전체 task 품질이나 모든 layer의 bitwise KV를 이 결과에서 추정하지 않았다.

## 같은 문서의 quantum과 CUDA event 확인

4,084토큰 사례의 원본 문서 하나를 그대로 써서 cut을 421, 828, 4,084토큰으로 좁혔다. 첫 1장, 2장, 10장의 원래 image closing delimiter에 각각 해당한다. 모든 shape와 recipient를 먼저 prime한 뒤 독립 시작에서 조기·늦은 Arm을 반복했다. E의 원래 CUDA IPC CUDA event 자체를 query한 진단을 추가했다. 최대 prefix event completion 상한과 최대 전체 event completion 하한의 차이는 실제 준비 선행의 보수적 하한이다. `inline descriptor`가 처음 보인 시각과 별도로 계산했다. [같은 문서 cut 계약](../mllm_epd_modality_aware_idle_execution_discovery/results/readiness_followup_cases_shaped.json)

**36/36 완료. Early P와 Early D 각각 9/9가 E의 HTTP 응답 전에 제출됐다. Early Arm 18개 모두 첫 Visual LM retirement가 전체 E CUDA event 완료 관측 상한 이후였다.** D의 첫 retirement는 그 상한보다 31.72–427.26 ms 뒤였다. 마지막 descriptor 최초 관측 및 E HTTP ACK 기준으로도 Early Arm의 선행 retirement는 0/18이었다. E에서 2,580개의 producer event 완료를 관측했고 drop은 0이었다. CUDA query 계측의 5% observer 보정은 별도 실행하지 않았으므로 이 attribution 에폭은 `DIAGNOSTIC`이다. [GPU 준비 진단 원자료](../mllm_epd_modality_aware_idle_execution_discovery/evidence/readiness_followup/shaped.manifest.json) · [독립 실행 결과](../mllm_epd_modality_aware_idle_execution_discovery/results/readiness_followup_shaped/)

Early D의 descriptor 관측 선행 중앙값은 442.76 ms였다. 실제 E producer event 선행 중앙값은 하한 5.83 ms와 상한 422.10 ms 사이였다. 이 넓은 구간 때문에 실제 준비 선행 시간을 한 값으로 특정할 수 없다. 첫 prefix IPC import·clone·동기화의 host 구간 중앙값은 Early P/D 417.95/397.16 ms, Late P/D 75.74/27.22 ms였다. 이 구간에는 CUDA event 대기와 동기화가 포함되며, 순수 DMA 시간은 별도로 측정하지 않았다. 조기 제출은 호스트 준비와 대기 일부를 겹칠 수 있었지만 E의 전체 준비 전 완료한 Visual LM 행은 없었다.

| 고정 cut | P의 TTFT 절약 중앙값 | D의 TTFT 절약 중앙값 | 요청별 D 절약 − P 절약 중앙값 |
| ---: | ---: | ---: | ---: |
| 421 | 127.34 ms | 112.91 ms | −14.43 ms |
| 828 | 197.64 ms | 95.37 ms | −75.30 ms |
| 4,084 | 149.08 ms | 110.13 ms | −51.80 ms |
| 전체 9 quartet | 151.98 ms | 110.13 ms | −51.80 ms |

Shaped의 36개 prompt digest가 원본과 일치했고 동일 recipient의 Early/Late 출력 token도 P 9/9, D 9/9 일치했다. 동일 cut의 P→D와 D-local 출력이 전부 일치한 quartet은 0/9였다. 36개 모두 직렬 실행의 절대 Decode SLO를 충족했다. 다중 요청 Decode 보호와 전체 태스크 품질은 미측정이다.

## 판정

**현재 Native E producer와 같은 recipient에서 이어가는 방식은 `NO_GO_INCREMENTAL`이다.** 선행 descriptor만으로 D의 완료 가능한 Visual LM 작업을 확보하지 못했고, 조기 제출의 효과를 같은 기준의 Early P도 얻었다. 처음 72개의 paired D 추가 절약 중앙값은 5.08 ms였고, shape를 prime한 별도 진단에서는 −51.80 ms였다. 따라서 D 전용 온라인 통합으로 승격하지 않았다. 다음 반증 조건은 E에서 원래 이미지 그룹을 순서대로 실제 계산·발행해 prefix CUDA event의 선행 시점을 좁은 오차로 증명하고, 동일한 E producer로 Early P와 Early D를 다시 비교하는 것이다.

## 재현

```bash
runtime=../mllm_epd_runtime_env/bin/python
raw=../h200_readiness_followup_runs
study=poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery
manifest=../h200_idle_execution_runs/dataset/primary_very_long_n24.json
receipt=poc_flashvep/mllm_epd_modality_aware_idle_execution_discovery/results/readiness_prefix.json
"$runtime" -m "$study.readiness_followup_cases" --manifest "$manifest" \
  --readiness "$receipt" --out "$raw/cases.json"
"$runtime" -m "$study.readiness_followup_cases" --manifest "$manifest" \
  --readiness "$receipt" --same-doc-shorter --out "$raw/cases_shaped.json"
"$runtime" -m "$study.readiness_followup_runner" --plan "$raw/plan.json" \
  --run-dir "$raw/repro-matched" --inline-reroute --wall-seconds 1800
"$runtime" -m "$study.readiness_followup_runner" --plan "$raw/plan_shaped.json" \
  --run-dir "$raw/repro-shaped" --inline-reroute --image-ready --wall-seconds 1800
```

두 에폭의 `configs/frozen_plan.json`, 실제 GPU·런타임·소스 SHA, trial/retirement 로그 및 producer event를 각 증거 아카이브에 보관했다. 경로를 재배치하는 경우 아카이브의 manifest, `readiness.cases` 및 dataset root를 로컬 원본 값에 맞춘다. 승인된 GPU0–7의 `train_lm.py` 복원을 확인했다. [Reducer](../mllm_epd_modality_aware_idle_execution_discovery/readiness_followup_reduce.py) · [GPU 복원](../mllm_epd_modality_aware_idle_execution_discovery/results/readiness_followup_burner_restore.json)
