# Joint Encode–Prefill discovery 실험 계약

최초 코드는 `d76e9785e0928811051c9fd8dd32bf5c7497df5b`에서 시작했다. 이전 108개 실험에서는 두 recipient 모두 조기 제출을 수락했으며 E가 전체 요청에 응답하기 전 완료된 Visual LM 행은 없었다. D에 보이는 IPC descriptor만으로 실제 CUDA 준비 여부를 판정할 수 없었다. 별도 세 GPU EPD의 E/P/D 실행 비중과 idle 규모는 기존 DIAGNOSTIC 범위를 유지한다. [CUDA-ready Follow-up](mllm_epd_ready_prefix_recipient_followup.md) · [공통 구간 계측](mllm_epd_idle_motivation_qualification.md)

## 고정 설정

- Host `henn-0`. H200 GPU4/5/6을 E/P/D로 할당했다. 다른 다섯 GPU의 승인된 Burner를 유지했다.
- GPU4–6은 NVSwitch의 NV18 경로와 NUMA 1을 공유했다. Driver 580.126.16, Python 3.14.6, PyTorch 2.11.0+cu129, vLLM 0.26.0+cu129, Transformers 5.16.1, NIXL 1.4.1을 사용했다. vLLM wheel SHA-256은 `6ce4ca30616f0a35810391015622b197a7b8b267ed27f8716f0789db79ff578b`다.
- Qwen2.5-VL-7B-Instruct `cc594898137f460bfe9f0759e9844b3ce807cfb5`. BF16, 원본 image grid·MRoPE·FA3·CUDA Graph. P/D B8192/S128, worker별 KV 42 GiB, CUDA IPC 및 NIXL.
- APC와 요청 간 Encoder Embedding 재사용을 사용하지 않는다. 실제 이미지의 occurrence별 식별자를 분리했다. 원본 Very-long Document QA의 전체 프롬프트와 Fixed16을 사용한다.
- Native Open-loop 기준은 Poisson QPS1.465와 seed 2026112809다. 물리 비교의 짧은 prefix probe 및 내부 discarded sample은 서비스 성능 실험과 별도다.
- 후보의 온라인 승격 기준은 Native 대비 TPOT p95 및 ITL p99 각각 +5% 이하, 절대 Decode SLO 95% 이상 및 Native 비열등, TTFT p95 15% 이상 감소, 같은 자원에서 원본 품질 유지다. 절대 SLO는 TTFT 5,000 ms, TPOT 33 ms, 각 ITL 34 ms다.

실제 모델, 런타임·소스 파일·이미지 해시, GPU UUID와 프로세스 소유권을 각 run의 `state/environment.json`, `state/lease.json` 및 `configs/frozen_plan.json`으로 기록한다. 모든 GPU 작업에는 승인된 리소스 lease와 종료 후 Burner 복원을 적용한다. 독립 실험의 P/D 포트와 GPU 그룹을 공유하므로 각 epoch는 순서대로 실행했다. [기존 환경 및 모델 계약](mllm_epd_modality_aware_idle_execution_poc_spec.md)

## A. CUDA-ready Encoder 그룹

E가 실제 `GPUModelRunner._execute_mm_encoder()`에 전달하는 `scheduled_encoder_inputs`를 원래 순서 그대로 나눈다. 각 그룹에 대해 원래 Vision forward와 native CUDA IPC export를 호출한다. 해당 호출 이후 같은 E CUDA stream에 end event를 기록한다. 완료를 확인한 뒤 그 그룹의 `cuda_group_ready` marker를 쓴다. 동기 sweep은 E가 직접 완료를 기다린다. 비동기 sweep은 별도 host thread가 기록 순서대로 CUDA event를 조회하며 E가 후속 그룹을 발행할 수 있게 한다. `whole`과 `awhole`은 같은 방식으로 전체 그룹 하나를 사용하고, `stock`은 이전 native worker 경로를 사용한다. 그룹화, event 대기와 원본 IPC 호출 비용은 후보 실행 시간에 들어간다.

일치하는 2×2 비교는 Late/Early × P/D다. 같은 문서와 이미지 종료 cut, 같은 recipient 내부의 prompt shape, 같은 E 그룹을 유지한다. Early P는 이후 정상 NIXL P→D를 수행한다. Early D는 같은 D에서 계속한다. 각 arm의 TTFT에는 최초 요청 ingress 이후 모두 포함한다. E의 실제 CUDA event 기록, 준비 marker, HTTP ACK, P/D의 실제 Visual LM retirement는 다른 시간 경계다. E GPU event와 recipient retirement 사이의 인과 순서는 증명하되 동시 커널 합집합은 별도 계측으로 판단한다.

온라인 pilot의 grouped Late P, Early P와 Guarded D는 같은 Encoder 모드와 앞 8장 이후 causal cut을 사용한다. Guarded D는 IPC 준비 시점에 P HTTP 작업이 두 개 이상 남고 D HTTP 요청이 없을 때 D를 선택하며, D worker에서 Native 요청이 대기 또는 실행 중이면 다음 추가 prefill chunk를 발행하지 않는다. 이미 발행된 LM chunk는 최대 512개 행이며 실제 채널 도착과 그 지연은 관측에 포함한다. Native와 Whole-request Isolation은 `stock` Encoder로 같은 도착열과 모델 설정을 사용한다. [온라인 비교의 구성과 계측](mllm_epd_joint_execution_online_serving.md)

## B. D-local Joint Encode–Prefill

한 이미지 또는 연속된 이미지 그룹을 D에서 실제 Vision으로 인코딩하고 원래 causal KV에 연결한다. Prefix probe가 P를 실제 지원하려면 반환 KV까지 포함해야 한다. Suffix 경로는 P의 앞부분을 실제 NIXL로 수신한 뒤 D에서 후속 이미지의 원래 pixel/grid로 Vision과 LM을 계속한다. 같은 cut에서 E가 suffix embedding을 준비하는 generic 제어 및 전체 Native EPD와 대조한다. E와 P가 생략한 작업을 검증하고 D의 CUDA/HBM 동기화 및 전송을 포함해 후보의 50/100/150/250/500 ms 유효량을 보고한다.

## C와 증거 판정

C는 모델 원문과 설치된 코드, 구성·weight 접근을 확인하고 원래 연산의 의존성을 검증한다. 부족한 배포 기능은 `NOT_RUN`으로 표시한다. `MEASURED_ONLINE`은 실제 open-loop 호출에서 사용한다. 실제 장치 tensor/forward/transfer의 직렬 또는 부하 제어는 `MEASURED_PHYSICAL`이다. 계측 보정을 확인하지 못한 활동 시간은 `DIAGNOSTIC`이다. 추가 이득, 정확성 및 Decode 보호 물리 게이트가 없으면 신규 온라인 통합을 진행할 근거가 부족하다. [모델 구조](mllm_epd_architecture_specific_parallelism_discovery.md) · [기존 시스템과 비교](mllm_epd_joint_execution_prior_work.md)

## H200 재실행

트리 루트에서 실행한다. 각 archive의 `frozen_plan.json`에는 원본 자산 roster와 실제 요청·그룹 설정이 기록되어 있다. 다른 환경에서는 같은 SHA-256의 로컬 fixture 및 모델로 absolute roster 경로를 바꾸고, 사용 가능한 GPU lease의 포트 및 소유권을 확인한다. 아래 경로의 H200 전용 `*-plan.json`은 보존한 로컬 원본이며 각 재실행의 출력 디렉터리는 새 경로다.

```bash
../mllm_epd_runtime_env/bin/python \
  -m poc_flashvep.mllm_epd_joint_encode_prefill_discovery.group_runner \
  --plan ../h200_joint_encode_prefill_runs/group_async_plan.json \
  --run-dir ../h200_joint_encode_prefill_runs/group-async-replay \
  --inline-reroute --wall-seconds 2800

MLLM_JOINT_D_QUANTUM_ROWS=512 ../mllm_epd_runtime_env/bin/python \
  -m poc_flashvep.mllm_epd_joint_encode_prefill_discovery.group_runner \
  --plan ../h200_joint_encode_prefill_runs/online_pilot_plan.json \
  --run-dir ../h200_joint_encode_prefill_runs/group-online-replay \
  --inline-reroute --wall-seconds 2600
```

`awhole`, `a2`, `a4`, `a8`, `av4096`는 비동기 CUDA 완료 대조다. 물리 결과에서 실행 비용과 ready 지점을 함께 확인해, 온라인 비교에 미리 구성한 `a8` 모드를 사용했다. [고정 부하와 온라인 집계](mllm_epd_joint_execution_online_serving.md) · [세부 B 재현](mllm_epd_joint_encode_prefill_quantum.md)
