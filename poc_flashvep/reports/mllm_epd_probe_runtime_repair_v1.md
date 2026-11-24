# Online Probe Harness의 준비 검사와 역할 계약 수정

기존 `interference_PROBE_C40R1`은 첫 Common Native 정합성 실행에서 중단됐다.
E 버퍼 반환 검사에서 Live Image1개와 Outstanding Export0개가 남았다.
Probe 성능 결과는 생성하지 않았다. 별도 Stock Native 실행 결과는 보존했다.

## 실제 실패 원인

새 Proxy의 Port18845 Bind가 실패했다.
이전 연구 소유 `metadata_B1_grid_only` Proxy가 같은 포트에서 실행 중이었다.
기존 준비 검사는 HTTP200만 확인했다. 서로 다른 Proxy의 응답을 새 서버의 준비 완료로 받아들였다.
실제 `/health` 이름과 Port Owner PID385981이 이 잘못된 연결을 확인했다.
해당 이전 Handler는 현재 E/D 실행의 버퍼 반환 정책을 적용하지 않았다.

이 결과를 CUDA IPC 자체의 상태 손상으로 단정하지 않는다.
실패한 경로는 올바른 새 Proxy를 사용하지 않았다.
원래 Failure JSON, Bind Error, Request Raw와 잘못된 Ownership 상태는 보존했다.

## 통제한 수정

필요한 포트가 비어 있는지 GPU 초기화 전에 확인한다.
준비 응답의 실제 이름이 새 Coordinator 이름과 같아야 한다.
같은 응답을 받은 뒤에도 새 Process가 실행 중이어야 한다.
실제 Launcher의 잘못된 응답 수락을 재현한 CPU 검사가 수정 전 실패하고 수정 후 통과했다.

이전 Metadata Proxy3개의 PID는 385979·385980·385981이었다.
기록한 PID·소유 UID·Session·Command·Health 이름을 확인했다.
연결된 기존 모델의 모든 Lease가 종료됐고 Pending Request가 0인 것도 확인했다.
그 세 연구 소유 Proxy만 종료했다. 다른 사용자 Process는 종료하지 않았다.

## 별도의 D Teacher RPC 계약 문제

정합성 Preflight에서 D Prefix Probe용 Teacher RPC도 검사했다.
실제 Worker RPC는 `pd`만 허용했다. 요청 경로는 명시적으로 승인한 `decode` Probe도 지원했다.
이 역할 계약 충돌은 별도의 CPU Red/Green 검사로 재현했다.
관측된 Port Collision의 원인과 이 별도 결함을 혼합하지 않는다.

수정한 RPC는 Prefix Probe가 켜진 Worker에서만 동작한다.
`decode`는 추가 `MLLM_EPD_ALLOW_DECODE_PROBE=1`을 요구한다.
승인하지 않은 Decode·Prefill 역할과 일반 Native 요청은 계속 거부하거나 원래 경로를 유지한다.
Assertion을 제거하지 않았다. Model·Attention·MRoPE·KV와 출력은 수정하지 않았다.

## 재실행 범위

새 Namespace는 `interference_PROBE_C40R2`다.
후속 실행은 `suffix_QUANTA_C41R1`과 `whole_DEFLECTION_C43R1`이다.
기존 C41/C43은 실패한 선행 실행의 Gate에서 멈췄다. GPU Lease를 얻지 않았다.
기존 Config·로그·Manifest는 덮어쓰지 않았다.

Main100·Arrival·SLO·128/512 Token Probe·Acceptance Rule은 변경하지 않았다.
Supplemental Whole-deflection Holdout도 기존 동결 Manifest와 동일한지 확인한다.
CPU 검사 통과는 GPU 정합성이나 안전한 Online Headroom의 증거가 아니다.
새 실행의 N1/N8·KV·Decode 보호 결과는 별도로 검사해야 한다.

실패와 수정의 증거는 다음 경로에 있다.

- `state/stale_metadata_proxy_port_collision_v0.json`
- `state/probe_harness_repair_decision_v0.json`
- `logs/proxy_identity_RED_v0.log`와 `logs/proxy_identity_GREEN_v0.log`
- `logs/decode_probe_teacher_rpc_RED_v0.log`와 `logs/decode_probe_teacher_rpc_GREEN_v0.log`

경로 기준은 `mllm_epd_prefill_heavy_token_split_20261124/`다.
