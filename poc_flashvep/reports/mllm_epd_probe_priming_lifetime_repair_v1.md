# Prefill Probe Priming의 소유권과 HTTP 계약 수정

이 보고서는 Probe 준비 경로의 실패와 CPU 검증을 기록한다.
새 Probe의 N3 간섭 결과나 Safe Headroom을 보고하지 않는다.

## C40R2의 실제 실패

Native와 추가 LLM의 N1/N8 출력 검사는 통과했다.
그 뒤 준비용 E Embedding을 보관하면서 전역 Inline 전달 정책을 되돌리려 했다.
원래 Worker RPC는 Live Image나 Outstanding Export가 있으면 정책 변경을 거부한다.
이 안전 검사가 실제 HTTP500을 발생시켰다.

검사를 제거하거나 Live Image를 완료된 것으로 표시하지 않았다.
사용 중인 Embedding을 조기에 반환하지도 않았다.
실패한 실행은 종료된 연구 소유 Lease와 함께 보존했다.
모델 자체의 정합성 실패로 분류하지 않는다.

## 준비 경로의 수정

원래 Inline P 정책을 계속 유지한다.
원래 E가 준비용 Calibration 이미지 전체를 처리한다.
이어 P가 같은 E 출력을 실제 Full28 Forward로 소비한다.
P는 Remote-KV Flag 없이 폐기할 1-token Sample만 만든다.
이 경로는 원래 Inline Export의 실제 Clone과 ACK를 완료한다.

이후 E Image는 보관하지만 Outstanding Export는 0이어야 한다.
나중에 E/D Helper는 그 원래 Image를 On-demand Clone으로 가져온다.
모든 Probe와 Main 작업이 끝난 뒤에만 준비용 E 소유권을 반환한다.
준비용 P Forward는 모든 측정 구간 밖에 있다. Client Task의 완료나 Throughput에 더하지 않는다.

이 수정은 실제 Worker의 정책 Guard를 거치는 CPU 실패 검사에서 재현했다.
실제 E→P 소비 순서와 ACK 완료를 검사하는 경로는 수정 후 통과했다.
실제 GPU Probe의 상태 검사는 여전히 별도 Preflight를 통과해야 한다.

## C40R3의 HTTP 방식 오류

원래 E 상태 Endpoint는 GET 전용이다.
초기 수정은 POST 기반 RPC Helper로 이 Endpoint를 호출해 실제 HTTP405를 받았다.
이 실행은 Probe Cost나 추가 Main 성능을 생성하지 않았다.
상태 조회는 원래 GET Endpoint를 직접 사용하는 경로로 수정했다.

CPU 검사는 POST를 실제 405 방식으로 거부하는 Transport Fixture를 사용한다.
같은 검사에서 GET은 보관 중인 E Image와 Export 상태를 반환한다.
단순히 임의의 RPC 응답을 반환하는 Fixture로 HTTP 방식을 생략하지 않는다.
기존 두 실패와 수정 전후 로그는 보존했다.

## 독립 연구의 실행 순서

Whole-request Deflection은 이 Probe 준비의 정합성 결과에 의존하지 않는다.
종료된 Probe의 모든 GPU Lease를 확인한 뒤 Whole 기준선을 별도 Namespace에서 실행했다.
Probe의 후속 실행은 Whole 작업이 GPU를 반환한 뒤에만 시작한다.
이 순서는 Resource Dependency다. Whole의 결과를 Probe의 정합성 증거로 빌리지 않는다.

후속 Probe는 새 초기화에서 E/P/D·Helper·128/512 Prefix KV를 먼저 검사한다.
그 Preflight가 통과해야 Main100 N3 간섭 검사를 시작한다.
Dataset·Arrival·SLO·Quantum·Probe 수락 Rule은 변경하지 않았다.

## 재현 자료

- C40R2 실패: `state/interference_PROBE_C40R2_failure.json`
- C40R3 실패: `state/interference_PROBE_C40R3_failure.json`
- 소유권 수정: `state/probe_priming_lifetime_repair_v0.json`
- 독립 실행 순서: `state/whole_baseline_independent_priority_v0.json`
- 소유권 Red/Green: `logs/probe_priming_lifetime_{RED,GREEN}_v0.log`
- HTTP Red/Green: `logs/probe_encoder_state_HTTP_{RED,GREEN}_v0.log`

경로 기준은 `mllm_epd_prefill_heavy_token_split_20261124/`다.
