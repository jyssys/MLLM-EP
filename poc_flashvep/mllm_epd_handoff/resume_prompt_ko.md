# 새 서버 시작 프롬프트

아래 내용으로 현재 MLLM EPD Prefill-heavy 실험만 재개하라.
Dense AR, Layer/Hybrid, Video 또는 새 Dataset 연구는 시작하지 않는다.

Repository는 https://github.com/jyssys/MLLM-EP 이다.
Branch는 `flashvep/mllm-epd-prefill-heavy-token-split-20261124`다.
기존 과학 결과 Commit은 `38e67f91f0099dc546d470c83ff03af638f7af16`이다.
Branch의 최신 인수인계 문서를 먼저 읽어라.

필수 문서:

1. `poc_flashvep/mllm_epd_handoff/README.md`
2. `poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_final_summary.md`
3. `poc_flashvep/reports/mllm_epd_generic_token_split_correctness.md`
4. `poc_flashvep/reports/mllm_epd_generic_token_split_online_serving.md`
5. `poc_flashvep/reports/mllm_epd_final_provenance_discrepancies_v1.md`

현재 상태는 HOLD다. Token Runtime의 판정은 NO_GO_ON_TESTED_GRID다.
Native 대비 P HTTP 절감이 D Admission 대기로 상쇄됐다. GPU-exclusive 비용으로 해석하지 않는다.
전체 Task Quality와 고부하 Prefill-heavy는 미확정이다. 과거 결과를 새 서버 실측으로 사용하지 않는다.

Model은 `Qwen/Qwen2.5-VL-7B-Instruct`다.
Model/Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
원래 BF16, BI0, Greedy, Full28, Natural EOS Cap2048, Prefix Cache Disabled를 유지한다.
Primary는 동결 Main100이다. 부모 Main400과 Calibration32는 보존한다.
Main100 SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`이다.
P/D B8192/S1024, KV42GiB, Block128을 시작 조건으로 유지한다.
E는 원래 Processor/Encoder를 유지한다. 상세 설정은 README를 따른다.

먼저 새 GPU·UUID·Topology·남은 Memory·사용 중인 Process를 확인하라.
다른 사용자의 작업을 종료하지 않는다. 원래 서버 PID와 Burner Lease를 복사하지 않는다.
원래 `common.py`와 Resource Guard의 경로·Budget·UUID를 감사하라.
기존 Evidence는 수정하지 않는다. 별도 새 서버 Namespace와 Resource Guard를 먼저 준비하라.

`mllm_epd_handoff.assets verify`로 전송한 이미지와 Token Metadata를 검사하라.
`prepare`로 새 경로의 파생 Manifest를 만들어라. 원본 Manifest를 덮어쓰지 않는다.
Model은 고정 Revision으로 확보하라. 원래 환경 목록은 `source_environment.json`에 있다.
vLLM Version 문자열이 같아도 Wheel Hash가 다르면 새 Runtime으로 기록하라.

원래 Encoder Batch Geometry 변화에서 Embedding/KV 차이가 있었다.
10% Token Split N1/N8은 같은 Geometry에서 통과했다. 25% N8은 KV 실패로 제외했다.
새 Runtime에서는 Native N1→N8, Token10 N1→N8을 먼저 검사하라.
Batch 변화의 Token 불일치와 구현 오류를 구분하라. Task Quality Gate를 생략하지 않는다.

실험 재개 순서:

1. 환경·Asset·GPU 소유권 검사와 새 서버 최소 설정.
2. Native EPD 정합성과 QPS0.5 저부하 Smoke Test.
3. 필요한 경우 Native/Agg의 QPS2/4 비교. 동일3GPU·Arrival·Natural Output 계약을 유지.
4. 5% Observer Gate를 통과하는 고부하 E/P/D 계측과 P Queue 검증.
5. P 병목과 실제 Safe Headroom이 확인될 때만 Token Split 후속 비교.

기존 TTFT≤5000ms, 모든 Client ITL≤34ms, TPOT≤33ms/token을 유지한다.
Native-relative ITL p99와 TPOT p95의 Paired CI 상한≤1.05를 별도 검사한다.
독립 Start를 반복 단위로 사용한다. N1과 N3를 혼합하지 않는다.
Physical Idle을 Decode-safe Window나 실제 Speedup으로 표기하지 않는다.
폐기 Sample의 Probe는 실제 요청의 Prefill 절감으로 계산하지 않는다.

새 서버 사용 예산과 Burner 경로는 사용자에게 확인하라.
Burner 복원은 새 서버에서 승인된 Model·GPU·Process에만 수행하라.
중간 보고는 최소화하라. 새 결과와 실패를 보존하고 Commit/Push 후 Remote SHA를 확인하라.
