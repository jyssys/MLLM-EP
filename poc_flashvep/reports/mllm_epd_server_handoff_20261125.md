# MLLM EPD Prefill-heavy: 서버 이전 요약

현재 MLLM EPD 실험만 인수인계한다. 새로운 GPU 성능 측정은 실행하지 않았다.
기존 결과·실패·Raw·코드·보고서는 보존했다.

과학 결과 Commit은 `38e67f91f0099dc546d470c83ff03af638f7af16`이다.
Branch는 `flashvep/mllm-epd-prefill-heavy-token-split-20261124`다.
기존 최종 선택 목록은 14,978개 파일, 합계1,645,785,957 Bytes다. Network 전송량은 아니다.

최종 연구 판정은 `HOLD`다. 검사한 Token Runtime은 `NO_GO_ON_TESTED_GRID`다.
전체 품질·고부하 Prefill-heavy·미래 Decode-safe Window는 미확정이다.
부분 정합성과 Physical Idle을 전체 품질 또는 안전한 Serving 개선으로 표기하지 않는다.

새 서버의 Primary는 원문 기반 동결 Main100이다. 부모 Main400과 Calibration32도 보존한다.
실행용 실제 이미지와 Token Metadata는 합계820,809,959 Bytes다.
Main400+Calibration32의 Unique Request432개, 파일3,566개를 포함한다.
Model Weight 약16.60GB와 중간 Tensor250,822,942,944 Bytes는 GitHub에 포함하지 않는다.
대형 중간 Tensor는 새로운 Serving 실행에 필수 입력이 아니다. 원래 서버에서 보존한다.

이전 명령과 실행환경은 [인수인계 안내](../mllm_epd_handoff/README.md)에 있다.
새 서버용 [재개 프롬프트](../mllm_epd_handoff/resume_prompt_ko.md)를 제공한다.
경로 변환 도구는 파생 Manifest만 만든다. 기존 PID·UUID·Budget을 복사하지 않는다.
새 서버 연결과 GPU 실행은 아직 검증하지 않았다. Resource Guard와 N1→N8 검사가 먼저다.

새 도구의 CPU6개 검사와 기존 MLLM CPU252개 검사를 통과했다.
Replay Asset3,566개와 Manifest3개의 SHA256 검사를 통과했다.
사용자 Checkout·Index·동결 파일 보존 검사도 통과했다.
이번 인수인계에서 기존 모델·Evidence·사용자 파일을 삭제하지 않았다.
