# 새 서버 Codex에 붙여 넣을 지시문

이 새 서버에서 MLLM EPD Prefill-heavy의 실험 환경을 다시 설정해줘.
지금은 환경 준비와 최소 정합성 검사가 목적이야. 대규모 Sweep은 아직 시작하지 마.
Dense AR, Layer/Hybrid, Video 실험은 재개하지 마.

Repository: https://github.com/jyssys/MLLM-EP
Branch: `flashvep/mllm-epd-prefill-heavy-token-split-20261124`

먼저 이 Branch를 별도 경로에 Clone해줘. 가능하면 `--depth 1 --single-branch`를 사용해줘.
다음 파일을 순서대로 읽어줘.

1. `poc_flashvep/mllm_epd_handoff/README.md`
2. `poc_flashvep/mllm_epd_handoff/resume_prompt_ko.md`
3. `poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_final_summary.md`

Primary Dataset은 기존 동결 Main100이야. Dataset과 SLO를 다시 고르지 마.
Model은 Qwen2.5-VL-7B-Instruct야. README의 Model/Processor Revision과 Runtime 계약을 유지해줘.
원래 서버에서 파일을 옮기거나 내게 SSH 주소·수동 복사를 요청하지 마.
Token Metadata432개는 GitHub에 포함했어. 원래 NPZ와 Hash를 그대로 사용해줘.
실제 이미지와 Model Weight는 README의 `bootstrap_download`로 공식 Source에서 자동 다운로드·재구성해줘.
기본 범위는 동결 Main100+Calibration32야. 기존 Revision·Request ID·이미지 순서·질문을 유지해줘.
필요한 Parquet/PDF만 받아줘. Dataset 전체를 다시 Sampling하거나 Natural Output을 재생성해서 재선정하지 마.
이미지·Pixel·PDF·Model Hash가 다르면 원인을 분석해줘. 다른 Sample이나 다른 Revision으로 조용히 대체하지 마.
접근 권한이나 네트워크 문제를 해결할 수 없다면 실패를 기록하고 필요한 접근 권한만 알려줘.
기존 MLLM 대형 Tensor와 과거 AR Tensor는 다운로드하지 마. 필요한 새 Teacher Tensor는 새 서버에서 생성해줘.

새 서버의 GPU 종류·개수·UUID·Topology·사용 중인 Process·디스크를 먼저 확인해줘.
다른 사용자 작업과 기존 Checkout을 보호해줘. 원래 서버 PID·GPU UUID·Budget·Burner Lease를 복사하지 마.
기존 코드의 절대 경로와 Resource Guard를 감사하고, 새 Namespace와 소유권 Record를 준비해줘.
`bootstrap_download`의 Asset Hash 검사와 파생 Manifest Receipt를 확인해줘.
원래 Binary와 다른 Runtime을 쓰면 그 차이를 기록해줘. 같은 Version 문자열만으로 동일하다고 판단하지 마.

설정 후 Native N1→N8, Token10 N1→N8 정합성을 먼저 검사해줘.
GPU 사용 범위와 시간 예산은 실행 전에 내게 확인해줘. 승인 없이 Burner를 실행하지 마.
현재 연구는 HOLD이고 Token Runtime은 NO_GO_ON_TESTED_GRID야. 이를 성공 결과로 바꾸지 마.
중간 보고는 최소화해줘. 마지막에 환경 설정·검증·미해결 항목과 실제 실행 명령을 알려줘.
