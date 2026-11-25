# MLLM EPD Prefill-heavy: 새 서버 인수인계

현재 MLLM 실험만 다룬다. Dense AR 연구를 다시 실행하지 않는다.
실험 결과 Commit은 `38e67f91f0099dc546d470c83ff03af638f7af16`이다.
Branch는 `flashvep/mllm-epd-prefill-heavy-token-split-20261124`다.

## 보존한 결과

- [최종 연구 요약](../reports/mllm_epd_prefill_heavy_token_split_final_summary.md)
- [Native/Token Online 결과](../reports/mllm_epd_generic_token_split_online_serving.md)
- [정합성 검사](../reports/mllm_epd_generic_token_split_correctness.md)
- [E/D Idle 최종 요약](../reports/mllm_epd_encoder_decoder_idle_final_summary.md)
- [Provenance 예외](../reports/mllm_epd_final_provenance_discrepancies_v1.md)

연구 판정은 `HOLD`다. 검사한 Token Runtime은 `NO_GO_ON_TESTED_GRID`다.
P 절감보다 D Admission 대기 증가가 컸다. 고부하 Prefill-heavy와 미래 Decode-safe Window는 미입증이다.
전체 Task Quality도 HOLD다. 이 한계를 새 서버의 검증 PASS로 바꾸지 않는다.

## 이전할 항목

GitHub에는 코드·구성·분석·Figure·압축 Raw와 원본 Manifest를 보존했다.
마지막 과학 결과의 선택 목록은 14,978개 파일, 합계 1,645,785,957 Bytes다.
이는 파일 크기의 합계다. Git 압축 후 Network 전송량이 아니다.

| 항목 | 이전 방법 | 크기 및 범위 |
|---|---|---|
| 코드·보고서·공개 Evidence | 아래 Branch Clone | 이전 Commit과 실패 결과 포함 |
| 실제 이미지·Token Metadata | `transfer_20261125/replay_assets.rsync.txt` | 820,809,959 Bytes. Main400+Calibration32의 432개 Unique Request |
| Qwen2.5-VL 원본 Model | 아래 Revision Download 또는 Snapshot 복사 | 원본 Model 파일 약16.60GB. GitHub에 Weight를 넣지 않음 |
| 대형 중간 Tensor | 필요할 때 원래 서버에서 별도 복사 | 원본 계측 Tensor 250,822,942,944 Bytes. 새 Serving 실행에 필수 아님 |

Main100은 부모 Main400의 부분집합이다. 기존 Manifest와 모든 Tail을 그대로 같다고 주장하지 않는다.
공개 Raw Archive와 대형 Tensor의 Hash Audit는 원래 Evidence Namespace에 있다.
이 도구는 모델·이미지·대형 Tensor를 GitHub에 공개 업로드하지 않는다.

## 최소 준비

1. 새 서버에서 이 Branch를 Clone한다. 기존 사용자 Checkout을 덮어쓰지 않는다.

```bash
git clone --single-branch --branch flashvep/mllm-epd-prefill-heavy-token-split-20261124 https://github.com/jyssys/MLLM-EP.git
```

2. 원래 서버에서 다음 명령으로 이미지와 Metadata만 복사한다. `USER`, `HOST`, `NEW_REPO`는 실제 값으로 바꾼다.

```bash
rsync -av --files-from=/home/esjung/MLLM-EP-github/poc_flashvep/mllm_epd_handoff/transfer_20261125/replay_assets.rsync.txt /home/esjung/MLLM-EP-github/ USER@HOST:NEW_REPO/
```

3. 새 서버의 저장소 Root에서 Hash를 검사한다. Python3.10 이상이 필요하다.

```bash
python3 -m poc_flashvep.mllm_epd_handoff.assets verify --repo "$PWD" --catalog poc_flashvep/mllm_epd_handoff/transfer_20261125/replay_assets.json
python3 -m poc_flashvep.mllm_epd_handoff.assets prepare --repo "$PWD" --catalog poc_flashvep/mllm_epd_handoff/transfer_20261125/replay_assets.json --output ../mllm_epd_new_session
```

`prepare`는 새 Manifest의 실행 경로만 변경한다. 원문·Prompt Hash·Token Metadata와 원래 Manifest는 보존한다.
원본 Parquet/PDF 경로는 Provenance로 남긴다. 준비 완료는 GPU 정합성 PASS가 아니다.

## Runtime 계약과 재개 순서

Model과 Processor Revision은 모두 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Model은 `Qwen/Qwen2.5-VL-7B-Instruct`다. BF16·BI0·Greedy·Natural EOS·Full28을 유지한다.
Primary는 Main100이다. P/D는 B8192/S1024·KV42GiB·Block128·Prefix Cache Disabled다.
E는 B114688·Renderer2·MM Cache0·Sync Scheduling·Custom Ops None이다.

실제 원래 환경은 Python3.14.6, vLLM0.26.0+cu129, Torch2.11.0+cu129, Transformers5.16.1, NIXL1.4.1이다.
상세 관측값은 `source_environment.json`에 있다. 새 서버에서 지원·동작한다고 아직 검증하지 않았다.
구버전 설치 안내는 [vLLM0.26.0 문서](https://docs.vllm.ai/en/v0.26.0/getting_started/installation/gpu/)를 사용한다.
최신 버전으로 자동 Upgrade하지 않는다.

설치된 vLLM Wheel Receipt Hash는 `7632856147650da3ed8d1652b1b05ffaadcc62ea8e910fdaa6f8ce055b201ebf`다.
현재 공식 같은 이름 Asset의 Hash는 `6ce4ca30616f0a35810391015622b197a7b8b267ed27f8716f0789db79ff578b`다.
같은 Version 문자열만으로 원래 Binary와 같다고 판정하지 않는다. 새 Runtime은 N1→N8을 다시 검사한다.

기존 `common.py`와 Resource Guard에는 원래 서버의 절대 경로·GPU UUID·PID·Budget이 있다.
기존 Campaign Controller를 그대로 실행하지 않는다. 새 Namespace와 소유권 Record부터 만든다.
원래 PID, Burner Lease와 Preservation Record를 새 서버 권한으로 재사용하지 않는다.
이 패키지는 서버를 시작하거나 Process를 종료하지 않는다.

다음 단계는 [새 서버 시작 프롬프트](resume_prompt_ko.md)를 새 Codex 대화에 붙여 넣는 것이다.
새 서버 SSH 접속과 GPU 실행은 아직 수행하지 않았다.

## 이번 인수인계 검증

새 도구의 CPU6개 검사와 기존 MLLM CPU252개 검사를 통과했다.
실제 Replay Asset3,566개와 원본 Manifest3개의 SHA256을 다시 검사했다.
기존 Checkout·Index·동결 파일 보존 검사를 통과했다. 이번 작업에서 새 GPU 성능 실험은 실행하지 않았다.
