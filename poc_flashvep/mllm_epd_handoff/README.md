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

## 자동으로 준비할 항목

GitHub에는 코드·구성·분석·Figure·압축 Raw와 원본 Manifest를 보존했다.
마지막 과학 결과의 선택 목록은 14,978개 파일, 합계 1,645,785,957 Bytes다.
이는 파일 크기의 합계다. Git 압축 후 Network 전송량이 아니다.

| 항목 | 준비 방법 | 크기 및 범위 |
|---|---|---|
| 코드·보고서·공개 Evidence | 아래 Branch Clone | 이전 Commit과 실패 결과 포함 |
| Token Metadata | Git Clone으로 받음 | 원래 NPZ432개, 1,815,684 Bytes. 기존 SHA256 유지 |
| 실제 이미지 | 고정 Source의 Parquet/PDF를 다운로드해서 추출·렌더링 | 기존 Request ID와 이미지 순서를 유지. 원래 서버 접속 불필요 |
| Qwen2.5-VL 원본 Model | 고정 Revision에서 자동 다운로드 | 원본 Model 파일 약16.60GB. GitHub에 Weight를 넣지 않음 |
| 대형 중간 Tensor | 다운로드·복사하지 않음 | 새 Serving 실행에 필수 아님. 필요한 새 정합성 Tensor는 새 서버에서 생성 |

Main100은 부모 Main400의 부분집합이다. 기존 Manifest와 모든 Tail을 그대로 같다고 주장하지 않는다.
공개 Raw Archive와 대형 Tensor의 Hash Audit는 원래 Evidence Namespace에 있다.
이 도구는 모델·이미지·대형 Tensor를 GitHub에 공개 업로드하지 않는다. 작은 Token Metadata만 추가로 포함했다.
원래 전체 Replay Asset의820,809,959 Bytes는 최종 이미지·Metadata 크기다. Source 다운로드 용량과 다르다.

## 최소 준비

1. 새 서버에서 이 Branch를 Clone한다. 기존 사용자 Checkout을 덮어쓰지 않는다.

```bash
git clone --depth 1 --single-branch --branch flashvep/mllm-epd-prefill-heavy-token-split-20261124 https://github.com/jyssys/MLLM-EP.git
```

2. 새 저장소 Root에서 CPU 다운로드용 환경을 준비한다. 기존 Runtime 환경을 덮어쓰지 않는다.

```bash
python3 -m venv ../mllm_epd_download_env
../mllm_epd_download_env/bin/python -m pip install -r poc_flashvep/mllm_epd_handoff/download_requirements.txt
```

3. 다음 명령으로 모델과 실제 이미지를 준비한다. SSH·rsync·수동 파일 복사는 필요하지 않다.

```bash
../mllm_epd_download_env/bin/python -m poc_flashvep.mllm_epd_handoff.bootstrap_download --repo "$PWD" --work-dir ../mllm_epd_autosetup --scope main100
```

기본 범위는 동결 Main100+Calibration32의132개 Unique Request다. 다시 Sampling하지 않는다.
부모 Main400까지 모두 준비할 때만 `--scope main400`을 사용한다. 이 범위는432개 Unique Request다.
기본 범위의 Model·Source 파일 크기 합계는 약21.69GB다. 전체432개 범위는 약23.16GB다.
Runtime Wheel과 Git Clone 크기는 포함하지 않는다. Cache 적중 시 실제 Network 전송량은 줄어든다.

`download_plan.json`은 필요한 파일140개의 공식 Repo·Revision·크기·SHA256을 보존한다.
도구는 필요한 파일만 받는다. 전체 Dataset Pool을 재수집하거나 Natural Output으로 Sample을 다시 고르지 않는다.
원래 PDF는 PyMuPDF2.1/MuPDF1.28.6에서72DPI·RGB·전체 페이지·Alpha 없음으로 렌더링했다.
이미지의 원본 Bytes·Pixel Hash·해상도·순서·질문·정답이 다르면 도구는 중단한다.
다른 이미지나 다른 Request로 자동 대체하지 않는다. Renderer 차이를 무시하지 않는다.

작은 Token Metadata는 GitHub에서 받은 원래 파일을 사용한다. 새 Tokenization으로 원본 NPZ를 덮어쓰지 않는다.
도구는 Source와 Asset Hash를 검사하고 새 경로의 파생 Manifest 및 준비 Receipt를 만든다.
원본 Manifest와 Parquet/PDF Provenance 경로는 보존한다. 준비 완료는 GPU 정합성 PASS가 아니다.
Source 접근 권한·네트워크·Pinned File 문제는 새 서버에서 해결하거나 실패로 기록한다. 원래 서버 파일 복사를 요구하지 않는다.
Source License와 접근 조건을 유지한다. 인증이 필요하면 정상적인 접근 권한만 요청한다.
Hugging Face의 [Revision 지정 다운로드](https://huggingface.co/docs/huggingface_hub/guides/download) 경로를 사용한다.

과거 `replay_assets.rsync.txt`는 기존 인수인계 기록으로만 남긴다. 기본 설정 절차에서 사용하지 않는다.

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
원래 서버의 Burner는 이후 사용자 요청으로 종료했다. 별도 승인 없이 다시 실행하지 않는다.

## 이번 인수인계 검증

새 도구의 CPU6개 검사와 기존 MLLM CPU252개 검사를 통과했다.
실제 Replay Asset3,566개와 원본 Manifest3개의 SHA256을 다시 검사했다.
기존 Checkout·Index·동결 파일 보존 검사를 통과했다. 이번 작업에서 새 GPU 성능 실험은 실행하지 않았다.
자동 다운로드 변경의 검증 범위는 `download_bootstrap_validation_20261125.json`에 별도로 기록한다.
새 임시 경로에서 기본132개 Request의 전체 CPU 준비 검사를 통과했다. 기존 생성 이미지는 재사용하지 않았다.
이 검사는 원래 호스트의 공식 Source Cache를 사용했다. 새 서버나 전체 신규 Network 다운로드를 검증한 결과는 아니다.
모델 설정 파일1개는 새 Cache로 실제 다운로드한 뒤 원래 Hash를 확인했다.
다운로드 도구 CPU12개와 기존 MLLM CPU252개 검사도 통과했다. 새 서버의 GPU 검사는 별도다.
