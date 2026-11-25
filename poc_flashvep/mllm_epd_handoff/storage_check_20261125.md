# 디스크 사용량과 추가600GB 확보 가능성

2026-11-25 16:49 KST 전후에 읽기 전용으로 확인했다. 파일은 삭제하지 않았다.
`GB=10^9 Bytes`, `GiB=2^30 Bytes`, `TiB=2^40 Bytes`다.
아래 사용량은 `du -x -B1`의 할당량이다. 압축 파일의 원본 크기와 다르다.

## 현재 사용량

`/dev/vda1`의 Available은983,979,425,792 Bytes다. `df -h`는917G로 표시했다.
이 파일시스템은 다른 사용자와 공유한다. 전체 사용량을 사용자 개인 사용량으로 해석하지 않는다.

| 경로 기준 | 할당량 GB | 비고 |
|---|---:|---|
| `/home/esjung` 전체 | 1739.05 | 약1.582TiB. Home 경로 기준이며 UID별 전체 파일시스템 집계는 아님 |
| `MLLM-EP-github` | 1345.49 | 대부분 과거 실험 Raw와 Archive |
| `MLLM-EP-github/poc_flashvep/reports` | 1304.68 | 저장소 크기의 주원인 |
| `.cache` 전체 | 165.12 | 아래 모델을 포함 |
| `.cache/huggingface/hub` | 147.78 | 모델과 Dataset Snapshot |
| `.venvs` | 33.55 | 현재 MLLM Runtime도 포함 |
| `anaconda3` | 28.89 | 현재 Python과 Burner Runtime도 포함 |

표의 하위 경로는 상위 경로에 포함된다. 서로 더하지 않는다.
사용자 소유 `/tmp` 최상위 항목의 최대 크기는 약0.200GB였다. 600GB급 원인은 아니었다.

MLLM EPD Namespace는274.74GB다. 그중 Raw는267.15GB다.
Raw의 `llm_reference`가226.38GB를 차지한다. 대형 정합성 Tensor가 주요 부분이다.
이 Tensor는 보존 대상이다. 단순 캐시로 분류하지 않았다.

Hugging Face에서 Qwen3-30B-A3B는61.08GB, 현재 Qwen2.5-VL은16.60GB다.
Qwen3-30B-A3B는 실행 중인 승인 Burner가 사용한다. 현재 MLLM 모델과 함께 보호해야 한다.
캐시 전체165.12GB를 제거해도600GB에는 미달한다. 필수 모델을 제외하면 더 작다.

## 추가600GB를 확보할 수 있는가

캐시 정리만으로는 불가능하다. 과거 Dense AR Raw를 다른 저장소로 이전하면 공간상 가능하다.
다음 세 디렉터리의 합계는692,928,667,648 Bytes다. 약692.93GB 또는645.34GiB다.

| 이전 후보 | 할당량 GB |
|---|---:|
| `reports/dense_ar_longbench400_2d_20261121` | 342.75 |
| `reports/dense_ar_longbench400_v2_20261122` | 236.76 |
| `reports/dense_ar_p_critical_path_20261123` | 113.41 |

이 값은 이전 후보의 전체 할당량이다. 즉시 삭제 가능한 중복량이 아니다.
일부 Archive와 Raw가 함께 있다. 파일별 완전 복원 검사는 이번 용량 조사에서 수행하지 않았다.
GitHub Push만으로 모든 대형 Tensor의 별도 백업이 있다고 판단하지 않는다.

이전 후 같은 할당량이 실제 반환되면 Available은 약1.525TiB가 된다. 다른 프로세스 쓰기는 제외한 추정이다.
다음 순서가 필요하다.

1. 백업할 외부 경로나 새 서버 용량을 확인한다.
2. 대상 Evidence를 복사한다. SHA256과 복원 가능성을 검사한다.
3. 삭제할 정확한 원본 경로에 대해 사용자 승인을 받는다.
4. 승인된 원본만 제거한다. 실제 `df` 반환량을 확인한다.

현재는 조사를 완료했다. 추가600GB를 실제 확보한 상태는 아니다.
기존 MLLM Asset·Raw·Runtime, 사용자 Checkout·Index, 실행 중인 Burner를 보존했다.
