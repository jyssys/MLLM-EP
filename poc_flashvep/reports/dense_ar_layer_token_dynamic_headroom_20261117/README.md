# Dense AR Layer/Token/Dynamic Evidence

이 Namespace는 기존 Evidence를 수정하지 않는다.
실험별 원본 Config와 Source Snapshot을 보존한다.
실패·정합성 검사·Clean 성능·Offline Model을 구분한다.

## 우선 확인할 파일

- `../dense_ar_layer_token_dynamic_headroom_final_summary.md`: 최종 판정과 제한.
- `analysis/experiment_inventory.json`: 실제 완료·실패·미실행 목록.
- `analysis/heldout_start_statistics.json`: 독립 Start 평균과 Paired95% CI.
- `analysis/heldout_per_start_metrics.csv`: Start별 지표.
- `analysis/heldout_policy_matched_actual_geometry_statistics.json`: 실제 Work·Geometry Matching.
- `analysis/heldout_normal_lifetime_credit_audit.json`: 정상 KV·ACK·Credit 복원.
- `analysis/matched_Last12_independent_start_statistics.json`: 별도 N144 같은 작업량 Control.
- `state/candidate_confirmation_freeze_O16.json`: Output16 후보 선정 계약.
- `state/candidate_confirmation_freeze_O64.json`: Output64 후보 선정 계약.
- `state/model_weights_SHA256_verification.json`: Original BF16 Weight 검사.

Start 수와 Request 수는 다르다.
N128은 독립 Start128회를 뜻하지 않는다.
N128 완료율도 Sustainable Capacity가 아니다.
물리적인 GPU Kernel Idle은 `NOT_MEASURED`다.

## Directory

| 경로 | 내용 |
|---|---|
| `configs/` | Immutable 실험 설정과 Frozen Cost/SLO Model |
| `data/` | 실제 Prompt ID·Token ID·Hash와 선택 규칙 |
| `raw/<name>/` | Config, Command, Source ZIP, Output, 상태와 Runtime 로그 |
| `logs/` | 실패를 포함한 Coordinator/Worker 실행 로그 |
| `analysis/` | 원본에서 재계산한 Gate·통계·진단 |
| `figures/` | PNG/PDF와 별도 원자료 |
| `state/` | 보존·GPU Lease·예산·후보 동결 기록 |
| `archives/` | 큰 Evidence의 손실 없는 분할 Archive와 Hash |

Raw의 `command.json`은 실제 실행 명령과 Source ZIP SHA256을 포함한다.
`evidence_input_snapshot.json`은 해당 실행의 Cost Model과 데이터 Hash를 포함한다.
이후 수정한 분석 Script가 당시 Worker Source였다고 가정하지 않는다.
실제 Worker는 해당 Run의 Source ZIP에 있다.

## Archive 복원

GitHub의 개별 파일 제한 때문에 큰 원본을48MiB 단위로 나눈다.
압축을 해제한 모든 Member의 SHA256을 원본과 비교한다.
로컬 원본은 삭제하지 않는다.
Manifest의 `verified_lossless`가 참인지 먼저 확인한다.

Repository Root에서 다음 명령을 사용한다.
명령은 별도 임시 Directory에 복원한다.
기존 Raw 파일을 덮어쓰지 않는다.

```bash
dense_ar_verify_dir="$(mktemp -d)"
cat poc_flashvep/reports/dense_ar_layer_token_dynamic_headroom_20261117/archives/large_evidence.tar.zst.part* \
  | zstd -dc --long=31 \
  | tar -xf - -C "$dense_ar_verify_dir"
```

실제 원본 경로·Byte 수·SHA256은 `archives/large_evidence_manifest.json`에 있다.
Archive를 복원하지 않으면 일부 큰 Raw 경로가 GitHub에서 직접 보이지 않는다.
`state/publication_SHA256_manifest.json`은 공개 Payload 파일의 SHA256을 기록한다.
Manifest 자신은 제외한다.
최종 Git Commit이 Manifest를 포함한다.

## 분석 재실행

필요한 Python 환경과 원본 Evidence를 먼저 준비한다.
아래 명령은 GPU 실험을 다시 시작하지 않는다.

```bash
python -m poc_flashvep.dense_ar_layer_token_pd.statistics_report
python -m poc_flashvep.dense_ar_layer_token_pd.policy_control_statistics
python -m poc_flashvep.dense_ar_layer_token_pd.lifecycle_audit
python -m poc_flashvep.dense_ar_layer_token_pd.predictor_validation
python -m poc_flashvep.dense_ar_layer_token_pd.budget_variance
python -m poc_flashvep.dense_ar_layer_token_pd.main_figures
python -m poc_flashvep.dense_ar_layer_token_pd.timeline_figures
```

GPU Replay에는 새 Namespace와 별도 Resource Lease가 필요하다.
완료 Marker를 삭제해 기존 실행을 덮어쓰지 않는다.
Worker Source, Model Revision, Prompt와 Arrival을 먼저 확인한다.
원래 Failed Receipt를 성공 결과로 교체하지 않는다.
일부 N32 Gate는 Parent Commit의 기존 독립 Reference를 사용한다.
Fresh Clone에서는 해당 이전 Archive도 복원해야 한다.
새 Prompt N128의 독립 Reference는 이번 Namespace에 보존한다.

## 공개 상태

Publication Parent는 `f0e80395aef488a7b3c6148d4155784f87291df2`다.
공개 Branch는 `flashvep/dense-ar-layer-token-dynamic-headroom-20261117`다.
새 Source와 보고서만 별도 Git Index로 Stage한다.
기존 User Checkout과 Index는 그대로 둔다.
최종 Remote SHA는 Push 후 Local Publication Receipt에 기록한다.
최종 사용자 응답에도 동일 SHA를 제공한다.
