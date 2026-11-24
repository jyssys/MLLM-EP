# Native-relative TTFT·Joint SLO 원인 조사

## 먼저 읽을 자료

[최종 요약](../dense_ar_ttft_slo_regression_final_summary.md)이 과학적 결론의 기준이다.
`analysis/scientific_verdict.json`은 판정의 구조화된 사본이다.
`state/research_checkpoint.json`은 진행 중 Snapshot이다. 최종 완료 기록으로 사용하지 않는다.
최종 Science·Resource·Publication 기록은 각각의 Audit 파일을 사용한다.

## 증거 Family

| Family | 범위 | 해석 |
|---|---|---|
| Historical Main400 | 원래 Natural15개 주요 실행, 총209개 과거 Timing 감사 | 원시 지표 재현이다. 새 GPU 반복이 아니다. |
| Current Main400 | Natural13개 Arm×독립 Start3회 | 새 Warm Runtime이다. 기존 v2와 합쳐 평균내지 않는다. |
| Current Heldout192 | Equal-output10개 Arm×독립 Start3회 | 실행당11208개 출력 Token이다. Main과 다른 요청·Seed다. |
| Serial32 | Batch1/1/1·Output16·prospective CPC | Native·Layer·Hybrid N3만 적격이다. Stock은 간섭 실패다. |
| Frozen-path·Gate Oracle | 관측한 Metric 경계의 가상 제거 | 이후 Queue·KV·ITL을 갱신하지 않는다. Online Speedup이 아니다. |

## 코드와 재계산

코드는 `poc_flashvep/dense_ar_native_ttft_slo/`에 있다.
기준 Source는 `0e2e062260103feb53bd0afbb0ac964ebb027cd9`다.
실행 CWD는 `/home/esjung/MLLM-EP-github`다.
Python은 `/home/esjung/.venvs/looped-pd-abc/bin/python`이다.

원래 환경에서 다음 CPU 명령으로 증거를 재계산한다.
각 명령은 기존 Raw를 읽는다. GPU 실험을 시작하지 않는다.

```bash
python -B -m poc_flashvep.dense_ar_native_ttft_slo.failure_analysis
python -B -m poc_flashvep.dense_ar_native_ttft_slo.request_trace
python -B -m poc_flashvep.dense_ar_native_ttft_slo.policy_trace
python -B -m poc_flashvep.dense_ar_native_ttft_slo.causal_analysis
python -B -m poc_flashvep.dense_ar_native_ttft_slo.intervention_analysis
python -B -m poc_flashvep.dense_ar_native_ttft_slo.profile_analysis
python -B -m poc_flashvep.dense_ar_native_ttft_slo.verification figures
python -B -m poc_flashvep.dense_ar_native_ttft_slo.verification cpu
```

실제 측정 Runtime은 vLLM0.26.0+cu129·PyTorch2.11.0+cu129·NIXL1.4.1이다.
정확한 Source ZIP·Config·Model Revision·GPU UUID는 실행별 기록에 있다.
다른 머신에서는 절대 경로와 보존 Snapshot을 먼저 감사해야 한다.
원래 User Checkout·Index의 보존 검사 실패를 자동으로 우회하지 않는다.
원본 Prompt는 고정한 공식 Dataset Revision과 공개 Identity Roster로 복원한다.
실측을 다시 실행하려면 별도의 GPU 소유권·자원 검사가 필요하다.

## Table·Figure

`figures/metrics_per_start.csv`는 실행별 Metric이다.
`figures/metrics_start_mean_and_pooled.csv`는 Start-mean과 Pooled 통계를 분리한다.
`figures/slo_eight_categories.csv`는 여덟 Gate Mask의 Count·Token Weight·Rate다.
`figures/controlled_intervention_contrasts.csv`는 같은 Start의 정책 대비와 CI다.
`figures/controlled_intervention_decode_gates.csv`는 두 Native 기준과 Absolute Gate다.
`analysis/figure_manifest.json`은 역사적 Figure10개를 설명한다.
`analysis/intervention_figure_manifest.json`은 추가 Figure7개를 설명한다.
모든 Figure는 대응 PNG·PDF·CSV가 있다.
`analysis/figure_regeneration_audit.json`은51개 파일의 두 번 생성 Byte 일치를 검사한다.

## Raw와 무손실 압축

`archives/evidence_archive_manifest.json`은 Raw Archive의 Member별 SHA256을 보존한다.
공개 Part를 Manifest 순서로 연결하면 원래 Public Archive가 된다.
연결한 SHA256을 검사한 뒤 Zstandard로 해제한다.
압축 해제 후에도 Member별 Byte 수와 SHA256을 검사한다.
Full Worker Log·Prompt·Reference·Tensor는 로컬 Full Archive에 보존한다.
모델 Weight와 원문 Dataset Context는 공개 Commit에 넣지 않는다.

80MiB 이상 파생 CSV는 공개용 `.csv.gz` 사본을 제공한다.
`archives/large_derived_csv_gzip_manifest.json`에 원본·압축 SHA256이 있다.
압축을 풀면 같은 CSV가 나온다. 로컬 원본 CSV도 삭제하지 않는다.
실패한 Capture와 `NOT_RUN` 계획도 보존한다.

## 최종 Audit

`state/science_completion_audit.json`은 증거 범위와 종결 상태를 검사한다.
이 Audit의 PASS는 새로운 Serving Speedup 성공을 뜻하지 않는다.
`state/resource_closeout.json`은 승인된 GPU0–7 Burner의 소유권과 실제 실행을 검사한다.
`state/publication_commit_science.json`은 로컬 Commit과 Remote SHA를 검사한다.
추가 Closeout Commit이 있으면 별도 Publication 기록을 사용한다.
