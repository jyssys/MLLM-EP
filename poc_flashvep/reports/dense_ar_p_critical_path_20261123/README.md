# Dense AR P-critical-path PoC 재현 안내

실험별 자원은 Qwen3-8B와 H100 GPU2개입니다. LongBench400-v2와 BI0 BF16을 유지합니다.
Primary Offered QPS는2 requests/s입니다. C3 Scheduler는 Continuous Batching을 사용합니다.
B8192/S1024는 Scheduler 상한입니다. Actual Batch가 아닙니다.

Source Module은 `poc_flashvep.dense_ar_p_critical_path`입니다.
기준 Commit은 `0ae6cd973a4544c6dce13387882479b94858e0d0`입니다.
연구 Branch는 `flashvep/dense-ar-p-critical-path-20261123`입니다.
사용자 Checkout과 Staged Index는 변경하지 않습니다.

## 증거 범위

실험별 Command, Config, Source Archive, GPU UUID, Request Event와 Lifetime 검사를 보존합니다.
Configuration Inventory는 실패와 미실행도 포함합니다.
실행 완료 기록만으로 Output 정합성, Decode 보호나 Profiler 적격성을 증명하지 않습니다.

Equal-output Replay는 독립 C1에서 측정한 요청별 Output 길이를 유지합니다.
Natural EOS에서는 BI0 실행 Geometry에 따라 Output 길이가 달라질 수 있습니다.
그 차이를 보고합니다. Natural Output Work가 동일하다고 주장하지 않습니다.
성능 CI의 독립 단위는 Server Start입니다. Request나 Token을 독립 Start로 사용하지 않습니다.

Credit Retirement, Partition Geometry, D Admission과 Coordinator 개선은 별도 실험 Family입니다.
서로 다른 Family의 Start를 합치지 않습니다.
Strong Stock Token은 원래 Compiled Model 경로를 유지합니다.
Common-policy Control로 강한 기준선을 대체하지 않습니다.

Kernel Union, Host Enclosure와 Request Queue 중첩은 분모가 다릅니다.
Kernel 미관측이 GPU Physical Idle을 증명하지 않습니다. DMA와 HBM 사용량은 측정하지 않았습니다.
과거1.76초 Host 차이의 독점적인 인과 비중은 완전히 분해하지 못했습니다.
Attention-head PoC는 별도 실험입니다. 실제 입력 Capture와 유휴 D GPU를 사용합니다.
이 PoC는 Full-model 정확도, Decode 보호나 Online Speedup을 증명하지 않습니다.

## CPU 재현

새 Checkout에서는 먼저 기준 Commit의 기존 Evidence를 복원합니다.
`dense_ar_longbench400_v2_20261122/publication/minimal_evidence/manifests`에 Run별 Manifest가 있습니다.
독립 C1 Reference와 `V2_C3_Heldout192Replay_q2.0_s0/s1/s2`의 Layer·StockToken·Hybrid Raw가 필요합니다.
각 Manifest의 분할 Archive Hash를 검사합니다. 새 복원 Directory에 추출한 뒤 Member Hash도 검사합니다.
Historical 재계산의 `source.zip`은 같은 Run의 Full Evidence Manifest가 지정한 Source SHA에서 복원합니다.
공식 LongBench Metric 구현과 Dependency는 기존 Provenance의 Hash를 유지합니다.
현재 머신의 동결 Raw와 기존 사용자 Checkout 위에 압축을 해제하지 않습니다.
주요 재사용 Namespace의 Python562개는 기준 Commit과 Byte 내용이 일치합니다.
`analysis/predecessor_base_blob_audit.json`은 Git Blob별 비교를 보존합니다. 사용자 Index는 변경하지 않았습니다.

다음 Command를 `/home/esjung/MLLM-EP-github`에서 실행합니다.
기록된 Python은 `/home/esjung/.venvs/looped-pd-abc/bin/python`입니다.
설치된 vLLM, CUDA와 Model 경로는 Raw Runtime 기록에 있습니다.
Config는 이 머신의 절대 경로를 보존합니다. 다른 머신에는 동등한 Dependency와 경로가 필요합니다.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_p_critical_path.verification cpu
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_p_critical_path.reproduce analyze
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_p_critical_path.reproduce figures
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_p_critical_path.reproduce verify-figures
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_p_critical_path.verification metrics --final
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_p_critical_path.verification budget --final
```

위 Command는 Model Serving을 시작하지 않습니다. 추출한 Evidence를 읽고 Derived Analysis를 작성합니다.
Dataset 복원에는 새 Directory를 사용합니다. 동결한 데이터를 덮어쓰지 않습니다.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_p_critical_path.reproduce dataset --archive /path/to/pinned_longbench_data.zip --output /path/to/new_reconstruction_directory
```

LongBench v1 Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`입니다.
Model Revision은 `b968826d9c46dd6066d109eabc6255188de91218`입니다.
복원 Command는 Archive Hash, Tokenizer Output, Prompt Hash와 Dataset Byte Hash를 검사합니다.
원본 이용 조건에 따라 Source Archive와 Model을 확보해야 합니다.
Repository에는 Request ID와 복원 Metadata를 공개합니다. 대형 원문 Context와 Model Weight는 재배포하지 않습니다.

## Archive와 GPU 재현

Evidence Archive Manifest는 Full Local Archive와 Public Essential Archive의 분할 파일을 기록합니다.
각 Archive와 보존한 Member의 SHA256을 제공합니다.
Public 분할 파일을 기록된 순서대로 결합합니다. 결합한 SHA256을 검사한 뒤 별도 Evidence Directory에 추출합니다.
Full Local Archive는 Public Archive에서 제외한 Tensor Capture와 Worker Log도 보존합니다.
Exclusion Manifest는 해당 파일을 기록합니다. 실패한 Raw Evidence는 삭제하지 않습니다.

GPU 재실행에는 새 Campaign Namespace와 Resource Lease가 필요합니다.
완료한 Raw Folder에 기존 Controller를 다시 실행하지 않습니다.
Dispatcher는 기존 Raw Folder의 재사용을 거부합니다. Source, Dataset과 Composition Gate도 검사합니다.
재현 실험에서도 동결 SLO와 정합성 계약을 유지합니다.
이 증거만으로 Active-decode Offloading을 시작하지 않습니다. 미래 Decode-safe Window는 검증하지 못했습니다.

## Compiler와 Dispatch Protocol

같은 Source 경로만으로 같은 Compiler Cache를 보장하지 못했습니다.
고정 NIXL Port 검사에서는 Cache Root가 일치해도 최초 컴파일과 캐시 재사용의 State가 달랐습니다.
이 실패는 유지합니다. 별도의 Warm-only3회 Strict 검사는 통과했습니다.
Partial Model은 Full Backbone과 Prefix·Interval Cache를 함께 사용합니다.
CPU 판독기는 각 Cache Region의 Root와 실제 Graph 재사용을 검사합니다.
공식 Full-model Region은 AOT 재사용도 검사합니다. 새 Graph 컴파일이 있으면 해당 Warm-regime 결과는 Primary로 확정하지 않습니다.

새 구성은8→32→400 정합성·품질 Gate를 요구합니다.
Strict Control의 `controlled_pipeline=True`를 실제 C3의 `False`와 같은 설정으로 취급하지 않습니다.
Timing Primer는 실제 Continuous 구성에서 만듭니다. Priming Process를 독립 Serving Start로 세지 않습니다.
같은 Pair·Source·Model Contract의 완료된 Primer만 재사용할 수 있습니다.
실제 Timing Log의 Cache Region 검사도 생략하지 않습니다.

Warm-regime Confirmation은 Equal-output N3를 먼저 실행합니다.
Natural N3는 관측한 실행시간과 남은 Dispatch Budget으로 전체 Stage를 완료할 수 있을 때만 시작합니다.
이 결정에는 Layer·Token의 승패를 사용하지 않습니다. 미실행 Natural Cell은 NOT_RUN으로 유지합니다.
최대10시간의 종료 시각은 변경하지 않았습니다. Dispatch 종료 후75분을 Closeout에 배정했습니다.

## 최종 검사

Science Audit는 완료한 작업과 미해결 계측 요구사항을 함께 기록합니다.
부정적인 연구 결과를 성능 향상으로 바꾸어 표현하지 않습니다.
Resource Audit는 연구 GPU Process가 없음을 검사합니다. 승인된 Burner의 전체8GPU 사용률도 두 번 확인합니다.
Publication 기록은 Push한 연구 Commit과 별도로 확인한 Remote SHA를 포함합니다.
