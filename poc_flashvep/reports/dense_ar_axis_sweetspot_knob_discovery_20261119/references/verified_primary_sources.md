# 선행연구와 계측 범위

이번 캠페인은 새 Serving Framework를 재현하지 않는다. 아래 원문은 가설과 비교 범위를 정하는 데 사용했다.

## 실행 축과 Admission

[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)는 Chunked Prefill과 Decode의 실행을 함께 조정한다. 이 기능은 실제 P→D Partial State Migration과 다르다. Co-batching 자체를 새로운 기여로 주장하지 않는다.

[Layered Prefill](https://arxiv.org/html/2510.08055v2)은 연속 Layer Group을 실행 단위로 사용한다. 주된 효율 근거에는 MoE Expert Weight 재사용이 포함된다. §5.13의 Dense Qwen3-8B 결과는 Chunked Prefill보다 느렸다. 이 결과를 현재 2GPU Remote Continuation의 불가능성으로 해석하지 않는다. 반대로 MoE의 효율을 Dense 모델에 그대로 적용하지 않는다.

[FluidPD](https://arxiv.org/html/2610.06917v1)의 §IV-B는 완료한 Token Prefix의 KV를 전달한다. D는 그 KV를 앞 문맥으로 사용해 Suffix를 실행한다. 이 계약은 현재 Token Baseline의 Causal Dependency와 대응한다. 그러나 원문 §IV-B3은 승인한 Suffix를 다음 유효 Iteration에 실행한다고 설명한다. 현재 Runtime의 Whole-cohort Drain Guard와는 다르다. 따라서 원문의 Queue 모델과 성능을 현재 구현의 근거로 재사용하지 않는다.

[Kairos](https://arxiv.org/html/2607.02043v1)는 전체 Prefill 요청을 D로 우회한다. 원문은 Batch, Context와 Chunk 진행 상태에 따른 Step Cost 변화를 다룬다. §3.4의 Oracle은 Simulator 상태를 사용한다. 이를 배포 가능한 Online Scheduler나 현재 Runtime의 실측 이득으로 표현하지 않는다.

위 구분은 실행 계약의 비교다. 새로운 논문 기여나 선행연구의 부재를 입증하지 않는다.

## GPU 계측

[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)는 CUDA API, NVTX와 GPU Activity를 구분한다. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)는 Replay, Cache와 Clock 설정이 측정에 영향을 줄 수 있음을 설명한다. [CUPTI 문서](https://docs.nvidia.com/cupti/overview/overview.html)는 GPU Activity 계측 인터페이스를 설명한다.

현재 분석은 Host Timestamp와 CUDA Event Enclosure를 Exclusive Kernel Time으로 취급하지 않는다. Online Observer는 동일 Seed의 A/B/A로 검증한다. 사전 기준은 각 Metric의 5% 이내 변화다. 기준을 넘는 Observer 결과는 `DIAGNOSTIC`으로 남긴다. 고립 Operator의 Observer Gate는 Online Serving의 Gate를 대체하지 않는다.

## 기존 기록의 우선순위

Primary Evidence는 Git Commit `4cf3f9b80a2a8618da3bdddb1140fab7f403c3c7`의 원시 기록이다.

사용자가 제공한 [실험 아카이브](https://app.notion.com/p/1062a9a81d8641f895f0bdf6c0702489)와 [Dense AR 체크리스트](https://app.notion.com/p/4170ee8277e481e9848ac7283309f7df)를 읽기 전용으로 확인했다. 체크리스트는 `UNVERIFIED` 상태였다. 마지막 편집 시각은 `2026-11-16T16:24:44.170Z`였다. 이 오래된 상태를 새 GPU 측정의 판정으로 사용하지 않는다. Notion 문서는 수정하지 않았다.
