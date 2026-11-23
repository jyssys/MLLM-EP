# Attention-head Offloading 소규모 PoC

상태: 실측 완료 여부와 정합성 Gate는 아래 표를 따릅니다. 기존 P-critical-path 캠페인과 별도입니다.

Qwen3-8B, Original BF16, BI0를 사용합니다. 원문 Prompt와 LongBench400-v2는 변경하지 않습니다.
입력 길이만으로8K·16K·32K에 가장 가까운 원본 Request를 사전 선택합니다.
길이별 원본 Prompt는1개입니다. 길이 효과를 Task·Prompt 차이와 분리하지 않았습니다.
Original Full36 vLLM의 Eager Capture에서 실제 Q/K/V와 Paged KV, RoPE 이후 Query를 저장합니다.
Capture는 B8192/S1024와 KV42GiB를 사용합니다. CPU Tensor 복사는 침습적이므로 Capture 시간은 성능 결과가 아닙니다.

재생 실험은 LocalFull, LocalSplit과 원격 실행 경로를 비교합니다.
RemoteSplit은 Chunk마다 전체 Prefix KV를 전송합니다.
RemoteCachedSplit은 Layer별 원격 KV를 보존합니다. 새 Page와 갱신한 경계 Page만 전송합니다.
두 원격 경로는 같은 LocalFull 기준과 비교합니다. KV 재사용이 없는 경로만으로 전체 방법을 판정하지 않습니다.
KV Head2/8과4/8을 옮깁니다. Query Head는 GQA 비율4:1을 유지합니다.
모든 Head를 계산합니다. Head를 버리는 근사 기법이 아닙니다.
1·2·4 Layer는 선택한 Layer들의 Attention 호출 열을 뜻합니다. 전체 Transformer Layer 실행시간이 아닙니다.
QKV Projection, FFN, 다음 Layer Hidden의 준비 비용은 포함하지 않습니다.
왕복 시간에는 Packing, Torch Peer Copy, 두 Device의 Launch·Wait와 결과 병합이 포함됩니다.
Tensor·Event·Scheduler Metadata의 초기 준비는 Timing 전에 수행합니다. 동적 Metadata 생성 비용은 이 수치에 포함하지 않습니다.
실제 NIXL/NCCL 경로와 별도입니다. 이 시간으로 Production 통신 비용을 확정하지 않습니다.
재생 중 전체 Model Weight와42GiB Serving KV Cache는 상주하지 않습니다.
D는 이 실험 외에는 계산하지 않습니다. Native Decode 보호와 실제 Safe Window는 NOT_RUN입니다.

| Input Tokens | Layers | 원격 경로 | 원격 KV Heads | 독립 Process | LocalFull ms | Remote ms | 논리 Payload MiB | Remote/LocalFull p50 [95% CI] | Strict Gates |
|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 8181 | 1 | RemoteSplit | 2/8 | 3 | 0.8447 | 0.9652 | 39.957 | 1.142671 [1.1376018563694505, 1.1491842914840829] | {'full_vs_captured': True, 'split_vs_full': True, 'remote_vs_local_split': True} |
| 8181 | 1 | RemoteCachedSplit | 2/8 | 3 | 0.8447 | 0.9634 | 39.957 | 1.140608 [1.13269841989626, 1.1466463859984315] | {'full_vs_captured': True, 'split_vs_full': True, 'cached_vs_local_split': True} |
| 8181 | 1 | RemoteSplit | 4/8 | 3 | 0.8346 | 0.8930 | 79.914 | 1.070004 [1.0667634207134684, 1.07335735610669] | {'full_vs_captured': True, 'split_vs_full': True, 'remote_vs_local_split': True} |
| 8181 | 1 | RemoteCachedSplit | 4/8 | 3 | 0.8346 | 0.8708 | 79.914 | 1.043393 [1.0423534827596517, 1.0446607553151899] | {'full_vs_captured': True, 'split_vs_full': True, 'cached_vs_local_split': True} |
| 8181 | 2 | RemoteSplit | 2/8 | 3 | 1.7384 | 1.9148 | 79.914 | 1.101429 [1.0747119758376908, 1.1388276810238112] | {'full_vs_captured': True, 'split_vs_full': True, 'remote_vs_local_split': True} |
| 8181 | 2 | RemoteCachedSplit | 2/8 | 3 | 1.7384 | 1.8848 | 79.914 | 1.084204 [1.0560332931635739, 1.123050897574488] | {'full_vs_captured': True, 'split_vs_full': True, 'cached_vs_local_split': True} |
| 8181 | 2 | RemoteSplit | 4/8 | 3 | 1.6853 | 1.7411 | 159.828 | 1.033116 [0.9997658798997462, 1.0605776554578794] | {'full_vs_captured': True, 'split_vs_full': True, 'remote_vs_local_split': True} |
| 8181 | 2 | RemoteCachedSplit | 4/8 | 3 | 1.6853 | 1.7219 | 159.828 | 1.021748 [0.9907775173739654, 1.0464980957530918] | {'full_vs_captured': True, 'split_vs_full': True, 'cached_vs_local_split': True} |
| 8181 | 4 | RemoteSplit | 2/8 | 3 | 3.5930 | 3.7532 | 159.828 | 1.044593 [1.0268327157129165, 1.0696261038142212] | {'full_vs_captured': True, 'split_vs_full': True, 'remote_vs_local_split': True} |
| 8181 | 4 | RemoteCachedSplit | 2/8 | 3 | 3.5930 | 3.9310 | 159.828 | 1.094070 [1.0659615120235122, 1.1255271948560346] | {'full_vs_captured': True, 'split_vs_full': True, 'cached_vs_local_split': True} |
| 8181 | 4 | RemoteSplit | 4/8 | 3 | 3.6067 | 3.4257 | 319.656 | 0.949836 [0.9358554537518983, 0.9649925235987251] | {'full_vs_captured': True, 'split_vs_full': True, 'remote_vs_local_split': True} |
| 8181 | 4 | RemoteCachedSplit | 4/8 | 3 | 3.6067 | 3.4131 | 319.656 | 0.946338 [0.9328584029162422, 0.9607767843759925] | {'full_vs_captured': True, 'split_vs_full': True, 'cached_vs_local_split': True} |
| 16477 | 1 | RemoteSplit | 2/8 | 3 | 3.3818 | 3.3069 | 104.488 | 0.977836 [0.9503802519054493, 1.004123444144335] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 16477 | 1 | RemoteCachedSplit | 2/8 | 3 | 3.3818 | 3.2063 | 80.488 | 0.948109 [0.9234510862826669, 0.9749982787995098] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 16477 | 1 | RemoteSplit | 4/8 | 3 | 3.3408 | 2.8010 | 208.977 | 0.838415 [0.8006541734284575, 0.8798007253555374] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 16477 | 1 | RemoteCachedSplit | 4/8 | 3 | 3.3408 | 2.5835 | 160.977 | 0.773314 [0.7371247031340449, 0.8137087780434029] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 16477 | 2 | RemoteSplit | 2/8 | 3 | 6.9163 | 6.6067 | 208.977 | 0.955234 [0.9486382584959532, 0.9623297632811133] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 16477 | 2 | RemoteCachedSplit | 2/8 | 3 | 6.9163 | 6.4032 | 160.977 | 0.925823 [0.918039022464435, 0.9350035809219029] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 16477 | 2 | RemoteSplit | 4/8 | 3 | 6.9252 | 5.5060 | 417.953 | 0.795065 [0.7903247325016506, 0.800459110203118] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 16477 | 2 | RemoteCachedSplit | 4/8 | 3 | 6.9252 | 5.1385 | 321.953 | 0.742003 [0.7389263568003764, 0.7457527499513514] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 16477 | 4 | RemoteSplit | 2/8 | 3 | 14.0490 | 13.1051 | 417.953 | 0.932813 [0.9207196460424275, 0.9420232959183188] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 16477 | 4 | RemoteCachedSplit | 2/8 | 3 | 14.0490 | 12.9955 | 321.953 | 0.925009 [0.9109177771567316, 0.9375135184339618] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 16477 | 4 | RemoteSplit | 4/8 | 3 | 13.6072 | 11.0203 | 835.906 | 0.809884 [0.8052048773388639, 0.8161209807900777] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 16477 | 4 | RemoteCachedSplit | 4/8 | 3 | 13.6072 | 10.3069 | 643.906 | 0.757459 [0.7537298185574589, 0.7618847911388826] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 32971 | 1 | RemoteSplit | 2/8 | 3 | 13.1869 | 11.6060 | 241.043 | 0.880121 [0.8786627415426865, 0.881895909761599] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 32971 | 1 | RemoteCachedSplit | 2/8 | 3 | 13.1869 | 11.3197 | 161.043 | 0.858411 [0.8561901617937434, 0.86013667457896] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 32971 | 1 | RemoteSplit | 4/8 | 3 | 13.1896 | 9.0084 | 482.086 | 0.682994 [0.6814080952813626, 0.6847265540185367] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 32971 | 1 | RemoteCachedSplit | 4/8 | 3 | 13.1896 | 8.4183 | 322.086 | 0.638248 [0.6356581994500176, 0.6409228059946449] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 32971 | 2 | RemoteSplit | 2/8 | 3 | 27.1634 | 23.5589 | 482.086 | 0.867303 [0.8637910302738591, 0.8715394969793574] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 32971 | 2 | RemoteCachedSplit | 2/8 | 3 | 27.1634 | 23.2642 | 322.086 | 0.856453 [0.8459821440344326, 0.8683265360123131] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 32971 | 2 | RemoteSplit | 4/8 | 3 | 27.3912 | 18.0549 | 964.172 | 0.659148 [0.6542287591626682, 0.6647624515648516] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 32971 | 2 | RemoteCachedSplit | 4/8 | 3 | 27.3912 | 17.1304 | 644.172 | 0.625399 [0.6217185284186806, 0.6285470877598609] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 32971 | 4 | RemoteSplit | 2/8 | 3 | 55.1179 | 47.0913 | 964.172 | 0.854372 [0.8511577487178609, 0.8571978700845544] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 32971 | 4 | RemoteCachedSplit | 2/8 | 3 | 55.1179 | 46.1425 | 644.172 | 0.837159 [0.8363887709080932, 0.8381154114323516] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |
| 32971 | 4 | RemoteSplit | 4/8 | 3 | 54.9687 | 36.1509 | 1928.344 | 0.657663 [0.6563974447975366, 0.6587721417077698] | {'full_vs_captured': True, 'split_vs_full': False, 'remote_vs_local_split': True} |
| 32971 | 4 | RemoteCachedSplit | 4/8 | 3 | 54.9687 | 34.3229 | 1288.344 | 0.624409 [0.6225608632278372, 0.6263059354701807] | {'full_vs_captured': True, 'split_vs_full': False, 'cached_vs_local_split': True} |

CI는 독립 Benchmark Process를 사용합니다. 독립 Serving Start CI가 아닙니다.
동일 Capture를3개 Process가 재생합니다. 새 원본 Prompt3×3개를 생성한 실험은 아닙니다.
ms 열은 Process별 p50의 평균입니다. 모든 Iteration을 합친 p50이 아닙니다.
논리 Payload는 요청한 Tensor Bytes입니다. 실제 DMA 또는 Wire Bytes는 측정하지 않았습니다.
Remote와 LocalSplit의 Exact 비교는 같은 Head 형상의 최종 Attention 결과를 검사합니다.
이 비교만으로 Copy와 원격 Kernel의 개별 정확도를 각각 분해하지는 않습니다.
LocalSplit과 LocalFull의 차이는 Head Geometry 변경을 검사합니다.
전체 모델 Greedy Output·Task 품질·Online TTFT·SLO-goodput은 NOT_RUN입니다.
Isolated 비용 이득만으로 Layer-specific Serving Gain이나 Dynamic Scheduler 성공을 주장하지 않습니다.

## 정합성 결과의 구분

| 비교 | 관측 수 | Exact 불일치 수 | 최대 절대 오차 |
|---|---:|---:|---:|
| full_vs_captured | 216 | 0 | 0.0 |
| split_vs_full | 216 | 48 | 0.001953125 |
| remote_vs_local_split | 216 | 0 | 0.0 |
| cached_vs_local_split | 216 | 0 | 0.0 |

원격 결과는 같은 Head 형상의 LocalSplit과 비교합니다.
Head를 분할한 결과와 전체 Head Kernel의 결과는 별도로 비교합니다.
불일치한 Cell은 Strict 통과 비용 이득으로 확정하지 않습니다.
이 차이를 전체 모델 품질 저하 또는 품질 통과로 바꾸어 해석하지 않습니다.

| 실제 Query Tokens | 관측 수 | Split/Full 불일치 | Remote/LocalSplit 불일치 | Cached/LocalSplit 불일치 |
|---:|---:|---:|---:|---:|
| 93 | 24 | 24 | 0 | 0 |
| 203 | 24 | 24 | 0 | 0 |
| 8181 | 24 | 0 | 0 | 0 |
| 8192 | 144 | 0 | 0 | 0 |

이 표는 Query Shape별 관측입니다. 실제 Split-K 선택이나 Reduction Tree는 계측하지 않았습니다.
216개 관측은36개 Captured Attention 호출×2개 Head 분할×3개 재생 Process입니다.
이를216개 독립 Request로 취급하지 않습니다.
Kernel의 내부 선택을 불일치의 확정 원인으로 표현하지 않습니다.

## Protocol 이력

| Version | 상태 | Source Archive SHA256 |
|---:|---|---|
| 1 | STOPPED_PENDING_CONTROLLER_REVISION | `80b405c1554e005cdf711de0ba701667927ad42f64b941128b07667c35a957ee` |
| 2 | FAILED | `b981c5960c4e129409acf8fe0b7aa0754396c2ca8662d500352872e1b8beae27` |
| 3 | NOT_RUN | `a11b61cab3eed2e927f291001bac1fa7d3c356a1d47b802dd013f1376917fa91` |
| 4 | COMPLETE | `a11b61cab3eed2e927f291001bac1fa7d3c356a1d47b802dd013f1376917fa91` |

활성 Protocol: `attention_poc_v4.json`. 이전 Raw와 실패는 보존합니다.
이전 Protocol의 Process를 새 Protocol과 합쳐 독립 N을 늘리지 않습니다.

[attention_head_poc_analysis.json](dense_ar_p_critical_path_20261123/attention_poc/analysis/attention_head_poc_analysis.json)

관련 선행연구: [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509), [AFD 설계 공간 분석](https://arxiv.org/abs/2605.28302).
학습용 Sequence Parallelism과 대규모 MoE의 AFD 결과를 이 Dense8B PoC의 성능 근거로 사용하지 않습니다.
