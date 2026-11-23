# Compiler, Admission 및 정합성 원인 분석

상태: FINAL.
생성 시각: 2026-11-23T02:03:52.360311+00:00.
경과 연구 시간: 18.951시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 수정 범위

수정은 새 Source Capsule에만 적용했다. 기존 v1 Runtime과 Raw Evidence는 유지했다.
P Partial Compiler는 고유 Compilation Config와 Cache를 사용한다.
D Partial Compiler는 고정 Layer Interval을 별도 Graph로 실행한다.
원래 Weight와 KV를 사용한다. Full36 Decode의 원래 Compiled 경로는 유지한다.
Prompt의 CUDA Primitive Dispatch는 별도 Context로 제한한다.
Assertion을 제거하거나 Output Tensor를 덮어쓰지 않았다.
N8의 Strict State와 Natural32/400의 Coverage, Lifetime 및 품질 Gate를 모두 요구한다.
계측하지 않은 Operator-exclusive 시간 절감은 주장하지 않는다.

## Fused Hybrid Teacher의 Dispatch 오류

처음 두 Fused 비교는 Strict State에서 실패했다. 이 실패를 그대로 보존했다.
독립 Teacher의 필수 `teacher_eager_prefix_compiled_suffix` Flag가 빠진 것을 확인했다.
새 Version은 이 Teacher Dispatch Flag만 복구했다. Candidate 모델 실행은 바꾸지 않았다.
새 Teacher의 실제 실행 경로 Log를 요구했다. 기존 실패 Gate를 성공으로 변경하지 않았다.
ContextFixV15의 새 N8/N32/N400 결과는 다음 표에 별도로 표시한다.

| 정책 후보 | 전체 Gate PASS | Strict8 PASS | Natural Stage 수 | Gate 파일 |
| --- | --- | --- | --- | --- |
| Layer_Cap0_Credit8 | True | True | 2 | c3_policy_Layer_Cap0_Credit8_v2.json |
| Layer_Cap1_Credit8 | True | True | 2 | c3_policy_Layer_Cap1_Credit8_v2.json |
| Layer_Cap4_Credit8 | True | True | 2 | c3_policy_Layer_Cap4_Credit8_v2.json |
| Layer_Cap2_Credit16 | True | True | 2 | c3_policy_Layer_Cap2_Credit16_v2.json |
| Layer_EagerPrefix_Cap2_Credit8 | True | True | 2 | c3_policy_Layer_EagerPrefix_Cap2_Credit8_v2.json |
| Layer_CompiledPrefix_Cap2_Credit8 | True | True | 2 | c3_policy_Layer_CompiledPrefix_Cap2_Credit8_v2.json |
| Layer_DEagerInterval_Cap2_Credit8 | True | True | 2 | c3_policy_Layer_DEagerInterval_Cap2_Credit8_v2.json |
| Layer_DCompiledInterval_Cap2_Credit8 | True | True | 2 | c3_policy_Layer_DCompiledInterval_Cap2_Credit8_v2.json |
| StockToken_Cap0_Credit0 | True | True | 2 | c3_policy_StockToken_Cap0_Credit0_v2.json |
| StockToken_Cap1_Credit0 | True | True | 2 | c3_policy_StockToken_Cap1_Credit0_v2.json |
| StockToken_Cap4_Credit0 | True | True | 2 | c3_policy_StockToken_Cap4_Credit0_v2.json |
| StockToken_Cap2_Credit8 | True | True | 2 | c3_policy_StockToken_Cap2_Credit8_v2.json |
| StockToken_Cap2_Credit16 | False | 독립 Native/State Gate | 0 | c3_policy_StockToken_Cap2_Credit16_v2.json |
| Hybrid_Cap0_Credit8 | True | True | 2 | c3_policy_Hybrid_Cap0_Credit8_v2.json |
| Hybrid_Cap1_Credit8 | True | True | 2 | c3_policy_Hybrid_Cap1_Credit8_v2.json |
| Hybrid_Cap4_Credit8 | True | True | 2 | c3_policy_Hybrid_Cap4_Credit8_v2.json |
| Hybrid_Cap2_Credit16 | True | True | 2 | c3_policy_Hybrid_Cap2_Credit16_v2.json |
| Hybrid_EagerPrefix_Cap2_Credit8 | True | True | 2 | c3_policy_Hybrid_EagerPrefix_Cap2_Credit8_v2.json |
| Hybrid_CompiledPrefix_Cap2_Credit8 | True | True | 2 | c3_policy_Hybrid_CompiledPrefix_Cap2_Credit8_v2.json |
| Hybrid_DEagerInterval_Cap2_Credit8 | True | True | 2 | c3_policy_Hybrid_DEagerInterval_Cap2_Credit8_v2.json |
| Hybrid_DCompiledInterval_Cap2_Credit8 | True | True | 2 | c3_policy_Hybrid_DCompiledInterval_Cap2_Credit8_v2.json |
| Hybrid_FusedShort_DCompiledInterval_Cap2_Credit8 | False | 독립 Native/State Gate | 0 | c3_policy_Hybrid_FusedShort_DCompiledInterval_Cap2_Credit8_v2.json |
| Hybrid_FusedShort_Cap2_Credit8 | False | 독립 Native/State Gate | 0 | c3_policy_Hybrid_FusedShort_Cap2_Credit8_v2.json |
| Deflection_Cap0_Credit8 | True | 독립 Native/State Gate | 2 | c3_policy_Deflection_Cap0_Credit8_v2.json |
| Deflection_Cap1_Credit8 | True | 독립 Native/State Gate | 2 | c3_policy_Deflection_Cap1_Credit8_v2.json |
| Deflection_Cap4_Credit8 | True | 독립 Native/State Gate | 2 | c3_policy_Deflection_Cap4_Credit8_v2.json |
| Deflection_Cap2_Credit16 | True | 독립 Native/State Gate | 2 | c3_policy_Deflection_Cap2_Credit16_v2.json |
| Deflection_Cap2_Credit32 | True | 독립 Native/State Gate | 2 | c3_policy_Deflection_Cap2_Credit32_v2.json |
| NativeCommon_Cap0_Credit0 | True | 독립 Native/State Gate | 2 | c3_policy_NativeCommon_Cap0_Credit0_v2.json |
| NativeCommon_Cap1_Credit0 | True | 독립 Native/State Gate | 2 | c3_policy_NativeCommon_Cap1_Credit0_v2.json |
| NativeCommon_Cap4_Credit0 | True | 독립 Native/State Gate | 2 | c3_policy_NativeCommon_Cap4_Credit0_v2.json |
| Hybrid_FusedShort_Cap2_Credit8_ContextFixV15 | True | True | 2 | c3_policy_Hybrid_FusedShort_Cap2_Credit8_context_fix_v15.json |
| Hybrid_FusedShort_DCompiledInterval_Cap2_Credit8_ContextFixV15 | True | True | 2 | c3_policy_Hybrid_FusedShort_DCompiledInterval_Cap2_Credit8_context_fix_v15.json |

## Stock Token Strict 실패의 반증 검사

실패한 Credit16 Probe의 Serial Strict 설정은 Coordinator Credit0을 사용했다.
따라서 이 실패를 Credit16의 인과 효과라고 결론 내리지 않는다.
P Prefix KV와 D Layer0 KV는 일치했다. 최초 D State 불일치는 Layer1에서 관측됐다.
FX Graph는 같았다. 일부 Compiler Autotune Reduction 구성은 달랐다.
R0_BLOCK2048/4096 차이는 관측 근거다. 이것만으로 불일치 원인을 확정하지 않는다.
같은 Pair에서 Control → Pre-forward Fence → Control을 한 번씩 실행했다.
No-fence Control도 통과하면 Fence가 오류를 해결했다는 근거가 아니다.
기존 실패 Arm은 Primary 성능에서 제외한다. 통과할 때까지 반복하는 절차를 사용하지 않았다.

| 고정 실행 | 결과 | Fence 활성화 | 실제 Fence 수 |
| --- | --- | --- | --- |
| V2_Stock_PreFence_Control1_N8_V14 | PASS | False | 0 |
| V2_Stock_PreFence_PreFence_N8_V14 | PASS | True | 560 |
| V2_Stock_PreFence_Control2_N8_V14 | PASS | False | 0 |

[stock_policy_strict_failure_audit_v13.json](dense_ar_longbench400_v2_20261122/analysis/stock_policy_strict_failure_audit_v13.json)

[stock_compiler_nested_artifact_audit_v13.json](dense_ar_longbench400_v2_20261122/analysis/stock_compiler_nested_artifact_audit_v13.json)

[stock_pre_forward_fence_control_v14.json](dense_ar_longbench400_v2_20261122/analysis/stock_pre_forward_fence_control_v14.json)

## Compiler/Credit/Admission 독립 확인

다음 실험은 같은 192개 원문과 독립 C1 Output Length를 사용한 Fixed-length Replay다.
P Compiler On/Off는 같은 Source와 D 경로를 사용한다.
D Compiler On/Off도 같은 Source와 나머지 Flag를 유지한다.
Credit8/16은 Source Outstanding Window를 바꾼다. 실제 GPU Batch를 직접 고정하지 않는다.
Cap1/4는 같은 Admission Policy의 상한을 바꾼다. Queue와 Batch 변화는 그 총 효과의 일부다.
결과는 Controlled Replay다. Natural EOS Serving 이득으로 표현하지 않는다.

| 변경/대조 | Paired Starts | TTFT p95 비율 [CI] | ITL p99 비율 [CI] | TPOT p95 비율 [CI] | SLO-goodput 비율 [CI] | 변경 Arm Decode PASS | 대조 Arm Decode PASS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCompiled/PEager | 3 | 1.0001 [0.9769, 1.0337] | 1.0012 [0.9964, 1.0064] | 1.0059 [1.0050, 1.0071] | 0.9958 [0.9713, 1.0162] | True | True |
| Credit16/PCompiled | 3 | 1.0072 [0.9912, 1.0234] | 0.9970 [0.9947, 0.9994] | 1.0048 [1.0031, 1.0065] | 0.9619 [0.9139, 1.0022] | True | True |
| Cap4/Cap1 | 3 | 0.8957 [0.8924, 0.8995] | 1.0227 [1.0195, 1.0261] | 1.0078 [1.0041, 1.0124] | 1.0639 [1.0194, 1.1088] | True | True |
| DCompiled/DEager | 3 | 1.1577 [1.1427, 1.1772] | 1.0003 [0.9975, 1.0031] | 1.0019 [1.0015, 1.0026] | 0.1180 [0.0056, 0.2579] | True | True |
| DFused/DCompiled | 3 | 0.8060 [0.7889, 0.8245] | 0.9922 [0.9849, 0.9994] | 0.9925 [0.9909, 0.9939] | 8.8882 [4.1891, 185.5948] | True | True |

모든 Control은 QPS4.8의 과부하 Fixed-length Replay다. Sustainable Capacity 결과가 아니다.
다음 표는 Goodput 비율의 실제 분모를 함께 보존한다.

| Arm | TTFT p95 Start-mean ms | SLO-goodput Start-mean Tokens/s |
| --- | --- | --- |
| Cap1 | 59417.429 | 7.355 |
| Cap4 | 53217.367 | 7.825 |
| Credit16 | 55259.443 | 7.635 |
| DCompiled | 63697.823 | 0.936 |
| DEager | 55020.293 | 7.931 |
| DFused | 51337.640 | 8.318 |
| NativeCommon1 | 49202.712 | 8.819 |
| NativeCommon2 | 37560.391 | 9.598 |
| NativeCommon4 | 35516.194 | 10.447 |
| NativeStrong | 33294.292 | 9.021 |
| PCompiled | 54866.829 | 7.938 |
| PEager | 54862.666 | 7.971 |

별도의 사후 Compound 비교는 D Compiler와 Short Fusion을 함께 바꾼다.
이는 기존의 같은 N3를 재사용한다. 새 독립 실행 3회를 추가한 것이 아니다.
Compound TTFT p95 비율: 0.9331 [0.9209, 0.9449].
Compound SLO-goodput 비율: 1.0488 [1.0410, 1.0571].

[c3_knob_TTFT_ratio_controls_v2.png](dense_ar_longbench400_v2_20261122/figures/c3_knob_TTFT_ratio_controls_v2.png)

[c3_knob_absolute_TTFT_goodput_v2.png](dense_ar_longbench400_v2_20261122/figures/c3_knob_absolute_TTFT_goodput_v2.png)

[c3_knob_additive_host_path_v2.png](dense_ar_longbench400_v2_20261122/figures/c3_knob_additive_host_path_v2.png)

## Decode Tail의 시간 경계

Client ITL은 Worker Emission Gap과 Delivery Delay 변화로 분해한다.
같은 Notification에서 여러 Token을 전달하면 Co-delivery Interval0을 유지한다.
Host NIXL 중첩과 Request 등록은 Timeline 위치다. 이를 GPU 지연의 인과적 비중으로 계산하지 않는다.
NativeStrong과 Common-policy Native의 차이는 정책 Bundle의 총 효과다.
이 비교만으로 Cap, Remote READ 또는 Registration 하나의 효과를 확정하지 않는다.
## Held-out N3의 P/D 경로

다음 차이는 같은 Request와 Arrival을 사용한 Start별 Mean Host 구간이다.
두 Trace 모두 QPS2.0이다. Natural EOS에서는 실제 Output 작업량이 다를 수 있다.
Fixed-length Replay는 요청별 독립 C1 길이를 그대로 사용한다. 두 Trace를 합쳐 독립 Start 수를 늘리지 않는다.
P Prepare와 D Prepare는 Host 경계다. GPU Kernel 시작 시각이 아니다.
모든 Stage의 합은 Mean TTFT 차이다. TTFT p95를 더해서 분해한 표가 아니다.

| Trace | 비교 | Paired Starts | Host Stage | Mean 차이 ms [95% CI] |
| --- | --- | --- | --- | --- |
| Natural EOS | Layer/StockToken | 3 | Arrival_to_P_submit_ms | 533.9456 [184.2886, 955.7008] |
| Natural EOS | Layer/StockToken | 3 | P_submit_to_accept_ms | 31.8824 [29.1421, 34.4744] |
| Natural EOS | Layer/StockToken | 3 | P_accept_to_prepare_ms | 685.4995 [542.0097, 804.4440] |
| Natural EOS | Layer/StockToken | 3 | P_prepare_to_ready_ms | 206.8980 [186.3493, 222.3493] |
| Natural EOS | Layer/StockToken | 3 | Ready_to_D_submit_ms | -0.0938 [-0.2207, 0.0153] |
| Natural EOS | Layer/StockToken | 3 | D_submit_to_accept_ms | -342.0258 [-709.6690, -5.9785] |
| Natural EOS | Layer/StockToken | 3 | D_accept_to_prepare_ms | -183.5626 [-850.4762, 625.6913] |
| Natural EOS | Layer/StockToken | 3 | D_prepare_to_first_worker_ms | -150.1023 [-189.5895, -110.2431] |
| Natural EOS | Layer/StockToken | 3 | Worker_to_client_ms | -0.0165 [-0.0547, 0.0152] |
| Natural EOS | Layer/StockToken | 3 | TTFT_ms | 782.4246 [-476.9492, 1892.6111] |
| Natural EOS | Layer/StockToken | 3 | D_decode_residency_ms | -20.6361 [-39.6005, 2.3082] |
| Natural EOS | Layer/StockToken | 3 | D_submit_to_finish_ms | -696.3432 [-1789.3484, 511.7772] |
| Natural EOS | Hybrid/StockToken | 3 | Arrival_to_P_submit_ms | 2069.1314 [754.7199, 3496.9914] |
| Natural EOS | Hybrid/StockToken | 3 | P_submit_to_accept_ms | 2.8253 [-1.8265, 6.4870] |
| Natural EOS | Hybrid/StockToken | 3 | P_accept_to_prepare_ms | -87.6592 [-254.6614, 46.9377] |
| Natural EOS | Hybrid/StockToken | 3 | P_prepare_to_ready_ms | 40.1112 [31.2151, 51.3895] |
| Natural EOS | Hybrid/StockToken | 3 | Ready_to_D_submit_ms | 0.7562 [0.3306, 1.1722] |
| Natural EOS | Hybrid/StockToken | 3 | D_submit_to_accept_ms | -410.0671 [-859.7388, 4.8593] |
| Natural EOS | Hybrid/StockToken | 3 | D_accept_to_prepare_ms | -326.2940 [-1294.9278, 621.6643] |
| Natural EOS | Hybrid/StockToken | 3 | D_prepare_to_first_worker_ms | -67.2802 [-133.0305, -2.5848] |
| Natural EOS | Hybrid/StockToken | 3 | Worker_to_client_ms | 0.3785 [-0.0262, 0.9198] |
| Natural EOS | Hybrid/StockToken | 3 | TTFT_ms | 1221.9021 [-1121.1173, 2997.8131] |
| Natural EOS | Hybrid/StockToken | 3 | D_decode_residency_ms | -24.4610 [-59.6294, 13.8396] |
| Natural EOS | Hybrid/StockToken | 3 | D_submit_to_finish_ms | -827.7238 [-2347.3528, 638.6982] |
| Natural EOS | Hybrid/Layer | 3 | Arrival_to_P_submit_ms | 1535.1857 [466.5053, 2581.0270] |
| Natural EOS | Hybrid/Layer | 3 | P_submit_to_accept_ms | -29.0571 [-34.5021, -23.8572] |
| Natural EOS | Hybrid/Layer | 3 | P_accept_to_prepare_ms | -773.1587 [-1047.4855, -525.4625] |
| Natural EOS | Hybrid/Layer | 3 | P_prepare_to_ready_ms | -166.7868 [-186.1320, -148.0423] |
| Natural EOS | Hybrid/Layer | 3 | Ready_to_D_submit_ms | 0.8500 [0.3575, 1.3767] |
| Natural EOS | Hybrid/Layer | 3 | D_submit_to_accept_ms | -68.0413 [-150.0698, 10.8378] |
| Natural EOS | Hybrid/Layer | 3 | D_accept_to_prepare_ms | -142.7314 [-534.9137, 232.5662] |
| Natural EOS | Hybrid/Layer | 3 | D_prepare_to_first_worker_ms | 82.8220 [56.5589, 107.6583] |
| Natural EOS | Hybrid/Layer | 3 | Worker_to_client_ms | 0.3949 [-0.0130, 0.9204] |
| Natural EOS | Hybrid/Layer | 3 | TTFT_ms | 439.4774 [-833.3421, 1599.9647] |
| Natural EOS | Hybrid/Layer | 3 | D_decode_residency_ms | -3.8249 [-20.0289, 11.5314] |
| Natural EOS | Hybrid/Layer | 3 | D_submit_to_finish_ms | -131.3806 [-619.5799, 287.9646] |
| Fixed-length Replay | Layer/StockToken | 3 | Arrival_to_P_submit_ms | 669.2274 [513.5145, 832.5217] |
| Fixed-length Replay | Layer/StockToken | 3 | P_submit_to_accept_ms | 35.3218 [33.2145, 37.0730] |
| Fixed-length Replay | Layer/StockToken | 3 | P_accept_to_prepare_ms | 837.6843 [586.3361, 1112.8322] |
| Fixed-length Replay | Layer/StockToken | 3 | P_prepare_to_ready_ms | 212.9841 [188.3270, 231.5754] |
| Fixed-length Replay | Layer/StockToken | 3 | Ready_to_D_submit_ms | 0.0027 [-0.1276, 0.1045] |
| Fixed-length Replay | Layer/StockToken | 3 | D_submit_to_accept_ms | -108.0476 [-124.8460, -95.1514] |
| Fixed-length Replay | Layer/StockToken | 3 | D_accept_to_prepare_ms | -491.6785 [-730.8644, -194.1543] |
| Fixed-length Replay | Layer/StockToken | 3 | D_prepare_to_first_worker_ms | -197.0108 [-215.5791, -177.8422] |
| Fixed-length Replay | Layer/StockToken | 3 | Worker_to_client_ms | -0.0266 [-0.0519, -0.0074] |
| Fixed-length Replay | Layer/StockToken | 3 | TTFT_ms | 958.4566 [904.7708, 1007.7820] |
| Fixed-length Replay | Layer/StockToken | 3 | D_decode_residency_ms | -14.7709 [-15.9784, -13.8157] |
| Fixed-length Replay | Layer/StockToken | 3 | D_submit_to_finish_ms | -811.5344 [-1086.7711, -482.4059] |
| Fixed-length Replay | Hybrid/StockToken | 3 | Arrival_to_P_submit_ms | 2267.2639 [1825.0306, 2771.9800] |
| Fixed-length Replay | Hybrid/StockToken | 3 | P_submit_to_accept_ms | 8.0775 [2.3613, 13.5696] |
| Fixed-length Replay | Hybrid/StockToken | 3 | P_accept_to_prepare_ms | -55.4915 [-235.9051, 81.6945] |
| Fixed-length Replay | Hybrid/StockToken | 3 | P_prepare_to_ready_ms | 51.8934 [33.9671, 73.9293] |
| Fixed-length Replay | Hybrid/StockToken | 3 | Ready_to_D_submit_ms | 0.1382 [0.0618, 0.2384] |
| Fixed-length Replay | Hybrid/StockToken | 3 | D_submit_to_accept_ms | -159.0072 [-218.8449, -110.3148] |
| Fixed-length Replay | Hybrid/StockToken | 3 | D_accept_to_prepare_ms | -398.9687 [-839.9867, 86.4792] |
| Fixed-length Replay | Hybrid/StockToken | 3 | D_prepare_to_first_worker_ms | -68.5385 [-72.8812, -65.0765] |
| Fixed-length Replay | Hybrid/StockToken | 3 | Worker_to_client_ms | 0.0095 [-0.0067, 0.0300] |
| Fixed-length Replay | Hybrid/StockToken | 3 | TTFT_ms | 1645.3766 [1060.6699, 2403.0618] |
| Fixed-length Replay | Hybrid/StockToken | 3 | D_decode_residency_ms | -12.0920 [-13.3648, -10.8227] |
| Fixed-length Replay | Hybrid/StockToken | 3 | D_submit_to_finish_ms | -638.5969 [-1135.3849, -109.8267] |
| Fixed-length Replay | Hybrid/Layer | 3 | Arrival_to_P_submit_ms | 1598.0365 [1220.3567, 2088.5507] |
| Fixed-length Replay | Hybrid/Layer | 3 | P_submit_to_accept_ms | -27.2443 [-33.1134, -22.6392] |
| Fixed-length Replay | Hybrid/Layer | 3 | P_accept_to_prepare_ms | -893.1757 [-1344.9539, -514.5367] |
| Fixed-length Replay | Hybrid/Layer | 3 | P_prepare_to_ready_ms | -161.0907 [-187.3535, -139.1814] |
| Fixed-length Replay | Hybrid/Layer | 3 | Ready_to_D_submit_ms | 0.1355 [0.0273, 0.2301] |
| Fixed-length Replay | Hybrid/Layer | 3 | D_submit_to_accept_ms | -50.9596 [-123.6935, 14.5312] |
| Fixed-length Replay | Hybrid/Layer | 3 | D_accept_to_prepare_ms | 92.7098 [-418.6290, 730.4717] |
| Fixed-length Replay | Hybrid/Layer | 3 | D_prepare_to_first_worker_ms | 128.4723 [112.7657, 142.6980] |
| Fixed-length Replay | Hybrid/Layer | 3 | Worker_to_client_ms | 0.0361 [0.0152, 0.0536] |
| Fixed-length Replay | Hybrid/Layer | 3 | TTFT_ms | 686.9200 [108.0501, 1413.5750] |
| Fixed-length Replay | Hybrid/Layer | 3 | D_decode_residency_ms | 2.6789 [2.1466, 3.1716] |
| Fixed-length Replay | Hybrid/Layer | 3 | D_submit_to_finish_ms | 172.9375 [-359.7033, 864.8096] |

Stage 차이는 지연이 나타난 위치를 설명한다. GPU 연산 또는 Credit 하나의 인과 효과를 확정하지 않는다.
Layer와 Hybrid의 D Host 이득이 P 앞 Queue와 P Host 비용을 모두 상쇄하지 못할 수 있다.
Credit 및 Compiler Intervention 결과는 위의 별도 Fixed-length Replay에서 확인한다.

[c3_heldout_serving_v2_host_path_v2_comparisons.csv](dense_ar_longbench400_v2_20261122/figures/c3_heldout_serving_v2_host_path_v2_comparisons.csv)

## 남은 한계

일부 Stock Strict 불일치는 재실행에서 재현되지 않았다. 근본 원인은 미확정이다.
적격 Operator-exclusive GEMM/Attention Breakdown은 별도 GPU Gate가 없으면 `NOT_MEASURED`다.
최초 Geometry 선택 뒤 Compiler를 개선했다. 모든 Cut의 재튜닝은 `NOT_RUN`일 수 있다.
따라서 Best는 테스트한 후보 집합 안의 선택이다. 전역 최적이라고 표현하지 않는다.
