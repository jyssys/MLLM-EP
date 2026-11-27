# 강한 일반 대조 대비 Architecture 효과

## CogVLM2

동일 모델 내에서 먼저 일반 최적화를 통제한다. standalone Vision encoder는 모든 D-only Arm에서 빼고, Resident/Fast/Pruned 모두 같은 D GPU, Prompt KV와 생성된 Language role을 받는다. Native Original은 pinned upstream Decoder path이며, Resident Text Fast는 원래 산술을 보존하면서 zero-row expert dispatch를 건너뛰고 bank를 GPU에 유지한다. Architecture arm은 Visual bank를 명시적으로 제외한다. Generic reachability는 동일한 D phase의 출력을 알고 empty row 공급 경로의 exclusive parameter를 원본 Source에서 찾는다. 두 Pruned arm은 Model-specific AST로 검증하지만 실제 제외한 Key, Host forwarding path, KV Layout과 GPU Weight bytes가 동일하다. 구조의 역할은 **어떤 Weight가 Stage에서 도달 불가능한지를 보장하는 것**이다. [Liveness와 실제 Rank 거동](mllm_modality_parameter_state_liveness.md)

공통 두 GPU의 serving 대조에 같은 모델을 두 장에 각각 배치한 DP2를 포함한다. 공유 가능한 KV cache slot 실험과 두 GPU의 실제 open-loop는 서로 다른 측정이다. Co-located DP2에는 전달 비용이 없고, P+D에서는 E까지 P에 놓인 대신 지속 Decode batch를 D에서 실행할 수 있다. 생산 vLLM CogVLM2 path는 해당 환경에서 없었으며 이 대조군은 해당 checkpoint의 publisher 모델 기반 구현으로 한정된다. [실제 Online pilot](mllm_architecture_disaggregation_online_pilot.md)

## IDEFICS

Stock HF는 동일 immutable image latent를 8개 Cross layer에서 반복 투영한다. 강한 Generic은 동일 K/V를 D에서 최초 한 번만 만들고, 선택적으로 단일 packed GEMM으로 투영할 수 있다. 따라서 P에서 투영된 K/V를 미리 전달하는 Arm의 이득 전체를 stock HF 대비 수치로 해석하면 공통 memoization 효과가 포함된다.

두 번째 대조는 동일한 last-query access mask에서 도달 가능한 슬롯만 남긴다. 이 대조는 Image Label과 별개인 generic masked-state transport다. Compact SDPA와 retained-shape masked zero padding도 분리한다. 동일 mask, 동일 storage와 ALU 조건에서 visual slot을 고른 별개 스케줄러의 이득은 분리되지 않는다. [실험 경계와 모드](mllm_modality_phase_execution_graph.md)

## 기존 방법과의 관계

CogVLM2의 Scope는 stage별 Weight reachability다. 기존 EPD의 encoder 제거에 추가하여 Decoder 내부 실제 행의 expert bank를 나눈다. Dynamic PD, TP/PP/CP 및 일반 EP를 사용하더라도 Reachability-aware Weight loading을 구현하면 같은 효과를 낼 수 있다. IDEFICS의 static cross K/V는 기존 Encoder–Decoder cache와 겹친다. ModServe의 Cross-attention 범위와도 구조가 겹친다. AFD의 per-layer 통신 또는 Expert placement는 문헌 기준으로 비교했다. [범위 비교](mllm_architecture_prior_work_novelty_audit.md)

실제 IDEFICS 24-request의 Full P+D에서 Generic D memoization은 Stock recomputation 대비 요청별 비율 중앙값 −3.27%, Generic packed D는 −3.17%였다. Whole P-projection은 Generic packed D 대비 +0.09%였다. 같은 visibility와 D shape를 맞춘 원본 Raw+D 대비 P-projected 전달은 −0.12%였다. 두 마지막 비교의 요청 부트스트랩 구간은 0을 포함한다. Hook을 포함한 하나의 isolating GPU epoch에서 같은 요청을 세 번 돌렸으며 서버의 독립 시작이 아니다. [비율과 요청별 범위](../mllm_architecture_first_disaggregation_discovery/results/idefics/summary.json)

ERNIE의 native Triton FusedMoE는 Language decode의 visual compute를 기존부터 생략한다. 이번 12.67 GiB 변화는 scope-guarded local bank binding release의 allocation 효과다. Generic phase reachability와 분리되는 이득을 측정하려면 동일 native backend에서 일반 egress role policy를 포함한 D loader와 production PD가 추가로 필요하다. [Bank call과 실제 allocation](../mllm_architecture_first_disaggregation_discovery/results/ernie45vl/native_elision.json)
