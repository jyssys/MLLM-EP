# Context Parallelism × Expert Parallelism Direct Handoff PoC — Working Contract

이번 작업은 기존 scheduling/overlap successor screening 이후 새롭게 여는
`Context Parallelism × Expert Parallelism` focused PoC다.

목표는 단순히 CP와 EP를 함께 켜는 것이 아니다.

핵심 연구 질문:

"긴 MLLM MoE prefill에서 CP Attention 결과가 EP MoE로 넘어가는 경계가
불필요한 materialization/layout conversion/synchronization을 만들며,
정확한 token-block streaming handoff 또는 direct CP-to-expert transport로
request-critical latency를 줄일 수 있는가?"

이번에는 다음 두 candidate를 headroom-first로 검증한다.

1. Exact Streaming Handoff
   - exact attention output이 준비된 token/query block부터 Router와 EP Dispatch 시작
   - 뒤쪽 CP Attention과 앞쪽 EP MoE를 lossless하게 pipeline

2. Exact Direct CP-to-Expert Transport
   - Ulysses/A2A형 CP에서 post-attention redistribution 후 full hidden을
     materialize하고 다시 EP dispatch하는 경계를 재설계
   - distributed norm/router + route-aware shard transport로 expert owner에서
     full expert input을 직접 조립할 수 있는지 검증

==================================================
0. Mandatory working contract
==================================================

다음 spec을 repository에 생성하고 전체를 읽어 working contract로 사용하라.

poc_flashvep/reports/cp_ep_direct_handoff_poc_spec.md

사용자가 제공한 `cp_ep_direct_handoff_poc_spec.md` 전문을 그대로 저장한다.

Branch:

flashvep/cp-ep-direct-handoff-poc

==================================================
1. GPU rule — absolute
==================================================

모든 GPU 실행은 반드시:

CUDA_VISIBLE_DEVICES=4,5,6,7

만 사용한다.

물리 GPU 4/5/6/7 외 GPU 사용 금지.

시작 전에 nvidia-smi로 확인하고,
이전 작업에서 자신이 실행한 burn/process가 남아 있다면 종료한다.
다른 사용자의 process는 절대 종료하지 않는다.

GPU가 사용 불가능하면 억지로 실행하지 말고 정확히 보고한다.
Burn 금지.

==================================================
2. First action: prior-art and runtime audit
==================================================

바로 custom kernel을 구현하지 마라.

먼저 다음을 source/paper 수준에서 확인한다.

- NanoCP
- Helix Parallelism / HOP-B
- Megatron MoE Parallel Folding
- Ring/pass-KV/pass-Q CP
- DeepSpeed Ulysses/A2A CP
- 현재 TensorRT-LLM/SGLang/Megatron의 CP+EP support

특히 질문:

"이미 CP output을 EP input으로 exact streaming/direct handoff하는가?"

그리고 local environment에서 가장 faithful한 실행 path를 선택한다.

우선순위:

A. existing inference CP+EP path
B. Megatron forward-only CP+EP layer replay
C. actual Qwen hidden/weight 기반 minimal exact layer replay

하나의 environment setup에 2시간 이상 막히면 alternate path로 전환한다.
Environment failure를 method failure로 부르지 않는다.

==================================================
3. Mandatory tensor-layout proof
==================================================

코딩 전에 TENSOR_LAYOUT_CONTRACT.md를 작성한다.

Ring/pass-KV와 Ulysses/A2A를 구분한다.

Ring CP라면 각 rank가 local sequence shard의 full hidden output을 이미 가지므로
post-attention gather가 없을 수 있다.
이 경우 one-collective fusion을 주장하지 말고 exact block streaming만 본다.

Ulysses/A2A라면:

full sequence × hidden/head shard
→ return A2A
→ sequence shard × full hidden
→ norm/router
→ EP dispatch

경계가 실제로 존재하는지 source와 trace로 확인한다.

Direct transport를 구현하려면 먼저 다음 exact algebra를 증명한다.

- sharded residual/RMSNorm 가능성
- partial router logits + exact reduction
- top-k route identity
- hidden shard를 expert owner로 direct transport해 full H 조립
- expert combine 후 next CP layout

성립하지 않으면 ALGEBRAICALLY_BLOCKED로 정직하게 판정한다.

==================================================
4. Primary model/workloads
==================================================

Primary:
Qwen3-VL-30B-A3B-Instruct, BF16.

Primary phase:
long-context prefill.

Representative layers:
early / middle / late, 예: 4/24/44.

Context lengths:
4K, 8K, 16K, 32K, 64K if supported.

Workloads:
- text-only
- vision-heavy
- mixed
- multi-image

MLLM 비교는 actual model-input token count를 ±2% 이내로 맞춘다.
Nominal prompt length만으로 matched라고 쓰지 마라.

Kimi는 Qwen에서 strong gate를 통과한 뒤에만 실행한다.

==================================================
5. Gate 1: CP crossover
==================================================

먼저 CP1/CP2/CP4가 4×H100에서 실제로 이기는 long-context regime이 있는지 확인한다.

Measure:
- attention wall
- CP communication
- peak memory
- first/last block ready
- Attention+MoE layer wall
- native path가 있으면 TTFT/E2E

CP2/4가 16K+ 어느 regime에서도 이기지 않고 memory 필요성도 없으면
NO_CP_REGIME로 종료 가능.

==================================================
6. Gate 2: CP→EP boundary mass
==================================================

다음을 세부 계측한다.

CP attention end
→ output projection/return redistribution
→ residual/RMSNorm
→ Router/top-k
→ route layout/permutation
→ EP dispatch
→ first expert start

같은 GPU의 CUDA event만 duration에 사용.
Cross-GPU absolute timestamp subtraction 금지.
Detailed trace와 clean timing을 분리하고 observer overhead를 측정한다.

계산:
- materialization-free oracle
- barrier-free oracle
- block-streaming oracle
- direct-transport byte/time oracle

request/layer headroom <5%면 NO_HANDOFF_MASS.

==================================================
7. Gate 3A: Exact Streaming Handoff
==================================================

Baseline:
모든 CP block 완료 후 Router/EP.

Candidate:
block b의 exact attention output이 ready되는 즉시
block b Router/layout/EP dispatch/expert를 시작하고,
동시에 block b+1 CP attention을 계속한다.

Approximation/speculation 금지.

Block size sweep:
64/128/256/512/1024 tokens, actual kernel semantics에 맞게 조정.

반드시 비교:
- serial baseline
- streaming oracle
- real streaming
- same chunking but no overlap negative control

Small EP chunk overhead와 resource contention을 포함한다.

==================================================
8. Gate 3B: Exact Direct CP-to-Expert Transport
==================================================

Ulysses/A2A path에서만 수행.

구현 전 actual route snapshot을 이용해 bytes/messages를 계산한다.

Baseline:
post-attention A2A + full hidden materialization + Router + EP dispatch.

Candidate:
- distributed norm statistic
- distributed router partial logits + exact reduction
- route-aware hidden-shard transport
- expert owner에서 full expert input assembly
- exact expert execution/combine

추가 all-reduce, metadata, assembly 비용을 모두 포함한다.

Analytical layer oracle <8%면 구현하지 않는다.
>=12%일 때 bounded exact prototype을 허용한다.

"one All-to-All"이라는 이름을 위해 비용을 숨기지 마라.

==================================================
9. Correctness
==================================================

Reference와 비교:
- attention output cosine/L2
- router top-k agreement
- expert input reconstruction
- final layer output
- integrated path first-token greedy agreement

Target:
cosine >=0.9999,
relative L2 <=1%,
route agreement >=99.9%, 가능하면 100%.

Correctness가 깨진 성능 run은 모두 제외한다.

==================================================
10. MLLM matched gate
==================================================

같은 actual token count에서 Text/Vision/Mixed를 비교한다.

질문:
- optimal CP degree가 다른가?
- exact-ready spread가 다른가?
- CP→EP handoff mass가 다른가?
- streaming/direct gain이 다른가?

차이가 token volume으로 설명되면 MLLM-specific claim을 하지 않는다.
Generic CP→EP result가 강하면 generic 방향으로 재분류한다.

==================================================
11. Request-level gate
==================================================

Single-layer gain을 paper E2E로 부르지 마라.

Layer-level actual gain >=10% 또는 projected TTFT >=10%인 경우에만
full-model bounded integration을 시도한다.

Primary request metrics:
TTFT, E2E, prefill throughput, memory, correctness.

Key result는 3 independent restarts,
가능하면 5 paired restarts.

==================================================
12. Kill gates
==================================================

NO_GO if any decisive condition:

- CP regime 없음
- handoff mass <5%
- streaming oracle <5%
- direct transport analytical oracle <8%
- actual gain <5%
- exact correctness 유지 시 gain 소멸
- simple existing option이 benefit의 >=80% 회수
- prior art direct collision

HOLD only if oracle is strong but exact prototype/request integration remains genuinely incomplete.

STRONG_GO는 Qwen actual layer >=15% 또는 direct prefill TTFT/E2E >=12%,
correctness, non-triviality, generality/prior-art gate를 요구한다.

==================================================
13. Experimental discipline
==================================================

- warmup >=5, microbenchmark measured >=30
- randomized/interleaved policy order
- same route/input/shape
- clean timing primary, Nsight secondary
- observer tax report
- cold/JIT/allocator/DVFS confound 통제
- rank rows를 request latency에 중복 합산 금지
- layer proxy를 request speedup으로 둔갑시키지 마라

==================================================
14. Expected effort
==================================================

Initial decisive audit/crossover/handoff gate:
약 4–8시간.

Streaming prototype까지:
약 6–14시간.

Direct transport가 analytical gate를 통과해 구현하면:
약 12–24시간까지 가능.

초기 headroom gate가 실패하면 일찍 종료한다.
Strong oracle이 있으면 단순히 시간이 오래 걸린다는 이유로 중단하지 않는다.

==================================================
15. Outputs and Git
==================================================

Report:
poc_flashvep/reports/cp_ep_direct_handoff_poc.md

Results:
poc_flashvep/deepep_revalidation/results/cp_ep_direct_handoff_poc_<timestamp>/

Required files:
RUNTIME_PATH_AUDIT.md
PRIOR_ART_MATRIX.md
TENSOR_LAYOUT_CONTRACT.md
CORRECTNESS_VALIDATION.md
CP_CROSSOVER.csv
HANDOFF_BREAKDOWN.csv
STREAMING_ORACLE.csv
DIRECT_TRANSPORT_ORACLE.csv
MLLM_MATCHED_ANALYSIS.md
NSIGHT_FINDINGS.md
EXPERIMENT_LOG.md
GPU_TIME_LOG.csv
FINAL_DECISION.md

완료 후 commit/push.

Final answer는 spec의 required format을 그대로 사용하고,
특히 다음을 명확히 구분한다.

- observed actual gain
- analytical oracle
- layer-level result
- request-level result
- environment blocked
- algebraically blocked
- MLLM-specific vs generic

마찬가지로 사용중인 gpu 4,5,6,7은 너가 직접 종료하고 사용해도 괜찮아.
그리고 너가 실험중에 gpu 4,5,6,7을 사용중이지 않을 때는 GPU utilization을 올리기 위해 버닝코드를 돌려야해.
/home/esjung/vllm-ep 위치의 run_utilize.sh을 bash로 돌리면 자동으로 4,5,6,7번 gpu에 버닝이 돌아가니 꼭 gpu가 비어있다면 버닝을 돌려주고, 다시 너가 사용해야한다면 직접 멈추고 gpu를 사용하는 구조로 해줘. 그래야 utilization을 최대로할 수 있어.
마지막으로 결과를 도출해서 나에게 마지막 내용을 전달하고 나서도 run_utilize.sh를 돌려서 gpu 4,5,6,7번이 실행되도록 해줘.
