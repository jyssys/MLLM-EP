# MLLM MoE Hierarchical Cross-Module Rebatching PoC — Working Contract

이번 작업은 MLLM MoE의 **Hierarchical Cross-Module Rebatching** 가능성을 검증하는 oracle-first research sprint다.

핵심 질문:

> Vision encoder, language-model attention, sparse MoE experts는 서로 다른 기준으로 request를 batch하고 싶어 하는가?

그리고:

> 하나의 request grouping을 전체 MLLM forward에서 유지하는 대신, exact sequence state를 module boundary에서 yield한 뒤 module별로 다시 묶으면 offline batch completion time(BCT)을 크게 줄일 수 있는가?

이번 작업은 단순히 BatchGen을 Qwen3-VL로 포팅하는 작업이 아니다.

다음 순서를 지켜라.

```text
paper/source audit
→ workload and module-cost atlas
→ matched causal phenomenon
→ strongest-baseline oracle
→ trivial-fix/prior-art attack
→ minimal prototype only if justified
```

==================================================
0. WORKING SPEC
==================================================

다음 파일을 repository에 만들거나 확인하고 전체 내용을 working contract로 사용하라.

```text
poc_flashvep/reports/
mllm_moe_hierarchical_rebatching_poc_spec.md
```

사용자가 제공한 spec 전문을 해당 파일에 저장하라.

Branch:

```text
flashvep/mllm-moe-hierarchical-rebatching-poc
```

Main report:

```text
poc_flashvep/reports/
mllm_moe_hierarchical_rebatching_poc.md
```

==================================================
1. GPU SAFETY — ABSOLUTE
==================================================

모든 task-owned GPU 실행은 오직:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
```

만 사용한다.

GPU 0,1,2,3은 절대 사용하지 마라.

각 launch 전에 GPU UUID, logical mapping, process owner, available memory를 확인한다.

다른 사용자의 process를 종료하지 마라.

task-owned burn이 있으면 command/PID/owner를 확인한 뒤 measurement 전에만 종료한다.

새 burn을 실행하거나 자동 재시작하지 마라.

==================================================
2. FIRST ANSWER THE SCOPE QUESTION
==================================================

Primary scope는 **offline/asynchronous batch inference**다.

즉 많은 요청이 하나의 batch job으로 제출되고:

```text
BCT = 모든 요청이 완료될 때까지의 시간
```

을 줄이는 것이 목표다.

이것은 interactive serving의 TTFT/TPOT 최적화와 다르다.

Optional nearline/wave-arrival workload는 strong offline signal이 나온 뒤에만 수행한다.

Offline result를 interactive serving result로 부르지 마라.

==================================================
3. AUDIT MOE-GEN AND BATCHGEN
==================================================

최소 다음을 읽고 비교한다.

- MoE-Gen, arXiv:2503.09716
- BatchGen, OSDI 2026 / current official repository
- paper artifact pin and current main separately
- vLLM-Omni stage-wise batching/async chunk
- ModServe
- ElasticMM
- RPS-Serve
- BlendServe
- HeteroServe
- any newer exact prior art

특히 BatchGen paper가:

- attention→MoE boundary에서 YIELD/COMBINE;
- intra-forward yield points를 model compute 특성에 따라 once/static하게 선택;
- vision encoder 뒤 yield를 VLM future work로 언급

한다는 점을 확인하되, current main이 이미 바뀌었는지 반드시 감사한다.

다음은 novelty가 아니다.

- attention batch와 MoE batch를 다르게 사용;
- vision encoder 뒤에서 한 번 yield;
- stage마다 독립 batch size;
- sequence coroutine abstraction 자체;
- large expert batch formation;
- dynamic sequence migration/partition.

Create:

```text
SOURCE_AND_PRIOR_ART_AUDIT.md
BATCHGEN_PAPER_VS_MAIN.md
```

==================================================
4. DO NOT PORT BATCHGEN MLLM FIRST
==================================================

처음부터 full Qwen3-VL BatchGen port를 만들지 마라.

먼저 기존 verified Qwen3-VL stack과 actual module replay를 이용해:

- vision cost;
- LM attention cost;
- router/MoE cost;
- yield/combine/memory cost;
- hierarchical scheduling oracle

을 측정한다.

Native BatchGen MLLM port는 feasible oracle이 강할 때만 허용한다.

한 environment/port 문제에 90–120분 이상 묶이면 fallback diagnostic으로 전환하고 `ENVIRONMENT_BLOCKED`를 정확히 기록한다.

==================================================
5. PRIMARY MODEL AND TOPOLOGY
==================================================

Primary:

```text
Qwen3-VL-30B-A3B-Instruct
```

가능하면 project에서 이미 검증한 four-GPU MLLM MoE path를 사용한다.

예:

```text
TP2 / DP2 / EP4
```

단 runtime/source로 actual semantics를 다시 확인한다.

TP-only를 EP라고 부르지 마라.

Kimi는 Qwen에서 finalist gate를 통과한 뒤에만 실행한다.

==================================================
6. WORKLOADS
==================================================

최소 다음 classes를 준비한다.

```text
text-only
low-resolution image
high-resolution image
multi-image
long-text + image
natural mixed pool
```

Batch sizes:

```text
16 / 32 / 64 / 128
```

Output modes:

```text
max_tokens=1
fixed 16/32
teacher-forced/fixed-length control
natural EOS
```

Primary BCT workload는 offline closed-set batch다.

Strong signal이 나온 뒤 optional wave arrivals를 추가한다.

==================================================
7. BUILD THE MODULE COST ATLAS
==================================================

먼저 scheduler 없이 actual costs를 측정한다.

Vision:

- image count/resolution/grid;
- vision token count;
- preprocess/encoder/projector;
- padding or shape waste.

Attention:

- LM token lengths;
- phase;
- KV lengths;
- attention time;
- packed/padded efficiency.

MoE:

- routes;
- expert histogram;
- active experts;
- expert batch sizes;
- rank load;
- dispatch/expert/combine.

Yield/rebatch:

- hidden-state bytes;
- checkpoint/materialization;
- combine;
- metadata;
- GPU buffer;
- host spill;
- order restoration.

Clean BCT and observer-heavy trace must be separate.

==================================================
8. CENTRAL MATCHED EXPERIMENT
==================================================

Construct request pools with the same one-dimensional totals:

- request count;
- total pixels;
- total vision tokens;
- total LM tokens;
- image-count histogram;
- resolution histogram;
- LM-length histogram;
- output budget.

Create:

1. ALIGNED
2. CONFLICTED
3. RANDOM
4. NATURAL

The only intended difference is the correlation between:

```text
vision compatibility
attention compatibility
MoE route/expert-density compatibility
```

Question:

> Can two batches with the same total work have materially different BCT because module-optimal groupings conflict?

Programmatically assert the matched marginals.

Do not accept a synthetic result until natural-request frequency and actual module replay support it.

==================================================
9. BASELINES
==================================================

Measure or faithfully emulate:

P0. global fixed grouping
P1. vision-only buckets
P2. LM-length-only buckets
P3. BatchGen-style attention→MoE combine
P4. independent stage batching
P5. static hierarchical vision + attention→MoE yields
P7. feasible perfect hierarchical oracle

Positive claim must compare against the best of P1–P4, not P0 alone.

==================================================
10. ORACLE FIRST
==================================================

Build a precedence- and memory-aware schedule oracle from measured module cost tables.

Do not sum independent best module times as if they could all overlap freely.

Account for:

- exact dependencies;
- finite GPU/host memory;
- yielded hidden states;
- regrouping;
- combine overhead;
- batch waiting;
- request order restoration;
- actual GPU fleet usage.

Report:

- optimistic oracle;
- feasible oracle;
- vision incremental value over BatchGen-style LM-only;
- MoE incremental value over stage-only MLLM batching;
- adaptive-yield value over best static yield graph.

==================================================
11. HEADROOM GATES
==================================================

Best-existing-baseline 대비 direct BCT oracle:

```text
<8%     → NO-GO
8–12%   → weak
12–15%  → promising
>=15%   → serious
>=20%   → strong
```

Feasible oracle:

```text
<8%      → scheduler 구현 금지
>=10–12% → finalist 가능
>=15%    → strong
```

MoE-specificity:

```text
full hierarchy must beat independent vision/LM stage batching by >=5% BCT
```

MLLM-specificity:

```text
full hierarchy must beat BatchGen-style LM-only batching by >=8% BCT
```

위 조건을 통과하지 못하면 generic MLLM 또는 generic BatchGen extension으로 재분류한다.

==================================================
12. TRIVIAL FIX ATTACK
==================================================

반드시 다음을 먼저 시험한다.

- high/low resolution two buckets;
- short/long LM two buckets;
- text/image split;
- single/multi-image split;
- static encoder yield;
- static MoE yield;
- simple queue threshold;
- best BatchGen module batch sizes.

이 중 하나가 oracle의 80% 이상을 회수하면서 memory/SCT를 악화시키지 않으면:

```text
FOUND_INCREMENTAL_ONLY
```

복잡한 method를 만들지 마라.

==================================================
13. AUTONOMOUS HYPOTHESIS TREE
==================================================

최소 15개의 distinct causal hypothesis를 실제 measurement로 테스트한다.

Seed questions:

- same totals, different cross-module correlation;
- vision grouping hurts attention;
- attention grouping hurts MoE density;
- MoE waiting erases vision gain;
- static yield points fail across input classes;
- batch size alone is insufficient;
- yield overhead boundary;
- GPU vs host state buffering;
- natural output tail;
- when-to-combine;
- two buckets recover oracle;
- more yield points can hurt;
- static per-model vs per-pool plan;
- MLLM vs text-only;
- Kimi generality after promotion.

Seed를 그대로 체크리스트처럼 끝내지 말고 residual에서 child hypothesis를 생성한다.

각 node:

```text
EXPECTED
OBSERVED
FAILED ASSUMPTION
NEW SYSTEM FACT
BCT IMPLICATION
NEXT CHILDREN
```

을 기록한다.

==================================================
14. MINIMAL PROTOTYPE
==================================================

Feasible oracle가 >=10–12%일 때만 prototype을 만든다.

Prototype A:

```text
shape-aware vision batches
→ exact vision embeddings yield
→ LM-length-aware regroup
→ exact LM execution
```

Prototype B:

```text
vision regroup
+
BatchGen-style attention→MoE combine
```

Prototype C:

```text
choose no-yield / encoder-only / MoE-only / both
```

C는 A/B가 성공한 뒤에만.

At most two prototypes.

Production engine rewrite 금지.

==================================================
15. FINAL EVALUATION
==================================================

Headline actual prototype result:

- 5 independent paired restarts;
- randomized order;
- restart as statistical unit;
- exact/correct output;
- BCT primary;
- SCT p50/p90/p99;
- throughput;
- GPU/host memory.

Do not headline:

- module-only speedup;
- group makespan proxy;
- predicted oracle;
- observer-heavy run.

==================================================
16. FINAL STATUS
==================================================

Choose one:

```text
FOUND_STRONG_GO
FOUND_PROMISING_ORACLE_ONLY
FOUND_GENERIC_MLLM_BATCHING_ONLY
FOUND_INCREMENTAL_ONLY
NO_HIERARCHICAL_HEADROOM
ENVIRONMENT_BLOCKED
```

Strong GO requires:

- causal module compatibility conflict;
- direct feasible BCT >=10–12%;
- actual prototype moves BCT >=5%;
- best stage-only and BatchGen-style baselines beaten;
- simple buckets insufficient;
- no direct prior-art collision;
- clear path to Qwen+Kimi generality.

==================================================
17. REQUIRED OUTPUTS
==================================================

Maintain all files listed in the spec, including:

```text
MODULE_COST_ATLAS.csv
MATCHED_POOL_CONTROLS.csv
SCHEDULE_BASELINES.csv
HIERARCHICAL_ORACLES.md
BCT_RESULTS.csv
SCT_RESULTS.csv
MEMORY_RESULTS.csv
TRIVIAL_FIX_ATTACKS.md
RESEARCH_TREE.md
SURPRISE_LOG.md
GPU_TIME_LOG.csv
```

Final response must answer:

1. Is offline MLLM batch inference materially different from interactive serving?
2. Do vision, attention, and MoE actually want different groupings?
3. Is total token count enough to predict BCT?
4. Does hierarchical rebatching beat stage-only MLLM batching?
5. Does it beat BatchGen-style LM-only rebatching?
6. Does MoE routing add independent value?
7. Can simple buckets recover the oracle?
8. What is the direct BCT headroom?
9. What exact gap remains beyond BatchGen and vLLM-Omni?
10. Is there a clean paper-level method?

완료 후 commit/push하고 branch, commit, report, result path를 남겨라.

마찬가지로 사용중인 gpu 4,5,6,7은 너가 직접 종료하고 사용해도 괜찮아.
그리고 너가 실험중에 gpu 4,5,6,7을 사용중이지 않을 때는 GPU utilization을 올리기 위해 버닝코드를 돌려야해.
/home/esjung/vllm-ep 위치의 run_utilize.sh을 bash로 돌리면 자동으로 4,5,6,7번 gpu에 버닝이 돌아가니 꼭 gpu가 비어있다면 버닝을 돌려주고, 다시 너가 사용해야한다면 직접 멈추고 gpu를 사용하는 구조로 해줘. 그래야 utilization을 최대로할 수 있어. 마지막으로 결과를 도출해서 나에게 마지막 내용을 전달하고 나서도 run_utilize.sh를 돌려서 gpu 4,5,6,7번이 실행되도록 해줘
