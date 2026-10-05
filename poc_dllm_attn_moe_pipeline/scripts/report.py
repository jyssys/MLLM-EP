"""Build evidence-only figures and an explicit Stage-A stop handoff."""
import csv
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_dllm_attn_moe_pipeline.contract import *


def main():
    results=ROOT/'results/main'
    fig=ROOT/'figures/main'
    fig.mkdir(parents=True,exist_ok=True)
    summary=json.loads((results/'summary.json').read_text())
    correctness=json.loads((results/'correctness.json').read_text())
    request_rows=list(csv.DictReader((results/'request_metrics.csv').open()))
    paired=list(csv.DictReader((results/'producer_paired.csv').open()))
    consumers=json.loads((results/'consumer_components.json').read_text())
    native=json.loads((ROOT/'raw/main_v1/coding_00.events.0.rank0.json').read_text())
    events=[e for e in native['events'] if e['call_id']==1 and e['layer']==9]
    start=next(e['start_us'] for e in events if e['stage']=='moe')
    names=['router','dispatch','expert','combine','shared','source_all_gather']
    f,ax=plt.subplots(figsize=(9,3.5))
    for i,name in enumerate(names):
        e=next(e for e in events if e['stage']==name)
        ax.broken_barh([(e['start_us']-start,e['duration_us'])],(i-.3,.6))
    gather=next(e for e in events if e['stage']=='source_all_gather')
    ax.axvline(gather['start_us']-start,color='orange',ls='--',label='safe local publication')
    ax.axvline(gather['end_us']-start,color='red',ls='--',label='TP-visible whole-buffer release')
    ax.set_yticks(range(len(names)),names)
    ax.set_xlabel('Same-device CUDA-event offset (µs); heavy attribution UNVALIDATED')
    ax.set_title('Calibration request coding_00, hot call1 / layer9 / rank0\nIndividual-row mathematical completion UNOBSERVABLE')
    ax.legend(fontsize=8)
    f.tight_layout();f.savefig(fig/'native_publication_timeline.png',dpi=150);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3))
    component=json.loads((ROOT/'raw/components_v3/components.rank0.json').read_text())
    group_case=next(r for r in component['records'] if r['kind']=='producer' and r['layer']==9 and r['variant']=='split_serial' and r['repetition']==2)
    gap=group_case['tp_visible_gap_us']/1000
    ax.step([-gap*1.2,0,gap*.15],[0,1,1],where='post',label='native exposed release: all groups together')
    ax.step([-gap*1.2,-gap,0,gap*.15],[0,.5,1,1],where='post',label='engineered2-group: actual release-event gap')
    ax.set_xlabel('ms relative to last group publication (same-device local event clock)')
    ax.set_title('Calibration layer9 / rank0: release-only alignment, not rank clocks')
    ax.set_ylabel('Exposed groups ready fraction');ax.legend(fontsize=8)
    f.tight_layout();f.savefig(fig/'group_ready_fraction.png',dpi=150);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3.5))
    layers=(1,9,17)
    values_u=[np.median([float(r['unsplit_us']) for r in paired if int(r['layer'])==l])/1000 for l in layers]
    values_s=[np.median([float(r['split_us']) for r in paired if int(r['layer'])==l])/1000 for l in layers]
    x=np.arange(3)
    ax.bar(x-.18,values_u,width=.36,label='unsplit producer')
    ax.bar(x+.18,values_s,width=.36,label='split serial producer')
    ax.set_xticks(x,[f'Layer{l}' for l in layers]);ax.set_ylabel('Actual EP4 producer ms')
    ax.set_title('All shared/add/source-all-gather/repack costs included\nCalibration component probe, not an overlap seam')
    ax.legend();f.tight_layout();f.savefig(fig/'producer_split_tax.png',dpi=150);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3))
    for i,key in enumerate(('qkv_times_us','prefix_times_us','current_times_us','merge_times_us')):
        values=[np.median(e[key])/1000 for e in consumers]
        ax.bar(i,np.median(values))
    ax.set_xticks(range(4),['norm/QKV/RoPE','prefix FP32','fresh C FP32','FP32 merge'])
    ax.set_ylabel('Component ms');ax.set_title('Native cache snapshot component-only measurements\nNot an optimized attention kernel or seam speedup')
    f.tight_layout();f.savefig(fig/'consumer_component_costs.png',dpi=150);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3))
    ax.bar([0],[0],label='natural published-interface envelope=0%')
    ax.set_xlim(-.5,2.5);ax.set_ylim(-1,10)
    ax.set_xticks([0,1,2],['Natural interface','Engineered estimate','Actual overlap'])
    ax.text(1,4,'UNRESOLVED',ha='center');ax.text(2,4,'NOT_RUN',ha='center')
    ax.set_ylabel('Whole-request latency reduction %');ax.legend(fontsize=8)
    f.tight_layout();f.savefig(fig/'headroom_scope.png',dpi=150);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3))
    hot=sum(c['phase']=='hot' for c in native['calls']);cold=len(native['calls'])-hot
    ax.bar(['Hot cached native calls','Cold calls','Early-visible native groups'],[hot,cold,0])
    ax.set_title('Calibration invocation coverage; cold has serial fallback')
    f.tight_layout();f.savefig(fig/'boundary_coverage.png',dpi=150);plt.close(f)
    # No synthetic task gains: the missing prefix sweep is visibly NOT_RUN.
    f,ax=plt.subplots(figsize=(7,2.5));ax.axis('off')
    ax.text(.5,.5,'Prefix-length vs net gain: NOT_RUN\nNo candidate overlap seam / no synthetic task results',ha='center',va='center')
    f.tight_layout();f.savefig(fig/'prefix_gain_not_run.png',dpi=150);plt.close(f)
    ci=json.loads((results/'bootstrap_ci.json').read_text())['observer_overhead_pct']
    text=f'''# dLLM intra-refinement Attention–MoE pipeline — Stage-A discovery

## Outcome

**CONDITIONAL_INTERFACE_BLOCKED.** Real four-GPU EP and unchanged native
generation are validated. There is **no naturally exposed early token-group
publication** in this bridge. Its next-layer consumer waits for complete
combine, shared-expert/add, and source-row all-gather. Latent row arithmetic
completion is unknown, not zero. A real2-group producer can expose an earlier
gathered group, but the median producer cost is **{summary['headroom']['producer_split_ratio_median']:.3f}×** native.
This bounded experiment does not establish positive net headroom, so no
overlap seam or full scheduling runtime was promoted.

The classification is an interface/measurement feasibility limit, **not** a
claim that all Attention–MoE pipelines are impossible. Engineering a new kernel
publication interface/compiler would exceed this contract's bounded scope.

## 1. Scope, environment, and prior

Model `inclusionAI/LLaDA2.0-mini`, revision `{REVISION}`; BF16;20 decoder layers,
19 sparse layers;hidden2048;256 routed experts/layer;top8;one shared expert;
native routed scaling2.5. Actual loaded64 experts/rank with19 independently
sharded layer tensors is checked in `raw/*/ownership.rank*.json`.

Physical GPUs2,3,6,7 map to logical0,1,2,3 in that order. H10080GB HBM3,
all-pair NV18. DenseTP4 / source-SP4 / physicalEP4 / DP1 / PP1, actual DeepEP
normal dispatch/expert/combine and NCCL source gather. Actual runtime versions
are in ownership records; no package or external dirty-source modification.

Authoritative spec: `references/authoritative_spec.md`, exact SHA in environment.
Pinned earlier report/bridge at `dc539d3222193ab0902e0a342679edc3610f54bb` reused;
worker setup is reused by an inspectable AST prefix, not reimplemented.

Prior audit is [prior_overlap.md](prior_overlap.md). TokenWeave already performs
B1 sequence splitting; MPK already schedules cross-operator dependencies;
Epoch restores a full logical shard after fresh sparse work. None was executed
as a baseline. **EPOCH_RUNTIME_NOT_RUN**. Earlier5.773× is not Epoch performance;
earlier18% is not this pipeline's speedup or headroom.

## 2. Frozen workload and actual execution

Metadata-only seed10062026 selected32 fresh public task IDs:8 calibration and
24 held-out, balanced coding/reasoning, excluding previously used amplification
and residency task IDs. Only predeclared4calibration/8held-out early-kill requests
were executed. Remaining20 are untouched; no expansion after weak feasibility.

B1, block32, threshold0.9, temperature0, EOS-enabled native64-token bounded
generation. Prompt lengths vary naturally; no artificial prefix padding was
presented as task gains. No generated code was executed. These truncated
prefixes are **not GSM8K accuracy or HumanEval pass@1 evidence**.
`gen_length=64` is the unchanged native requested budget: dInfer rounds the
total prompt+generation length to block32, so returned suffix lengths can
exceed64. Actual per-request counts are in request_metrics.csv; no output
truncation or decoding-rule change was added to force a strict64-token limit.

Three warmups, two smoke prompts (short/cold and nonempty prefix), two clean
self-repeats plus event/trace lanes.12 native requests agree in output token
IDs and NFE; event/trace acceptance schedules agree across ranks and observers.
Clean repeats do not log acceptance positions, so full clean acceptance parity
is not claimed. Candidate generation/accuracy: **NOT_RUN**.

Append-only run ledger: main_v1 retained all native request captures, then an
unsupported CUDA-graph probe stalled in count coordination and was aborted;
components_v2 exposed an unhandled native `attention_mask=None`, fixed by
preserving the native unmasked semantics rather than introducing a new mask;
components_v3 completed the bounded component probes; route_sanity_v4 adds a
current physical route/owner sanity trace. Failed logs are retained, never reset.
The execution ledger records96 native `generate` calls across the four runs,
including warmups/smokes/retries; component repetitions are listed separately.
`integrity_audit.json` checks all four ranks, the actual per-expert6MiB BF16
weight size, source route IDs/weights, owner-local receive IDs, and lossless raw
archive hashes. `handoff_environment.json` records preservation of the original
dirty/staged worktree and external runtime plus final assigned-GPU state.

## 3. Dependency and readiness findings

[dependency_audit.md](dependency_audit.md) maps each layout, collective, stream,
residual and cache dependency. Source local publication is after routed/shared
sum; head-shard consumer visibility is only after the whole source gather.
No unfinished buffer is read; all diagnostic reads are after same-stream events
and final synchronization outside the measured interval.

| Readiness | Result | Interpretation |
|---|---|---|
| latent per-row producer arithmetic | UNKNOWN | Full grouped kernel is not a token completion trace |
| natural safe group-publication window |0µs at exposed whole-buffer interface | Not a latent-skew/noise-floor assertion |
| natural TP-visible window |0µs at exposed interface | Both groups published together |
| engineered gathered-group gap |median{summary['headroom']['engineered_same_device_tp_visible_window_us']['median']/1000:.3f}ms; range{summary['headroom']['engineered_same_device_tp_visible_window_us']['min']/1000:.3f}–{summary['headroom']['engineered_same_device_tp_visible_window_us']['max']/1000:.3f}ms | Same-device physical2-group producer, one calibration snapshot; not full hidden consumer seam |

GPU absolute timestamps are not subtracted across ranks. Global alignment and
latent rank skew remain unresolved. A component critical rank is selected by
cumulative paired cost and used for all components; unrelated maxima are never
summed. Natural identical release is established by interface ordering, not
fabricated sub-token timestamps. `readiness.csv` leaves unobservable values null.
The original collector recorded model-call ordinal separately from native
acceptance iterations; calls are not assumed1:1 refinements. Per-node native
block/refinement/cache version therefore remain null/unresolved, and are not
invented from per-layer timing. Future reproduction logs direct iteration
context, but does not retrofit that missing field into captured evidence.

![Native timeline](figures/main/native_publication_timeline.png)

## 4. Baseline credibility and split tax

Native fused norms, BF16 Triton grouped experts and all reductions retained.
Installed ServerArgs explicitly disables graph with `deepep_mode=normal`;
direct capture was bounded and failed. Current SGLang TBO/SBO documentation is
not assumed to describe flags supported by this manual0.5.3 runner.

A shared-expert dual-stream functional adaptation was executed separately and
returned identical sampled producer outputs. It is a generic baseline test,
not a new pipeline gain; timing variability/no consistent winner prevent a
strong full-runtime B1 optimization claim. Its raw five-repeat costs are saved.

| Calibration boundary | Unsplit producer ms |2-group serial ms | Ratio |
|---|---:|---:|---:|
'''
    for l,u,s in zip(layers,values_u,values_s):
        text+=f'|{l}→{l+1}|{u:.3f}|{s:.3f}|{s/u:.3f}×|\n'
    text+=f'''
All native router/dispatch/expert/combine completion, shared/add, source gather,
group construction and scatter costs are included. Source ownership is
preserved: each original8-row local shard is split4+4, not repartitioned as a
naive contiguous global16-row split. Split MoE sampled outputs are exactly equal
to unsplit. No expert path or precision is changed.

![Producer tax](figures/main/producer_split_tax.png)

## 5. Legal prefix work and numerical reference

After consumer-visible complete rows, residual/norm, QKV, QK norm, RoPE and
attention over immutable earlier-cache state are token-local. All fresh current
block K/V, **including accepted current positions**, remain a mandatory join.
Cold/refresh or unproved-cache states use serial fallback.

The selected native hot calibration snapshot has64 logical prior tokens,
96 prior physical cache slots, including32 zero-padding slots; native mask is
None. Those padding slots are **not extra real prefix tokens**. They are retained
in the exact native reference; the component probe does not silently remove them
or claim validation of a different mask/cache policy. Source cache behavior and
the explicit compatibility limitation are recorded in cache_mask_audit.json.

Reference implementation uses FP32 max/sum/weighted-sum softmax states and exact
global normalization across prefix/fresh current partitions. Independent FP32
maximum error is {correctness['independent_fp32_merge_max_abs']:.8g}. Relative-L2 versus
native BF16 is small, but late-layer17 exceeds the **pre-frozen absolute0.03125
tolerance** on all4 ranks (maximum0.066885). Tolerance was not relaxed, and
closed-loop candidate correctness is not claimed. These are component reference
checks, not a failed candidate-generation/accuracy result.

![Component work](figures/main/consumer_component_costs.png)

## 6. Envelope, timing validity, and stage gates

Natural published-interface dependency-only whole-request envelope is0%:
no early TP-visible group exists in that graph. It is **not** an envelope for a
modified fine-grained kernel. A measured native-stream DAG includes each real
operator, shared/layout barrier and intervening gap exactly once, reconstructed
from one cumulative rank; calibration error is in dependency_reconstruction.csv.

Engineered whole-request dependency envelope/schedule estimate remain
**UNRESOLVED**, rather than extrapolating one component case to all layers and
requests or deleting the split/layout cost. A real2-group gap exists, but its
large producer tax supplies no established profitable seam premise.

Held-out event observer overhead: mean{ci['mean']:.2f}%, request-bootstrap95%CI
[{ci['ci95'][0]:.2f},{ci['ci95'][1]:.2f}]%, above the3% target. Detailed fine-grained
attribution is **UNVALIDATED**. Clean native E2E remains separately saved;
there is no candidate speedup result. Repeats are aggregated by request, not
layers/iterations. Component repetitions are not held-out sample count.

| Arm | Status |
|---|---|
| native unsplit producer |measured physicalEP4 |
|2-group producer split-serial |measured physicalEP4, not entire seam |
| consumer norm/QKV/partial/merge |component-only |
| fixed-group generic full seam |NOT_RUN |
| readiness-aware prefix full seam |NOT_RUN |
| QKV-only full seam |NOT_RUN |
| full-runtime candidate/E2E/accuracy |NOT_RUN |
| Epoch/MPK/TokenWeave runtime |NOT_RUN |

![Envelope scope](figures/main/headroom_scope.png)

## 7. Answers and next action

This is **engineered static group completion**, not a measured native token
straggler effect. Consumer early readiness disappears at whole-buffer combine,
then shared/add and source all-gather. Top8 dependencies are never omitted.
Valid prior-cache work exists, but current bidirectional K/V must stay fresh.
Partial normalization is mathematically exact; native BF16 tolerance is not
fully met at the sampled late boundary. Physical contention and full seam
makespan were not executed because interface, observer validity, split tax and
numeric promotion evidence do not justify them. No superiority over fixed-group
overlap or additive improvement beyond actual Epoch was demonstrated.

No global engine, compiler, scheduler, stale-live reuse, expert replication or
next-refinement speculation was implemented. The smallest unresolved experiment
is a supported safe group-publication/source-layout primitive under a strong
generic fixed pipeline, with clean paired timing, **before** testing any extra
prefix-specific value. Do not invest in a full runtime on this evidence.

GPU policy: assigned devices were initiallyOFF, no burner existed. Contract§6
requires restoring only prior state and forbids a new background service, so
the initialOFF state is restored; no other GPU is touched.

## Final handoff

| Field | Value |
|---|---|
|VERDICT|CONDITIONAL_INTERFACE_BLOCKED|
|NATURAL_READY_WINDOW|0µs exposed; latent mathematical readiness UNKNOWN|
|NEXT_LAYER_VISIBLE_WINDOW|0µs native; engineered local gap1.305ms, not a seam|
|LEGAL_PREFIX_WORK|norm/QKV/RoPE + immutable native-cache partial; fresh current-block join preserved|
|DEPENDENCY_ONLY_E2E_ENVELOPE|0% existing publication graph; engineered whole-request UNRESOLVED|
|MEASURED_SEAM_GAIN_VS_UNSPLIT|NOT_RUN|
|MEASURED_SEAM_GAIN_VS_FIXED|NOT_RUN|
|MEASURED_FULL_E2E_GAIN_OR_NOT_RUN|NOT_RUN|
|CORRECTNESS / TASK_QUALITY_SCOPE|native12-request token/NFE parity; event/trace acceptance parity; sampled MoE exact; late BF16 reference tolerance unresolved; task scores NOT_RUN|
|EPOCH_BASELINE_STATUS|EPOCH_RUNTIME_NOT_RUN; old5.773× not measured Epoch|
|PRIOR_OVERLAP_STATUS|substantial generic overlap; no novelty/superiority claim|
|GPU_MASK|2,3,6,7|
|BRANCH|flashvep/dllm-attn-moe-pipeline-poc|
|COMMIT|see immutable Git commit containing this report (no self-referential SHA)|
|REPORT_PATH|poc_dllm_attn_moe_pipeline/report.md|
'''
    (ROOT/'report.md').write_text(text)
    print('REPORT_AND_FIGURES_COMPLETE')


if __name__=='__main__':
    main()
