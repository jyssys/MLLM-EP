"""Evidence-only reports; preserve failed attempts and unsupported quantities."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from poc_dispatch_combine_asymmetry.contract import ROOT, write, load
from poc_dispatch_combine_asymmetry.scripts.analyze import ci, native


def report(name, body):
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    (folder/name).write_text(body.rstrip()+'\n')


def main():
    data=json.loads((ROOT/'artifacts/final_metrics.json').read_text());m=data['metrics']
    accounting=load(ROOT/'artifacts/communication_accounting.json')['records']
    replay=json.loads((ROOT/'artifacts/replay_summary.json').read_text())
    overhead=json.loads((ROOT/'artifacts/overhead.json').read_text())
    audit=json.loads((ROOT/'artifacts/gpu_audit/environment.json').read_text())
    classes=json.loads((ROOT/'artifacts/nsight/kernel_classification.json').read_text())
    gpu_overhead={}
    path=ROOT/'artifacts/runs/mllm_overhead_v5/envelope.rank0.jsonl'
    pairs=collections.defaultdict(dict)
    for r in map(json.loads,path.read_text().splitlines()):
        if r['repeat']>=0:pairs[r['arm'],r['repeat']][r['lane']]=r['model_forward_ms']
    for arm in ('visual','text'):
        subset=[v for (a,_),v in pairs.items() if a==arm]
        assert len(subset)==10 and all(set(v)=={'bypass','clean'} for v in subset)
        gpu_overhead[arm]=ci([100*(v['clean']/v['bypass']-1) for v in subset])
    write(ROOT/'artifacts/model_envelope_overhead.json',{'metrics':gpu_overhead,'scope':'active source TP0 language-model envelope; both lanes have identical two-event/completion guard; includes trace drain overhead'})
    # First-layer contribution using the same rank as each headline repeat.
    layer0={}
    for model,arm,key in (('mllm','visual','visual'),('mllm','text','text'),('llm','text','llm_text')):
        rows=[]
        selected=[r for r in data['summaries'] if r['model']==model and r['arm']==arm and r['size']=='high' and r['lane']=='clean' and not r.get('overhead_pair')]
        for s in selected:
            events=[]
            for f in Path(s['run']).glob('stage.pid*.jsonl'):
                for r in map(json.loads,f.read_text().splitlines()):
                    if r['wave']==s['wave'] and r['ep_rank']==s['critical_rank']:events.append(r)
            zero=next(r for r in events if r['layer']==0)
            rows.append({'repeat':s['repeat'],'rank':s['critical_rank'],'first_dispatch_ms':zero['dispatch_ms'],
                         'remaining_layer_mean_dispatch_ms':np.mean([r['dispatch_ms'] for r in events if r['layer']!=0]),
                         'remaining_layer_mean_combine_ms':np.mean([r['combine_ms'] for r in events if r['layer']!=0]),
                         'remaining_layer_ratio':sum(r['dispatch_ms'] for r in events if r['layer']!=0)/sum(r['combine_ms'] for r in events if r['layer']!=0),
                         'first_layer_share_percent':100*zero['dispatch_ms']/(s['dispatch_ms']*48)})
        layer0[key]={'records':rows,**{k:ci([r[k] for r in rows]) for k in ('first_dispatch_ms','remaining_layer_mean_dispatch_ms','remaining_layer_mean_combine_ms','remaining_layer_ratio','first_layer_share_percent')}}
    write(ROOT/'artifacts/first_layer_readiness.json',layer0)
    byte_summaries={}
    for key in ('mllm_visual','mllm_text','llm_text'):
        for size in ('low','medium','high'):
            rows=[r for r in accounting if f"{r['model']}_{r['arm']}"==key and r['size']==size]
            byte_summaries[key+'_'+size]={'mean_logical_dispatch_bytes_layer':np.mean([r['logical_dispatch_bytes'] for r in rows]),
                'total_logical_dispatch_bytes_48_layers':sum(r['logical_dispatch_bytes'] for r in rows),
                'mean_max_rank_receive_bytes_layer':np.mean([r['max_rank_receive_bytes'] for r in rows]),
                'mean_rank_receive_max_mean':np.mean([r['rank_receive_max_mean'] for r in rows]),
                'runtime_owner_map_verified_layers':sum(len(r['runtime_expert_owner_maps'])==4 for r in rows),
                'actual_source_rows_layer0':[v['token_rows'] for v in next(r for r in rows if r['layer']==0)['executed_or_padded']]}
            for component in ('dispatch_ms','combine_ms'):
                byte_summaries[key+'_'+size][component.replace('_ms','_apparent_throughput_GBs')]=byte_summaries[key+'_'+size]['mean_logical_dispatch_bytes_layer']/m[key+'_'+size+'_clean'][component]['mean']/1e6
    write(ROOT/'artifacts/communication_summary.json',byte_summaries)
    v=m['mllm_visual_high_clean'];t=m['mllm_text_high_clean'];l=m['llm_text_high_clean']
    contrast=data['visual_minus_text']['high']
    visual_vs_llm=100*(v['dispatch_ms']['mean']/l['dispatch_ms']['mean']-1)
    verdict={'primary_verdict':'DISPATCH_CRITICAL_RANK_TAIL_DOMINANT',
        'secondary_verdict':'GENERIC_EP_DISPATCH_COMBINE_ASYMMETRY',
        'secondary_scope':'common remaining-layer base asymmetry; all-layer visual ratio includes additional first-layer readiness excess and is NOT identical to controls',
        'visual_runtime_effect':'VISUAL_PRODUCER_READINESS_EXCESS_OVER_SAME_MODEL_TEXT',
        'MLLM_VISUAL_DISPATCH_EXCESS_STRONG':False,
        'intrinsic_visual_transport_excess':'NOT_SUPPORTED_BY_ROUTE_CONTROLLED_REPLAY',
        'dominant_scope':'visual increment: evidence favors upstream producer readiness/rendezvous at first MoE layer; generic base: control/count/rendezvous and host/layout protocol composite, not isolated wire movement',
        'attribution_label':'INFERRED_RESIDUAL supported by source, clean layer trajectory and ready-barrier replay; exclusive fractions remain UNRESOLVED',
        'visual_minus_same_model_text_percent':contrast['dispatch_percent'],
        'visual_minus_standalone_llm_percent_unpaired_model_blocks':visual_vs_llm,
        'replay_visual_minus_text':replay['metrics']['high']['visual_minus_text_dispatch_ms'],
        'pure_data_movement_boundary':'UNRESOLVED','EffectiveBW':'N/A',
        'optimization_implemented':False,'next_action':'Do not build an MLLM-specific transport method from this result. Investigate generic count/rendezvous launch protocol and vision-producer admission separately; no speedup claim.',
        'primary_repeats_each_arm_size':10,'request_variation':'one frozen request/prompt per arm/size; repeat CI does not establish image-population generalization',
        'cross_model_paired_timing':'NOT_RUN: standalone LLM collected in separate loading block, descriptive cross-model comparison only',
        'overhead':gpu_overhead,'branch':'flashvep/dispatch-combine-asymmetry-discovery'}
    write(ROOT/'artifacts/final_verdict.json',verdict)
    table='| Arm | tokens | Dispatch ms/layer | Expert | Combine | mean request D/C ratio |\n|---|---:|---:|---:|---:|---:|\n'
    for key,label in [('mllm_visual','VL visual'),('mllm_text','VL text'),('llm_text','Qwen3 LLM text')]:
        for size in ('low','medium','high'):
            r=m[key+'_'+size+'_clean']
            table+=f"| {label} {size} | {r['prompt_tokens']} | {r['dispatch_ms']['mean']:.4f} | {r['expert_ms']['mean']:.4f} | {r['combine_ms']['mean']:.4f} | {r['ratio']['mean']:.3f} |\n"
    report('environment.md',f'''# Environment and baseline readiness

Resolved authoritative spec: `{audit['resolved_spec_path']}`. The requested basename without `_gpu0145` does not exist; this unique, matching specification was read in full and snapshotted. No earlier spec was substituted.

Base git SHA: `{audit['git']['stdout'].strip()}`. Separate worktree and new branch; original staged/dirty changes preserved.
CUDA_VISIBLE_DEVICES=0,1,4,5; logical 0/1/2/3 map to physical 0/1/4/5.
H100 80GB HBM3 (81559 MiB each); allowed GPU pairs/all four are NV18 connected via NVSwitch.
TP groups=(0,1)/(2,3), DP peer groups=(0,2)/(1,3), EP group=(0,1,2,3), PP1. EP=TP×DP, four GPUs total.

MLLM revision `{audit['models']['mllm']['revision']}`; LLM revision `{audit['models']['llm']['revision']}`. Both actual configs: L48/d2048/m768/E128/K8, BF16, norm_topk_prob=true. All shards exist. HF storage layouts differ but actual loaded banks [32,1536,2048]/[32,2048,768] match; 9,437,184 BF16 bytes/expert. Actual expert maps verified on all 48 layers/all 4 ranks in each of 9 captured arm-size strata.
This matches MoE geometry, not weights or the entire attention/position architecture (VL MRoPE/DeepStack vs LLM RoPE).

vLLM {audit['vllm']}, DeepEP {audit['deep_ep']} (source SHA 73b6ea4a439ba03a695563f9fd242c8e4b02b37c), torch {audit['torch']}, CUDA {audit['cuda']}; NCCL2.28.9. DeepEP HT + sparse Triton experts, eager, EP linear32/rank. EPLB/redundant experts/DBO/prefix cache off. Chunked prefill off, max_batched_tokens=20000/max_seqs=1, KV2GiB/rank, fixed one-token output. No environment upgrade or benchmark-quality reevaluation.

One real request is active on DP0; DP1 is idle/dummy, not a duplicate request. Source TP partitions the request; SP padding plus one dummy row on each idle source are separately accounted. Foreign loads on GPUs2/3/6/7 and shared host/NVSwitch are recorded but untouched. Only authorized user's 0/1/4/5 burner was stopped; all timing GPU PIDs belong to this PoC. Resource restoration uses existing run_utilize.sh with explicit VLLM_UTILIZE_GPUS=0,1,4,5 (script default is DIFFERENT).
Raw UUID/topology/owner/version evidence: artifacts/gpu_audit/environment.json.

After all measurements, the user requested GPU handoff and revoked burner restoration. The verified owned burner tree was stopped; final allowed-GPU compute PID snapshot is empty. No subsequent GPU run or burner restart. See resource_handoff.json.
''')
    report('native_stage_breakdown.md',f'''# Fresh native TRUE EP4 stage baseline

No past GPU2/3/6/7 absolute timing was reused. Warmup3 per arm/size; final10 repeats per arm/size. VL visual/text alternate order within repeat; LLM is a separate model-loading block, so VL-vs-LLM is NOT a same-engine paired experiment.

{table}

Each repeat selects ONE rank maximizing its cumulative 48-layer Dispatch+Expert+Combine. All reported components come from that rank; no independent component maxima or unrelated rank averages. These are sum/mean of local prepare/expert/finalize intervals, NOT a globally barrier-composed end-to-end latency. 48 layers are not treated as independent samples. Mean, median, P90 and repeat-bootstrap95% CI (2000) in final_metrics.json.

High same-model visual−text Dispatch = {contrast['dispatch_ms']['mean']:.4f} ms/layer, CI {contrast['dispatch_ms']['ci95']}; paired percentage {contrast['dispatch_percent']['mean']:.2f}%, CI {contrast['dispatch_percent']['ci95']}.
Combine delta {contrast['combine_ms']['mean']:.4f} ms/layer, CI {contrast['combine_ms']['ci95']}.
Visual−LLM Dispatch = {visual_vs_llm:.2f}% (unpaired model blocks); it does NOT exceed the standalone LLM.

Token scaling is descriptive, one image/text prompt per bin, not a fitted latency predictor. Actual text counts174/724/2033 versus visual175/721/2029 are explicitly retained.
''')
    report('instrumentation_audit.md',f'''# Instrumentation validity

Clean lane: pooled local CUDA events for router completion and prepare/expert/finalize boundaries. No route clone/download, no Nsight, no per-layer global host sync. One completion drain after the 48-layer model forward. Same backend/placement/dtype/boundaries in all arms. Final stage intervals include native DeepEP's stock count synchronization; they are not wire-only timers.

Initial high_baseline_v1 incorrectly captured32 external-DP idle/dummy forwards per wave under a common label. Fail-closed analyzer rejected1536 layers/rank rather than48. This run is EXCLUDED, not averaged or silently cleaned. Added CPU DP-coordinate selection: max(coordinated_dp_tokens)==actual prompt_tokens, regression test and new v2/v3 runs. Each accepted wave has exactly48 unique layers on every rank. No additional GPU reduction or .item() was added.

Paired hook-bypass host-wave diagnostic n5 is noisy and includes DP idle coordination; visual CI was too wide. A separate n10 common two-event language-model envelope on active source TP0 was therefore executed. Same envelope and completion guard apply to both lanes. Inclusive clean overhead: visual {gpu_overhead['visual']['mean']:.2f}% CI {gpu_overhead['visual']['ci95']}; text {gpu_overhead['text']['mean']:.2f}% CI {gpu_overhead['text']['ci95']}. This is measured overhead, not zero; compare effect sizes accordingly. LLM host-wave overhead n5 {overhead['paired_hook_bypass_host_wave']['llm_controls_v3_text']['mean']:.2f}%.

Internal API CUDA events + NVTX + compact capture perturb component times; >10% changed strata are UNVALIDATED_PERTURBED_INTERNAL_DURATION. Their phase durations and the subcomponent figure are DIAGNOSTIC ONLY, never the headline. Separate minimal-hook Nsight lane classified exactly48 calls for five kernel classes on each of12 native arm/rank ranges. Nsight profiler changes readiness/launch timing; kernel durations cannot be substituted into clean times or summed across ranks.

No pure cross-rank data-movement interval exists: fused dispatch contains packing/movement/polling, fused combine contains movement/reduction/polling. EffectiveBW=N/A. ApparentThroughput uses logical useful bytes divided by clean composite stage time and includes all-rank useful traffic (local destinations included), not per-link bandwidth.
''')
    report('dispatch_internal_decomposition.md',f'''# Dispatch: what is and is not separated

Installed source: vLLM prepare_finalize/deepep_ht.py + modular_kernel.py; DeepEP buffer.py + csrc/deep_ep.cpp + csrc/kernels/intranode.cu at73b6ea4.

| Requested boundary | Evidence | Limitation |
|---|---|---|
| A control/count/layout | get_dispatch_layout API DIRECTLY_TIMED (perturbed diagnostic); get_dispatch_layout and notify_dispatch KERNEL_CLASSIFIED | notify includes multi-rank barriers, count exchange and channel prefixes |
| B packing/permutation | KERNEL_CLASSIFIED intranode::dispatch | fused with C/D; no independent ms |
| C actual cross-rank movement | same fused dispatch kernel | pure movement UNRESOLVED; no true bandwidth |
| D receive/tail/readiness | notify_dispatch barriers + dispatch polling, KERNEL_CLASSIFIED; native first-layer excess INFERRED_RESIDUAL attribution supported by ready-barrier replay | network rendezvous vs upstream producer not directly split |
| E expert-ready epilogue | native receiver waits event, remaps IDs, constructs CPU-count metadata | composite prepare residual INFERRED_RESIDUAL, no fabricated exclusive duration |

Fresh count path has num_worst_tokens=0: notify_dispatch populates mapped host counters; C++ spins for received token/expert counts, then allocates receive buffers and launches payload kernel. This count/allocation handshake is absent from reverse combine's cached handle. Layout CPU/GPU launch gaps are part of native prepare, not HBM expert reads.

On the SAME clean cumulative-critical rank, high visual first-layer Dispatch {layer0['visual']['first_dispatch_ms']['mean']:.3f}ms versus text {layer0['text']['first_dispatch_ms']['mean']:.3f}ms. It comprises {layer0['visual']['first_layer_share_percent']['mean']:.1f}% of visual's48-layer Dispatch sum. Remaining47 visual layers average {layer0['visual']['remaining_layer_mean_dispatch_ms']['mean']:.4f}ms, text {layer0['text']['remaining_layer_mean_dispatch_ms']['mean']:.4f}ms.
Thus the visual increment is chiefly an upstream producer/rank-readiness encounter at the first MoE, not48 uniformly slower data transfers. All-producer-ready route replay removes that excess. Generic count/barrier/host-launch composite remains asymmetric. Nsight shows count/barrier kernels much longer than layout kernels and fused payload kernels on waiting ranks; classifier emits own-rank distributions and first/max times. Nsight starts/stops perturb readiness, so its exact first-layer duration is NOT asserted to reproduce the clean29ms.

Remaining47-layer mean request D/C ratios: visual {layer0['visual']['remaining_layer_ratio']['mean']:.3f}, same-model text {layer0['text']['remaining_layer_ratio']['mean']:.3f}, standalone LLM {layer0['llm_text']['remaining_layer_ratio']['mean']:.3f}. This is the scope of the generic-asymmetry secondary verdict: the complete48-layer visual ratio is NOT asserted identical to both controls.

Expert weight reads belong to expert execution, not Dispatch. No exclusive A+B+C+D+E numeric partition is claimed where kernels fuse these operations.
''')
    report('combine_internal_decomposition.md','''# Combine decomposition

Installed DeepEPHTPrepareAndFinalize._finalize uses the dispatch handle's rank/channel prefix matrices, source indices and send_head. TopKWeightAndReduce combines expert contributions locally first. Buffer.combine reuses the handle and does not recompute dispatch layout or perform fresh dynamic receive-count polling/allocation based on new routes.

| Boundary | Label | Status |
|---|---|---|
| A local weighted reduce/layout | native finalize composite DIRECTLY_TIMED | exclusive preparation not separately timed |
| B reverse payload movement | intranode::combine KERNEL_CLASSIFIED | fused with reduce/scatter/polling; pure wire N/A |
| C source-row reduction/scatter | same combine kernel + output.copy_ | fused/unresolved exclusive duration |
| D barriers/epilogue | cached_notify_combine KERNEL_CLASSIFIED; final event wait/output copy composite residual INFERRED_RESIDUAL | source attribution only, not fabricated independent ms |

buffer.combine API interval is DIRECTLY_TIMED in the perturbed diagnostic lane only. Reverse movement carries one source contribution per token/destination after local expert reduction, not8 independent full hidden rows. Useful logical D/C hidden bytes are equal under destination dedup; smaller Combine time cannot be explained merely by fewer useful hidden bytes. Cached metadata/less fresh count coordination is a source-backed candidate, not a counterfactual measured speedup.
''')
    report('mllm_vs_llm_control.md',f'''# Same-model modality and standalone LLM control

{table}

Same VL model: high visual Dispatch rises {contrast['dispatch_percent']['mean']:.1f}% with paired CI excluding zero. Combine does not materially rise. Standalone LLM has large generic Dispatch/Combine asymmetry too, and its Dispatch is GREATER than VL visual in these native blocks. Therefore MLLM_VISUAL_DISPATCH_EXCESS_STRONG criteria fail.

Both models use SP TP partitioning (not two full input copies) and exactly the same loaded MoE geometry and backend. Actual route IDs/weights vary because checkpoints/contexts differ. Actual source layout and padding/dummy rows are exported; all4 expert owner maps verified for every captured layer. The models' non-MoE producer, positional and attention paths need not have identical readiness; no architecture-causal conclusion is drawn from unpaired absolute differences.

Volume alone does not explain native timing: token counts change nearly12x, but large control/readiness floors persist. Useful traffic and skew are tabulated separately. Visual/text traffic differences are modest relative to native timing excess; token-and-source-matched replay tests bare route transport separately.
''')
    report('route_controlled_replay.md',f'''# ROUTE_CONTROLLED_TRUE_EP_REPLAY

Actual captured BF16 inputs and native route IDs/weights, selected layers0/24/47 only. Four physical ranks0,1,4,5; one world4 CUDA group plus CPU Gloo group; DeepEP HT default20SM, 1GiB NVLink buffer, RDMA0, QP1, stock EP4 dispatch/combine config. Token counts matched174/721/2029; original source TP logical sequence restored, trimmed, resharded, native SP-padding and idle-DP dummy rows retained. All rank producers are ready at an explicit barrier OUTSIDE CUDA timers.

No attention/router/expert model compute is executed in this payload replay; received hidden values serve as combine payload. It is not native model latency or a semantic-quality test. It measures actual physical DeepEP dispatch/layout/combine. Warmup3/repeat10, aggregate three selected layers within each repeat; one critical rank maximizing D+C, no layer-independent bootstrap.

High visual Dispatch {replay['metrics']['high']['visual']['dispatch_ms']['mean']:.4f}ms versus text {replay['metrics']['high']['text']['dispatch_ms']['mean']:.4f}ms; paired delta {replay['metrics']['high']['visual_minus_text_dispatch_ms']['mean']:.4f}ms, CI {replay['metrics']['high']['visual_minus_text_dispatch_ms']['ci95']}.
Combine visual {replay['metrics']['high']['visual']['combine_ms']['mean']:.4f}ms/text {replay['metrics']['high']['text']['combine_ms']['mean']:.4f}ms.
Native visual excess vanishes in this bounded replay, favoring producer/runtime readiness rather than intrinsic visual payload transport. This does not prove all48 layers/all images have identical transport.

Initial launcher used a nonexistent venv torchrun executable and exited before replay/model work; cleanup restored burner. Corrected entrypoint python -m torch.distributed.run ran successfully. No retries of a measured result or changes to acceptance thresholds.
''')
    report('final_summary.md',f'''# Dispatch–Combine asymmetry discovery — completed evidence

**Primary verdict: DISPATCH_CRITICAL_RANK_TAIL_DOMINANT.** Large asymmetry also exists in standalone MoE LLM (GENERIC_EP_DISPATCH_COMBINE_ASYMMETRY), but native visual adds a strong first-layer producer-readiness excess over same-model text. It is NOT supported as intrinsic visual EP transport excess: equal-token route replay removes the visual difference, and visual is not slower than standalone LLM.

Attribution is **INFERRED_RESIDUAL**, supported by the clean layer trajectory, installed source and producer-ready replay. Exclusive network-tail vs producer-wait fractions remain unresolved; the generic secondary verdict refers to the comparable remaining-layer base ratios, not identical all-layer visual/text ratios.

{table}

High same-model Dispatch delta={contrast['dispatch_ms']['mean']:.4f}ms/layer [{contrast['dispatch_ms']['ci95'][0]:.4f},{contrast['dispatch_ms']['ci95'][1]:.4f}]95%CI; paired gain/excess={contrast['dispatch_percent']['mean']:.1f}% [{contrast['dispatch_percent']['ci95'][0]:.1f},{contrast['dispatch_percent']['ci95'][1]:.1f}]. High replay difference={replay['metrics']['high']['visual_minus_text_dispatch_ms']['mean']:.4f}ms; CI spans zero. The repeat CI does not generalize beyond the single frozen input in each bin.

Pure wire bandwidth and exclusive packing/movement/sync subdivisions are unresolved. Logical accounting and ApparentThroughput are in communication_summary.json. Internal API times are perturbed and never headline; profiler kernel classes are diagnostic. Clean envelope overhead and confidence intervals are reported, not hidden. No optimization/speedup/quality claim.

## Required final block

```text
GPU_SET: 0,1,4,5
PHYSICAL_TO_LOGICAL_GPU_MAPPING: 0→0,1→1,4→2,5→3
GPU_MODEL: H10080GB HBM3
GPU_TOPOLOGY: allowed GPUs NV18/NVSwitch
MODEL_MLLM: Qwen3-VL-30B-A3B-Instruct @9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
MODEL_LLM: Qwen3-30B-A3B @ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
TOPOLOGY: TRUE TP2-DP2-EP4-PP1
DEEPEP_BACKEND: high-throughput + sparse Triton
DEEPEP_VERSION: {audit['deep_ep']}
VLLM_VERSION: {audit['vllm']}
MLLM_VISUAL_HIGH_TOKENS: 2029 total /1980 visual
MLLM_VISUAL_HIGH_DISPATCH_MS: {v['dispatch_ms']['mean']:.6f}
MLLM_VISUAL_HIGH_COMBINE_MS: {v['combine_ms']['mean']:.6f}
MLLM_VISUAL_HIGH_RATIO: {v['ratio']['mean']:.6f} (mean request ratio)
MLLM_TEXT_HIGH_TOKENS: {t['prompt_tokens']}
MLLM_TEXT_HIGH_DISPATCH_MS: {t['dispatch_ms']['mean']:.6f}
MLLM_TEXT_HIGH_COMBINE_MS: {t['combine_ms']['mean']:.6f}
MLLM_TEXT_HIGH_RATIO: {t['ratio']['mean']:.6f}
LLM_TEXT_HIGH_TOKENS: {l['prompt_tokens']}
LLM_TEXT_HIGH_DISPATCH_MS: {l['dispatch_ms']['mean']:.6f}
LLM_TEXT_HIGH_COMBINE_MS: {l['combine_ms']['mean']:.6f}
LLM_TEXT_HIGH_RATIO: {l['ratio']['mean']:.6f}
VISUAL_MINUS_MLLM_TEXT_DISPATCH_PCT: {contrast['dispatch_percent']['mean']:.6f} paired
VISUAL_MINUS_LLM_TEXT_DISPATCH_PCT: {visual_vs_llm:.6f} unpaired model-block means
MLLM_TEXT_MINUS_LLM_TEXT_DISPATCH_PCT: {100*(t['dispatch_ms']['mean']/l['dispatch_ms']['mean']-1):.6f} unpaired
DISPATCH_CONTROL_COUNT_MS: N/A exclusive; composite direct/API and classified kernels in artifacts
DISPATCH_PACK_LAYOUT_MS: N/A exclusive; fused kernel
DISPATCH_DATA_MOVEMENT_MS: N/A exclusive; fused movement/polling
DISPATCH_RECEIVE_TAIL_SYNC_MS: N/A exclusive; visual first-layer Dispatch {layer0['visual']['first_dispatch_ms']['mean']:.3f}ms is composite
DISPATCH_EPILOGUE_MS: N/A exclusive
DISPATCH_RESIDUAL_UNATTRIBUTED_MS: not forced into free fit constant
COMBINE_PREP_MS: N/A exclusive
COMBINE_DATA_MOVEMENT_MS: N/A exclusive
COMBINE_REDUCTION_SCATTER_MS: N/A exclusive
COMBINE_SYNC_EPILOGUE_MS: N/A exclusive
COMBINE_RESIDUAL_UNATTRIBUTED_MS: no fabricated exclusive subdivision
DISPATCH_APPARENT_THROUGHPUT_GBS: {byte_summaries['mllm_visual_high']['dispatch_apparent_throughput_GBs']:.3f}, all-rank logical useful bytes/composite critical-rank duration, NOT true bandwidth
DISPATCH_EFFECTIVE_BW_GBS: N/A
COMBINE_EFFECTIVE_BW_GBS: N/A
ROUTE_CONTROLLED_REPLAY_STATUS: PASS, layers0/24/47 only
ROUTE_CONTROLLED_VISUAL_MINUS_TEXT_DISPATCH_MS: {replay['metrics']['high']['visual_minus_text_dispatch_ms']['mean']:.6f}, CI spans0
INSTRUMENTATION_OVERHEAD_STATUS: clean measured {gpu_overhead['visual']['mean']:.2f}%/{gpu_overhead['text']['mean']:.2f}%; internal composite durations UNVALIDATED where perturbed; Nsight diagnostic only
CROSS_MODEL_PAIRED_TIMING: NOT_RUN; separate model-block comparison only, unlike paired same-VL visual/text
GPU_FINAL_STATE: experiment and burner stopped; 0,1,4,5 verified empty at user-requested handoff; do not restart
PRIMARY_DOMINANT_COMPONENT: count/rendezvous/producer-readiness composite, NOT proven pure data movement
GENERIC_VS_MLLM_SPECIFIC: generic asymmetry plus visual native readiness excess; NO intrinsic visual transport promotion
OVERALL_VERDICT: DISPATCH_CRITICAL_RANK_TAIL_DOMINANT
NEXT_ACTION: {verdict['next_action']}
BRANCH: {verdict['branch']}
COMMIT: see artifacts/release.json evidence commit and repository branch HEAD
```
''')
    # Compact raw inputs sufficient to re-run headline aggregation. Do not
    # publish model weights, images, hidden dumps or large profiler databases.
    raw=ROOT/'artifacts/compact_raw';raw.mkdir(exist_ok=True)
    index=[]
    for run in (ROOT/'artifacts/runs').iterdir():
        if not run.is_dir():continue
        for f in [*run.glob('stage*.jsonl'),*run.glob('envelope*.jsonl'),*run.glob('worker*.json'),run/'manifest.json',run/'status.json']:
            if not f.exists():continue
            target=raw/(run.name+'__'+f.name+'.gz')
            content=f.read_bytes()
            with gzip.GzipFile(filename=str(target),mode='wb',mtime=0) as stream:stream.write(content)
            index.append({'source':str(f),'compact':str(target.relative_to(ROOT)),'sha256_uncompressed':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
    write(raw/'index.json',index)
    write(ROOT/'artifacts/exclusion_audit.json',{'high_baseline_v1':'EXCLUDED_IDLE_FORWARD_MIXING_1536_VS_48; retained raw',
        'replay_launcher_v1':'NOT_EXECUTED_MISSING_TORCHRUN_EXECUTABLE; use python -m torch.distributed.run',
        'internal_subcomponent_lane':'PERTURBED_DIAGNOSTIC_NOT_HEADLINE',
        'nsight_lane':'DIAGNOSTIC_ONLY_NOT_HEADLINE','unrelated_worktree_diff':'preserved'})


if __name__=='__main__':main()
