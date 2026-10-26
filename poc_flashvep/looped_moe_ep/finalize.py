"""Build inspectable study tables and reports from completed evidence only."""
from collections import defaultdict
import csv
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from .common import ROOT, REPO, MODELS, SPEC, digest, write
from .analyze import rows, table, completed, ci

LABEL = dict(ifm_ouro='IFM Ouro', ifm_huginn='IFM Huginn', foil1='Foil-1')

def read_csv(name):
    return list(csv.DictReader((ROOT/name).open()))

def md(headers, records):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in record)+' |' for record in records])

def save(name, prose):
    path=ROOT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prose.strip()+'\n')

def compact():
    baseline=read_csv('summary/baseline_forward.csv')
    quality=read_csv('summary/engineering_quality.csv')
    ready=read_csv('summary/ready_opportunity.csv')
    a=read_csv('oracle/coalescing.csv')
    ag=read_csv('oracle/generation_coalescing.csv')
    b=read_csv('oracle/replication.csv')
    output=[]; quality_rows=[]; memory=[]; comm=[]; opportunity=[]; sensitivity=[]
    for model in MODELS:
        for ep in [4,8]:
            def select(source):
                return [r for r in source if r['model']==model and int(r['ep'])==ep]
            cell=dict(model=model,ep=ep)
            run=ROOT/'raw'/f'{model}_ep{ep}_baseline_s0_v1'
            fidelity=json.loads((run/'fidelity.rank0.json').read_text())
            q=[r for r in select(quality) if r['run']==f'{model}_ep{ep}_shape_quality_v2']
            assert len(q)==105 and all(r['pass']=='True' for r in q)
            quality_rows.append(dict(cell,examples=len(q),domains_32_each=3,shape_probes=9,
                max_relative_l2=max(float(r['last_logit_relative_l2']) for r in q),
                max_abs_delta_nll=max(abs(float(r['delta_nll'])) for r in q),
                top1_equal=sum(int(r['top1_equal_count']) for r in q),positions=sum(int(r['positions']) for r in q),
                native_greedy_smoke_tokens=8,native_greedy_exact=fidelity['token_exact'],
                downstream_accuracy='NOT_RUN',evidence='LIVE_MATCHED_NATIVE_VS_TRUE_EP'))
            for batch in [1,2,4]:
                r=next(r for r in select(baseline) if int(r['batch'])==batch and int(r['prefix'])==128)
                cell[f'b{batch}_forward_ms']=float(r['mean_ms'])
                cell[f'b{batch}_ci95_lo']=float(r['ci95_lo'])
                cell[f'b{batch}_ci95_hi']=float(r['ci95_hi'])
            cases=select(ready)
            cell['ready_cross_calls']=sum(int(r['distinct_cross_pass_eligible_calls']) for r in cases)
            cell['ready_decisions']=sum(int(r['decisions']) for r in cases)
            cell['ready_max_call_fraction']=max(float(r['eligible_call_fraction']) for r in cases)
            for source, name in [(select(a),'prefill'),(select(ag),'generation')]:
                group=defaultdict(list)
                for r in source:
                    group[(r['batch'],r['arrival'],r.get('prefix',r.get('source_run')))].append(r)
                ratios=[min(float(r['generic_ms']) for r in v)/min(float(r['cross_ms']) for r in v) for v in group.values()]
                cell[f'a_{name}_strong_ratio_max']=max(ratios)
                for key,v in group.items():
                    sensitivity.append(dict(model=model,ep=ep,scope=name,batch=key[0],arrival=key[1],source=key[2],
                        best_generic_ms=min(float(r['generic_ms']) for r in v),
                        best_cross_ms=min(float(r['cross_ms']) for r in v),
                        strong_ratio=min(float(r['generic_ms']) for r in v)/min(float(r['cross_ms']) for r in v),
                        evidence='SIMULATED_BOUNDED_WINDOW_SEARCH_NOT_GLOBAL_OPTIMUM'))
            cell['a_same_window_ratio_max']=max(float(r['cross_vs_generic_ratio']) for r in select(a))
            by=defaultdict(dict)
            for r in select(b):by[(r['batch'],r['prefix'],r['budget'])][r['policy']]=r
            cell['b_vs_static_ratio_max']=max(float(v['loop_conditioned']['modeled_speedup']) for v in by.values())
            cell['b_vs_generic_ratio_max']=max(float(v['loop_conditioned']['modeled_speedup'])/
                float(v['generic_best']['modeled_speedup']) for v in by.values())
            cell['b_unresident_load_bound_max']=max(float(v['omniscient']['modeled_speedup']) for v in by.values())
            cell['b_conservative_ratio_max']=max(float(v['loop_conditioned']['conservative_speedup']) for v in by.values())
            cell['track_a']='NO_GO_A_TESTED_POLICY'
            cell['track_b']='NO_GO_B_TESTED_POLICY'
            cell['production_o2']='CONDITIONAL_UNVALIDATED'
            output.append(cell)
            for rank in range(ep):
                own=json.loads((run/f'ownership.rank{rank}.json').read_text())
                f=json.loads((run/f'fidelity.rank{rank}.json').read_text())
                param=sum(t['bytes'] for v in own for t in v['tensors'])
                peak=max(json.loads(p.read_text())['peak_allocated_bytes'] for p in run.glob(f'timing_b*_p*.rank{rank}.json'))
                memory.append(dict(model=model,ep=ep,rank=rank,
                    resident_after_shard_bytes=f['resident_after_sharding'],routed_weight_bytes=param,
                    peak_allocated_bytes=peak,kv_bytes=0,kv_scope='OFFICIAL_CACHE_FREE_FULL_PREFIX_PATH',
                    peak_scope='Clean forward plus separate observer capture. Not isolated clean-only peak.',
                    evidence='LIVE_PYTORCH_ALLOCATION_NOT_TOTAL_NVML_HBM'))
            for batch in [1,2,4]:
                for length in [128,1024,2048]:
                    traces=[json.loads((run/f'trace_b{batch}_p{length}.rank{rank}.json').read_text()) for rank in range(ep)]
                    invocations=len(traces[0])
                    assert invocations==(18 if model=='foil1' else 112)
                    dispatch_remote=sum(r['remote_bytes'] for trace in traces for r in trace)
                    width=1024 if model=='foil1' else 1536
                    logical_payload=batch*length*width*2
                    comm.append(dict(model=model,ep=ep,batch=batch,prefix=length,moe_calls=invocations,
                        dispatch_a2a_calls_per_rank=invocations,combine_a2a_calls_per_rank=invocations,
                        metadata_allgather_calls_per_rank=invocations,restoration_allreduce_calls_per_rank=invocations,
                        dispatch_remote_payload_bytes_all_ranks=dispatch_remote,
                        combine_remote_payload_bytes_all_ranks=dispatch_remote,
                        restoration_tensor_bytes_per_rank=invocations*logical_payload,
                        metadata_input_bytes_per_rank=sum(len(r['expert_counts'])*8 for r in traces[0]),
                        wire_bytes='NOT_MEASURED_PROTOCOL_AND_COLLECTIVE_DEPENDENT',
                        evidence='LIVE_CALLS_AND_STRUCTURAL_TENSOR_BYTES_NOT_LINK_COUNTERS'))
    # A sampled ready snapshot does not prove an exhaustive opportunity ceiling.
    for run in completed():
        f=run/'ready_queue.jsonl'
        if not f.exists():continue
        config=json.loads((run/'request.rank0.json').read_text())['config']
        case=defaultdict(lambda:dict(eligible={},all_counts={},shared_pairs=0))
        for event in rows(f):
            obj=case[(event['batch'],event['arrival'])]
            for work in event['ready']:
                key=(work['request'],work['site'],work['loop'],work['step'])
                obj['all_counts'][key]=sum(work['expert_counts'])
                matching=np.zeros(len(work['expert_counts']),dtype=int)
                for other in event['ready']:
                    if other['request']==work['request'] or other['site']!=work['site'] or other['loop']==work['loop']:continue
                    matching=np.maximum(matching,np.minimum(work['expert_counts'],other['expert_counts']))
                if matching.sum():obj['eligible'][key]=max(obj['eligible'].get(key,0),int(matching.sum()))
        for (batch,arrival),v in case.items():
            opportunity.append(dict(model=config['model'],ep=len(config['gpus']),batch=batch,arrival=arrival,
                sampled_shared_activation_rows=sum(v['eligible'].values()),
                sampled_ready_activation_rows=sum(v['all_counts'].values()),
                evidence='LIVE_SAMPLED_READY_VOLUME_COUNTS_NOT_COMPLETE_COALESCIBLE_VOLUME'))
    table('summary/model_ep_verdicts.csv',output)
    table('quality/engineering_fidelity.csv',quality_rows)
    table('summary/hbm.csv',memory)
    table('summary/communication_accounting.csv',comm)
    table('summary/ready_activation_volume.csv',opportunity)
    table('oracle/strong_baseline_sensitivity.csv',sensitivity)
    return output,quality_rows

def figures(cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    folder=ROOT/'figures'
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(folder/(name+'.svg'))
        fig.savefig(folder/(name+'.png'),dpi=160)
        plt.close(fig)
    names=[f'{LABEL[c["model"]]} EP{c["ep"]}' for c in cells]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,batch in enumerate([1,2,4]):
        x=np.arange(len(cells))+(j-1)*.24
        y=np.array([c[f'b{batch}_forward_ms'] for c in cells])
        err=np.array([[c[f'b{batch}_forward_ms']-c[f'b{batch}_ci95_lo'] for c in cells],
            [c[f'b{batch}_ci95_hi']-c[f'b{batch}_forward_ms'] for c in cells]])
        axes[0].bar(x,y,.24,label=f'B{batch}',yerr=err,capsize=2)
    axes[0].set_xticks(range(len(names)),names,rotation=35,ha='right')
    axes[0].set_ylabel('Complete forward (ms), prefix 128')
    axes[0].set_title('LIVE: 3 start/cohorts, 10 warmups + 20 trials/start')
    axes[0].legend()
    for j,name in enumerate(['a_prefill_strong_ratio_max','b_vs_generic_ratio_max']):
        axes[1].bar(np.arange(len(cells))+(j-.5)*.35,[c[name] for c in cells],.35,label=['A: best window','B: equal memory'][j])
    axes[1].axhline(1.15,color='red',linestyle='--',label='1.15 research gate')
    axes[1].set_xticks(range(len(names)),names,rotation=35,ha='right')
    axes[1].set_ylim(.98,1.17)
    axes[1].set_ylabel('Modeled ratio vs strong generic baseline')
    axes[1].set_title('MODEL: maxima across tested cohorts, not live speedup')
    axes[1].legend(fontsize=8)
    finish(fig,'baseline_and_headroom')
    load=read_csv('summary/rank_loads.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for ax,c in zip(axes.T.ravel(),cells):
        q=[r for r in load if r['model']==c['model'] and int(r['ep'])==c['ep'] and r['run'].endswith('s0_v1')
            and r['batch']=='1' and r['prefix']=='128']
        matrix=np.array([json.loads(r['rank_counts']) for r in q],float)
        matrix/=np.maximum(matrix.mean(1,keepdims=True),1)
        image=ax.imshow(matrix.T,aspect='auto',vmin=0,vmax=3,cmap='magma')
        ax.set_title(f'{LABEL[c["model"]]} EP{c["ep"]}, B1/P128')
        ax.set_xlabel('Logical MoE invocation')
        ax.set_ylabel('EP rank')
    fig.colorbar(image,ax=list(axes.ravel()),label='Actual row load / mean rank load',fraction=.02)
    fig.suptitle('LIVE router counts: one observer forward, not GPU time',y=1.01)
    # Avoid constrained colorbar layout warnings.
    fig.subplots_adjust(wspace=.35,hspace=.5,right=.89)
    fig.savefig(folder/'rank_load_heatmap.svg',bbox_inches='tight')
    fig.savefig(folder/'rank_load_heatmap.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    ready=read_csv('summary/ready_opportunity.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c in cells:
        q=[r for r in ready if r['model']==c['model'] and int(r['ep'])==c['ep'] and r['arrival']=='staggered']
        axes[0].plot([int(r['batch']) for r in q],[100*float(r['eligible_call_fraction']) for r in q],
            marker='o',label=f'{LABEL[c["model"]]} EP{c["ep"]}')
    axes[0].set_xlabel('Independent requests')
    axes[0].set_ylabel('Sampled cross-pass eligible calls (%)')
    axes[0].set_title('LIVE ready quorum, 8 generated tokens, observer only')
    axes[0].legend(fontsize=8)
    a=read_csv('oracle/coalescing.csv')
    for model,ep in [('ifm_huginn',4),('foil1',4),('foil1',8)]:
        q=[r for r in a if r['model']==model and int(r['ep'])==ep and r['batch']=='4' and r['prefix']=='128' and r['arrival']=='staggered']
        axes[1].plot([float(r['window_us'])/1000 for r in q],[float(r['cross_vs_generic_ratio']) for r in q],
            marker='.',label=f'{LABEL[model]} EP{ep}')
    axes[1].axhline(1,color='black',linestyle='--')
    axes[1].set_xlabel('Shared waiting window (ms)')
    axes[1].set_ylabel('Modeled ratio vs same-window generic')
    axes[1].set_title('SIMULATION: best-window generic removes apparent gain')
    axes[1].legend(fontsize=8)
    finish(fig,'readiness_and_wait_sensitivity')
    # Each rank has its own time origin. Rows do not imply inter-rank simultaneity.
    fig,axes=plt.subplots(1,3,figsize=(13,3.3))
    for ax,model in zip(axes,MODELS):
        run=ROOT/'raw'/f'{model}_ep8_baseline_s0_v1'
        for rank in range(8):
            trace=json.loads((run/f'trace_b1_p128.rank{rank}.json').read_text())
            for call in trace[:4]:
                x=call['ready_forward_ms']
                for name,color in [('metadata_pack','#999999'),('dispatch','#4477aa'),('receive_grouping','#bbbbbb'),
                    ('expert','#228833'),('combine','#cc6677'),('restore','#aa3377')]:
                    width=call[name+'_ms']
                    ax.broken_barh([(x,width)],(rank-.35,.7),facecolor=color)
                    x+=width
        ax.set_title(f'{LABEL[model]} EP8, first 4 calls')
        ax.set_xlabel('Rank-local elapsed GPU event time (ms)')
        ax.set_ylabel('EP rank')
    fig.suptitle('LIVE observer API spans. Independent rank origins. Not a synchronized global Gantt.',y=1.02)
    finish(fig,'ep8_rank_local_timeline')

def reports(cells,q):
    meta={m:json.loads((ROOT/'raw'/f'{m}_ep4_baseline_s0_v1'/'native_metadata.rank0.json').read_text()) for m in MODELS}
    baseline_table=md(['Model','EP','B1 ms [95% CI]','B2 ms','B4 ms'],[
        [LABEL[c['model']],c['ep'],f'{c["b1_forward_ms"]:.2f} [{c["b1_ci95_lo"]:.2f}, {c["b1_ci95_hi"]:.2f}]',
            f'{c["b2_forward_ms"]:.2f}',f'{c["b4_forward_ms"]:.2f}'] for c in cells])
    headroom_table=md(['Model','EP','A vs best generic','B vs static, max','B vs generic, max','B load-only ideal'],[
        [LABEL[c['model']],c['ep'],f'{c["a_prefill_strong_ratio_max"]:.3f}×',f'{c["b_vs_static_ratio_max"]:.3f}×',
            f'{c["b_vs_generic_ratio_max"]:.3f}×',f'{c["b_unresident_load_bound_max"]:.3f}×'] for c in cells])
    quality_table=md(['Model','EP','Checks passed','Max logit relative L2','Max |ΔNLL|','Top-1 equal / positions','Native greedy'],[
        [LABEL[r['model']],r['ep'],f'{r["examples"]}/105',r['max_relative_l2'],r['max_abs_delta_nll'],
            f'{r["top1_equal"]}/{r["positions"]}','8/8 exact'] for r in q])
    gen=read_csv('summary/steady_generation.csv')
    generation_table=[]
    for c in cells:
        v=[r for r in gen if r['model']==c['model'] and int(r['ep'])==c['ep'] and r['kind']=='WARMED_FULL_GENERATION']
        assert len(v)==3
        s=ci([float(r['e2e_ms']) for r in v])
        generation_table.append([LABEL[c['model']],c['ep'],f'{s["mean"]/1000:.3f} [{s["ci95_lo"]/1000:.3f}, {s["ci95_hi"]/1000:.3f}]',
            f'{np.mean([float(r["ttft_ms"]) for r in v]):.2f}',f'{np.mean([float(r["tpot_ms"]) for r in v]):.2f}',f'{128/(s["mean"]/1000):.2f}'])
    generation_table=md(['Model','EP','B1 E2E s [95% CI]','TTFT ms','TPOT ms','Output tokens/s'],generation_table)
    readiness=read_csv('summary/ready_opportunity.csv')
    ready_table=md(['Model','EP','B1 eligible calls','B2 eligible calls','B4 eligible / decisions','B4 exact output'],[
        [LABEL[c['model']],c['ep'],sum(int(r['distinct_cross_pass_eligible_calls']) for r in readiness if r['model']==c['model'] and int(r['ep'])==c['ep'] and r['batch']=='1'),
            sum(int(r['distinct_cross_pass_eligible_calls']) for r in readiness if r['model']==c['model'] and int(r['ep'])==c['ep'] and r['batch']=='2'),
            f'{sum(int(r["distinct_cross_pass_eligible_calls"]) for r in readiness if r["model"]==c["model"] and int(r["ep"])==c["ep"] and r["batch"]=="4")}/{sum(int(r["decisions"]) for r in readiness if r["model"]==c["model"] and int(r["ep"])==c["ep"] and r["batch"]=="4")}',
            '2/2 arrival cases exact'] for c in cells])
    components=read_csv('summary/components.csv')
    component_table=md(['Model','EP','Clean ms','Observer ms','Pack','Dispatch','Group','Expert','Combine','Restore','Residual ms'],[
        [LABEL[c['model']],c['ep']]+[f'{float(r[k]):.2f}' for k in ['clean_mean_ms','observer_ms','metadata_pack_ms','dispatch_ms','receive_grouping_ms','expert_ms','combine_ms','restore_ms']]+
        [f'{float(r["observer_ms"])-sum(float(r[k+"_ms"]) for k in ["metadata_pack","dispatch","receive_grouping","expert","combine","restore"]):.2f}']
        for c in cells for r in components if r['model']==c['model'] and int(r['ep'])==c['ep'] and r['run'].endswith('s0_v1') and r['batch']=='1' and r['prefix']=='128'])
    save('README.md',f'''# Looped MoE expert-parallel discovery

Verdict: **NO_GO_A / NO_GO_B for the tested designs. No architecture GO.**
Production generalization remains **CONDITIONAL**.
The experiment ran the three exact checkpoints on TRUE EP4 and EP8.
All six configurations passed native-versus-EP engineering fidelity.
The tested recurrence-aware policies did not reach the 1.15× research gate.
No optimized A/B dispatcher receives a measured speedup claim.

{headroom_table}

Every ratio above is a modeled estimate, not measured E2E acceleration.
The B maxima can occur in different cells.
They are not ratios of independently selected maxima.
Column B versus generic uses matched model, degree, batch, prefix, and replica budget.

Read [report.md](report.md), [go_no_go.md](go_no_go.md), and [reproduce.md](reproduce.md).
See [methodology.md](methodology.md) for the frozen fidelity and timing limits.
All baseline numbers come from actual distributed GPU execution.
The native generation path recomputes the entire prefix without a KV cache.
These results do not establish a production DeepEP limit.
''')
    save('report.md',f'''# Looped MoE EP discovery report

## Executive conclusion

The tested cross-loop coalescing and loop-conditioned replication policies do not justify a new architecture prototype.
TRUE EP4 and EP8 work on all three required checkpoints.
The distributed baseline preserves the native logits and observed greedy outputs.
The best-window coalescing model gives 1.000× versus its best generic control.
Loop-conditioned replication adds less than 1% over matched generic replication in every tested cell.
No model clears both the 1.15× realistic headroom gate and the 5% recurrence-specific increment.

This is a bounded negative result, not a proof against all recurrent EP designs.
The coalescing scheduler is a finite policy search, not a global optimum.
The replica model is optimistic and not an actual replica-aware dispatcher.
The production asynchronous replay gate remains CONDITIONAL.

## 1. Models, loaders, and actual hardware

The study uses eight NVIDIA H100 80GB HBM3 GPUs.
All GPU pairs report NV18 connectivity.
The recorded CPU affinity is 0–191 on NUMA node 0.
The experiment does not infer achieved bandwidth from that topology label.
`preflight.json` records physical IDs, UUIDs, PCI addresses, and initial process ownership.

EP4 uses physical GPUs 0–3 or 4–7.
EP8 uses physical GPUs 0–7.
Each process exposes one physical GPU as CUDA device 0.
Latency runs use one active EP group at a time.
Owned burners remain on unused devices, as requested.
Therefore the quiet lane excludes concurrent experiment groups, not all board or CPU power effects.

| Model | Learned parameters | Physical routed sites | Native logical calls | Recurrence-eligible sites |
| --- | --- | --- | --- | --- |
| IFM Ouro | 8,780,889,600 | 28 | 28 × 4 = 112 | 28 |
| IFM Huginn | 8,783,251,968 | 28 | 8 + 12 × 8 + 8 = 112 | 12 |
| Foil-1 | 553,730,048 | 3 | 1 + 16 + 1 = 18 | 1 |

IFM uses the official manifest-verified native loader and supported FA4 attention.
FA3 is unavailable in this environment.
Both sides of each EP fidelity comparison use identical FA4 kernels.
The prior FA4-versus-explicit-attention limitation remains unchanged.
Foil uses its official remote code and its Foil-1 checkpoint subfolder.
The experiment retains Foil's original FP32 RoPE buffers.

The specification's Foil count needs an accounting qualification.
The loaded checkpoint has 553,730,048 learned parameters.
It also has 9,437,184 registered FP32 buffer elements.
Their sum is 563,167,232 persistent elements, or approximately 563.2M.
The experiment does not invent parameters to reconcile the specification.

## 2. TRUE EP implementation

`runtime.py` replaces only the native routed-expert expression.
Attention, router values, normalization, residuals, recurrence, and output heads remain native.
Experts have stable owners: `expert_id % EP_degree`.
IFM's 25 experts use ragged owner counts, not padded logical experts.
Foil's untied per-pass attention never enters expert coalescing.
Huginn prelude and coda remain one-off model work.

Every MoE invocation performs actual NCCL communication:

1. Partition source rows by rank.
2. All-Gather exact routing counts.
3. All-to-All BF16 expert inputs.
4. Execute owner-local native expert GEMMs.
5. All-to-All expert outputs.
6. Apply original top-2 scores and restore source rows with All-Reduce.

Dense parameters remain replicated.
Expert parameters become owner-only storage before measurements.
Every load checks exact native weight subsets.
Every completed worker checks retained expert storage identities.
This is TRUE EP, but not optimized DeepEP production serving.
Its count synchronization and source restoration can dominate small work units.
Those generic bridge costs are not a new loop-specific contribution.

## 3. Measured latency

The following complete-forward values use prefix 128.
Each B1/B2/B4 cell has three independent process starts.
Each start uses ten warmups and twenty timed repetitions.
The 95% bootstrap interval resamples process-start means, not individual kernels.
Starts 0/1/2 use public prompt-cohort offsets 0/4/8.
The interval therefore combines restart and prompt-cohort variation.
It does not isolate GPU-group variance.
EP4 and EP8 use the same prompt IDs at each corresponding start.
Three starts give limited precision for hardware variability.
Prefixes 1024 and 2048 have one start and exploratory repetition intervals.

{baseline_table}

Units are milliseconds.
`summary/baseline_forward.csv` contains all 54 model/degree/batch/prefix cells.
The duration includes attention, experts, metadata, restoration, and completion synchronization.
It is not a sum of kernel service times.

### Warmed generation control

The following B1 measurements use prefix 128 and 128 actual greedy output steps.
EOS is disabled for this hardware workload.
The path recomputes the full prefix on every step.
These values are not cached-decode TPOT or natural-completion benchmark accuracy.
Each cell uses three warmed full generations after one full shape-preparation generation.
The interval resamples those three serial generations from one process start.

{generation_table}

The first large matrix also covers B1/B2/B4 at output caps 32 and 128.
Those values remain in `summary/generation.csv`.
Their first-occurrence shapes were not fully warmed.
They are exploratory measurements, not the steady-state headline.

### Component evidence

The following separate observer uses B1 and prefix 128.
All components come from one cumulative-critical rank per forward.
Columns contain CUDA-event API spans, including host launch gaps.
The selected recurrent capture site also clones diagnostic expert inputs.
Its expert span therefore includes that observer copy.
They are not kernel-only service times or a clean E2E percentage partition.

{component_table}

All values are milliseconds.
Residual includes non-expert execution, observer overhead, and unresolved gaps.
Rank selection and host overlap prevent an exact clean critical-path attribution.
CPU synchronization and rank waiting remain PARTIAL.
`summary/communication_accounting.csv` separates logical tensor bytes from unmeasured wire bytes.

## 4. Fidelity and quality

{quality_table}

Each cell has 32 language examples, 32 GSM8K examples, 32 MATH-500 examples, and nine hardware-shape probes.
The teacher-forced examples use a 128-token input cap.
The shape probes cover B1/B2/B4 and 128/1024/2048 tokens.
They are not 105 free-running mathematical answers.
The table does not report downstream answer accuracy.
No candidate A/B implementation exists, so candidate answer accuracy is NOT_RUN.

Huginn's official initializer depends on the batch index.
Native B4 therefore differs from four independent native B1 initializations.
Every native-versus-EP check preserves batch geometry.
The independent-request observer compares against sequential B1 references instead.
The study does not use B4-versus-four-B1 equality as a quality claim.

## 5. Track A: ready coalescing

The diagnostic observer runs independent requests through unchanged native attention and recurrence.
Only one coordinator issues distributed collectives.
CUDA events protect input readiness and output lifetime.
All ranks vote on device readiness before a recorded candidate becomes eligible.
This avoids comparing unsynchronized absolute CUDA clocks across GPUs.
The observer never coalesces expert work.

{ready_table}

Each entry combines synchronized and staggered arrivals over eight generated steps.
The denominator counts observed MoE scheduling decisions, not tokens or unique experts.
An eligible call shares at least one physical expert with another ready request at a different pass.
Observed eligibility is not the fraction of work that can be removed.
The control broadcasts and readiness votes perturb queue timing.
These snapshots are sampled evidence, not an exhaustive opportunity ceiling.

B1 has no independent cross-request frontier.
The tested B2 observer also finds no cross-pass eligible calls.
Foil B4 shows actual cross-pass opportunities.
IFM's longer repeated layer chains produce little aligned physical-site readiness in this scheduler.
The absence of observed pairs is specific to this workload and coordination policy.

The O1 model preserves each request's causal layer and recurrent dependencies.
It charges measured complete expert-service costs and explicit waiting windows.
It compares generic same-pass and cross-pass policies within one executor.
The tested windows are 0/50/100/250/500 microseconds and 1/2/5/10 milliseconds.

The same-window comparison can show apparent improvements up to {max(c['a_same_window_ratio_max'] for c in cells):.3f}×.
However, the best generic window is zero in the tested cases.
Every matched best-cross-versus-best-generic ratio is 1.000×.
Waiting creates some opportunities but costs more than their measured batching benefit.

The 32-step O2-style replay also gives 1.000× against its best generic window.
That replay clones a representative request template.
It uses separately measured cohort scaling and a causal decision-cost lookup.
Huginn's batch-indexed initializer further limits that cohort-scaling equivalence.
The replay is not a tight bound for arbitrary independent-request trajectories.
It does not constitute independently validated asynchronous serving.
`oracle/replay_validation.csv` records held-out start drift and in-sample fit separately.
`oracle/generation_validation.csv` records the remaining generation-model mismatch.
No modeled ratio receives an M-level measured speedup label.

## 6. Track B: replicated experts

The study calibrates static replica sets on eight examples per domain.
The remaining examples and hardware cohorts do not tune those sets.
Generic hot, generic cost-aware, and loop-conditioned searches receive the same observations and copy budgets.
All searches use the same bounded placement proxy.
Runtime assignment uses exact current demand and only resident copies.
No expert weight moves between passes.

{headroom_table}

These are optimistic whole-forward projections, not actual optimized E2E timings.
The projection maps measured expert spans linearly to maximum rank-row load.
Those API spans can include observer copies and launch gaps.
The projection is not a precise expert-kernel critical-path model.
Communication remains unchanged.
The conservative sensitivity halves the computed saving and charges selector CPU time.
The omniscient column ignores replica residency and is not implementable.
It is a load-only diagnostic, not an absolute bound on all communication optimizations.

The loop-conditioned increment over generic replication remains below 1%.
It falls below the 5% recurrence-specific threshold.
With the same resident set and exact current routing, pass identity adds no information to this selector.
The same-resident-set control therefore yields exactly the same integer assignment.
This is a property of the implemented selector, not proof about all future selectors.

Replica memory is not free.
The study allocates and hashes real native BF16 copies at one and EP-degree extra copies per recurrent site.
Those controls measure initialization time and retained per-rank memory.
They do not execute replica-aware routing.
`summary/replica_memory.csv` and `summary/hbm.csv` contain actual bytes.

## 7. C0: overlap and routing lookahead

The operator control overlaps one independent ready dispatch with one native-weight expert GEMM.
Sequential and two-stream runs use identical tensors.
The final quality controls verify exact expert outputs.
The earlier operator controls did not record that output check.
It measures twenty repetitions after ten warmups.
CUPTI timelines distinguish NCCL kernels from other kernels.
`summary/kernel_overlap.csv` reports union and intersection durations.
`summary/overlap_operator.csv` reports complete operator wall-time ratios.
Concurrent kernels alone do not establish request-level acceleration.
Request critical-path saving is NOT_MEASURED.

Previous-pass routing recall uses up to 32 sampled positions per call.
Precision equals recall because both selections contain top-2 experts.
The diagnostic never predispatches an unknown next-pass activation.
False positives can waste replica memory even when expert sets overlap.
No learned gate or predictor is introduced.

## 8. Research decision and limitations

P0 passes all six native/TRUE EP cells.
P1 passes bounded characterization, with sampled-ready and critical-path limitations.
P2 provides calibrated policy estimates, but production O2 validation remains CONDITIONAL.
P3 optimized dispatch is NOT_RUN because neither track clears the research gate.
The tested designs receive NO_GO_A and NO_GO_B.
The study makes no universal impossibility claim about recurrent EP.

The strongest practical next step is generic EP backend optimization, not a named loop architecture.
A future recurrent proposal must first show a ready independent frontier and a gain beyond that optimized generic executor.
It must also validate natural answer accuracy and full generated-request timing.
No skipping, output reuse, gate change, quantization, or loop pipeline was implemented.
''')
    save('go_no_go.md',f'''# Decision record

Overall implementation decision: **NO_GO_EP_DIRECTION for the tested A/B policy families.**
Production extrapolation: **CONDITIONAL**, not a universal negative result.

{md(['Model','EP','P0 TRUE EP','P1 characterization','P2 models','A','B','P3 method'],[[LABEL[c['model']],c['ep'],'PASS','PASS, bounded','CONDITIONAL O2','NO_GO_A','NO_GO_B','NOT_RUN'] for c in cells])}

{headroom_table}

All headroom ratios are modeled, not measured optimized E2E speedups.
Neither track reaches 1.15× against its strong matched generic control.
Replication also fails the 5% incremental recurrence criterion.
The evidence supports stopping these prototypes, not declaring all looped EP research impossible.

B1 cannot coalesce independent requests because only one request exists.
B2 shows no cross-pass ready pairs in this diagnostic.
B4 Foil shows pairs, but bounded waiting does not beat tuned generic batching.
Loop-conditioned placement changes some sets but adds less than 1% over generic sets.
Actual optimized A/B accuracy and E2E are NOT_RUN.
DeepEP, production paged-KV serving, and a globally optimal scheduling bound are NOT_RUN.
''')
    save('quality/correctness.md',f'''# Correctness and quality

{quality_table}

The frozen limits remain in `methodology.md`.
All six EP cells meet those limits without relaxation.
The native repeated-logit check is exact.
The owner tensor checks confirm exact checkpoint subsets.
The distributed execution count is 112 calls for either IFM model and 18 for Foil.
The ready observer also checks every request's physical-site call count.
Its eight-token independent outputs match sequential B1 references.
Four complete steady-generation repetitions also retain identical output token IDs.

These checks establish sampled runtime fidelity, not complete downstream accuracy.
GSM8K/MATH-500 teacher forcing is not answer scoring.
EOS-disabled hardware generation is not natural answer evaluation.
There is no candidate optimizer to qualify because the headroom gate failed.
Candidate A/B numerical tests and downstream answer accuracy are NOT_RUN.

The earlier single-GPU skipping/reuse NO-GO remains unchanged.
The earlier FA4-versus-explicit-attention limitation also remains unchanged.
No earlier artifact was rewritten.
''')
    save('timing.md',f'''# Timing validity

The clean forward encloses native model execution and CUDA completion.
A distributed barrier precedes the interval.
Rank-local elapsed durations do not require synchronized absolute GPU clocks.
The maximum elapsed rank duration defines a cohort observation.
Three independent starts cover prefix 128 at every B1/B2/B4 cell.
The prompt cohort also changes across starts, at offsets 0/4/8.
Intervals therefore combine process-start and prompt-cohort variation.
Long-prefix cells use one start with twenty serial repetitions.
The latter confidence intervals are exploratory.

Heavy CUDA-event and CUPTI traces run separately.
API spans include launch gaps and Python control work.
They cannot form a clean E2E percentage pie.
The report uses one cumulative-critical rank, not different maxima per component.
Rank waiting and CPU overlap remain PARTIAL.

Foil's first 32-token clean trace took 2976.160 ms in one EP4 start.
The subsequent instrumented trace took 1922.372 ms and preserved token IDs.
The observed ratio was 0.646, not a credible negative instrumentation overhead.
The diagnosis separated first-shape preparation, hardware state, and observer-path changes.
The warmed follow-up uses the same uninstrumented path four times.
`summary/shape_drift.csv` records first/last and last-two ratios.
The repeated-generation control identifies condition drift but does not prove a specific JIT kernel caused it.

The headline generation table uses only warmed repetitions.
The original first-occurrence matrix remains unchanged and explicitly qualified.
Every generation measurement uses full-prefix recomputation without cached KV.
Output caps are hardware conditions, not natural EOS completion.
''')
    save('failures.md', '''# Failures and untested conditions

No completed GPU cell failed native-versus-EP fidelity.
Failure logs remain in the archives if an attempt produced one.
One CPU watcher initially imported Torch from the analysis-only Python environment.
The watcher failed before GPU launch.
The fix delayed the GPU-only import until the worker phase.
The original watcher log remains preserved.

Foil's first-occurrence generation timing failed the observer-parity check.
The timing report preserves that failed check and the warmed follow-up.
It does not reinterpret the faster heavy trace as an optimization.

The generation replay is not independently validated asynchronous serving.
The ready observer adds control broadcasts and readiness votes.
Its sampled queue eligibility is scheduler-dependent.
The replica model assumes row-proportional expert span savings.
It does not model a new replica-aware collective implementation.
The same-window coalescing comparison can overstate incremental gain.
The best-window generic comparison removes that apparent gain here.

DeepEP production comparison: NOT_RUN, unavailable in the validated Torch 2.11 environment.
TP and hybrid baselines: NOT_RUN, optional and outside the thin EP bridge.
Optimized A/B dispatcher: NOT_RUN, headroom gate not met.
Candidate natural task accuracy: NOT_RUN, no candidate implementation.
NCCL wire traffic and hardware link counters: NOT_MEASURED.
Globally synchronized rank waiting and complete clean critical-path attribution: PARTIAL.
No approximation, smaller proxy checkpoint, or architecture substitution fills these gaps.
''')

def main():
    for name,n in [('baseline',18),('final_quality',6),('ready_probe',6),('steady_generation',6)]:
        value=json.loads((ROOT/f'campaign_{name}.json').read_text())
        assert len(value)==n and all(v['status'] in ('PASS','ALREADY_COMPLETE') for v in value),(name,value)
    cells,q=compact()
    figures(cells)
    reports(cells,q)
    matrix=[]
    for run in completed():
        config=json.loads((run/'request.rank0.json').read_text())['config']
        matrix.append(dict(run=run.name,model=config['model'],ep=len(config['gpus']),physical_gpus=str(config['gpus']),
            phase=config['phase'],status='PASS',batches=str(config.get('batches','PHASE_SPECIFIC')),
            prefixes=str(config.get('lengths','PHASE_SPECIFIC')),generation_caps=str(config.get('generation_lengths','PHASE_SPECIFIC')),
            evidence='LIVE_DISTRIBUTED_EXECUTION',config='configs/'+run.name+'.json'))
    table('experiment_matrix.csv',matrix)
    preflight=json.loads((ROOT/'preflight.json').read_text())
    write(ROOT/'manifest.json',dict(spec_sha256=digest(SPEC),baseline_git_head=preflight['git_head'].strip(),
        branch=preflight['branch'].strip(),models=preflight['models'],gpu_provenance='preflight.json',
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        completed_runs=len(completed()),generated_at_unix=time.time(),
        xllm_commit=subprocess.check_output(['git','-C','/home/esjung/external/xllm-loop-ep-poc','rev-parse','HEAD'],text=True).strip(),
        foil_code_commit=subprocess.check_output(['git','-C','/home/esjung/external/how-to-loop-moe-poc','rev-parse','HEAD'],text=True).strip(),
        easyinfra_audit_commit=subprocess.check_output(['git','-C','/home/esjung/external/EasyInfra-looped-ep-audit','rev-parse','HEAD'],text=True).strip(),
        initial_index_tree=preflight['preserved_index'].strip(),
        measured='TRUE_EP_BASELINE_AND_OPERATOR_CONTROLS',candidate_e2e='NOT_RUN',
        oracle_qualification='BOUNDED_ESTIMATES_NOT_MEASURED_SPEEDUP'))
    print('REPORTS',len(cells),'model-degree cells')

if __name__=='__main__':main()
