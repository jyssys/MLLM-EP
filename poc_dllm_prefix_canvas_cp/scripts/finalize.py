"""Separate physical facts, stationarity estimates and synthetic kernel tests."""
import collections
import csv
import gzip
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_dllm_prefix_canvas_cp.contract import *


def main():
    timing=json.loads((ROOT/'results/critical_path_summary.json').read_text())
    correctness=json.loads((ROOT/'results/correctness.json').read_text())
    directory=ROOT/'results/raw/stage_b_refinement_order_v3'
    assert json.loads((directory/'status.json').read_text())['success']
    physical=[json.loads((directory/f'results.rank{r}.json').read_text())['results'] for r in range(4)]
    micro=[]
    for index,prefix in enumerate([512,8192,16384]):
        rank=int(np.argmax([sum(s['GPU_ms'] for s in physical[r][index]['samples']) for r in range(4)]))
        data=physical[rank][index]
        means={arm:statistics.mean(s['GPU_ms'] for s in data['samples'] if s['arm']==arm)
               for arm in ['native','local','distributed']}
        errors=[e for r in range(4) for e in physical[r][index]['correctness']]
        micro.append(dict(prefix=prefix,physical_keys=2*prefix,critical_rank=rank,unit='ms /60 attention calls (20 layers x3 states)',
            **{arm+'_ms':value for arm,value in means.items()},
            speedup_vs_native=means['native']/means['distributed'],speedup_vs_local=means['local']/means['distributed'],
            prefix_prepare_warm_ms=data['prep']['GPU_ms'],prefix_prepare_cold_ms=data['prep']['cold_GPU_ms'],
            amortized_speedup_vs_native={str(h):means['native']/(means['distributed']+data['prep']['GPU_ms']*3/h) for h in [4,8,16,32]},
            max_abs=max(e['max_abs'] for e in errors),max_relative_L2=max(e['relative_L2'] for e in errors),
            BF16_pass=all(e['pass_all'] for e in errors),repeat_count=10,
            population_CI='NOT_APPLICABLE: synthetic microbenchmark, not independent requests'))
    write(ROOT/'results/stage_b_summary.json',dict(results=micro,scope='SYNTHETIC_SHAPE_AND_NATIVE_Q_LAYOUT_MATCHED_PHYSICAL_MICROBENCH',
        actual_GPUS=GPUS,rank_reuse='head-sharded inputs -> sequence-sharded prefix -> exact softmax -> original head owners',
        payload='Q/canvas gather, LSE MAX, weighted-output reduce-scatter; prefix all-to-all preparation charged separately',
        no_new_kernel=True,task_correctness='NOT_RUN',Stage_C='NOT_RUN: Gate B fails even with free preparation'))
    with (ROOT/'results/microbench.csv').open('w',newline='') as stream:
        flat=[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in micro]
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    raw=ROOT/'results/raw/stage_a_eager_prefill_v6'
    profiles=[]
    for rank in range(4):
        with gzip.open(raw/f'profiler_cuda.rank{rank}.json.gz','rt') as f:events=json.load(f)
        sums=collections.defaultdict(float);counts=collections.Counter()
        for e in events:
            sums[e['name']]+=(e['end_us']-e['start_us'])/1000;counts[e['name']]+=1
        profiles.append(dict(rank=rank,classes=[dict(name=k,ms=v,count=counts[k]) for k,v in sorted(sums.items(),key=lambda x:-x[1])],
            scope='HEAVY_PROFILER_KERNEL_CLASSIFICATION; not headline or nonoverlapped E2E attribution',
            copy_attribution='memcpy128 is a physical copy-class candidate, not a proven unique assignment to KV slice_scatter'))
    write(ROOT/'results/profiler_classification.json',dict(profiles=profiles,
        additive_components=False,warning='NCCL annotation ranges can overlap kernels; do not sum all classified names as independent stages'))
    kv=[]
    for prefix in PREFIXES:
        source=raw if prefix in [512,16384] else ROOT/'results/raw/stage_a_native_v4'
        for rank in range(4):
            path=source/f'coding_00.p{prefix}.trace.0.rank{rank}.json.gz'
            with gzip.open(path,'rt') as f:record=json.load(f)
            kv.extend(dict(prefix=prefix,rank=rank,**s) for s in record['KV_captures'])
    assert all(all(s['prefix_copy_equal']) and (s['prefix_equal_previous'] is None or all(s['prefix_equal_previous'])) for s in kv)
    write(ROOT/'results/KV_lifetime_summary.json',dict(samples=kv,
        prefix_immutable='PASS on3 sampled layers, first3 refinements,4 ranks,6 prefix lengths',
        canvas_change_count=sum(s['canvas_equal_previous'] is not None and not all(s['canvas_equal_previous']) for s in kv),
        commit_boundary='Native lifecycle source-audited; Stage B synthetic version-update checked, not full candidate generation parity'))
    write(ROOT/'results/final_verdict.json',dict(verdict='NO-GO',Stage_0='Native TP4 verified; P16384 eager-prefill fallback required',
        Stage_A='Material attention/KV superset grows; full-request precise attribution remains an explicitly modeled bound',
        Stage_B='FAIL: exact communication-inclusive CP is slower than native and stronger local control at every tested P',
        Stage_C='NOT_RUN',held_out='24 frozen requests untouched',
        actual_E2E_improvement='NOT_RUN; do not substitute Amdahl estimates',
        correctness=correctness['all_parity'],benchmark_accuracy='NOT_RUN',
        microbench_correctness=all(r['BF16_pass'] for r in micro),new_method=False,
        novelty='Not established beyond generic distributed softmax/cache layout',
        next_action='Stop CP integration. If pursued separately, first isolate exact local canvas-update/cache-staging cost in the native runtime.',
        branch='flashvep/dllm-prefix-canvas-cp-poc'))
    figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    keys=[str(p) for p in PREFIXES];s=timing['prefixes'];x=np.arange(len(keys))
    def values(field):return np.array([s[k][field]['mean'] for k in keys])
    def save(name):plt.tight_layout();plt.savefig(figs/name,dpi=160);plt.close()
    pre=values('prefill_ms');warm=values('warm_decode_span_ms');wall=values('cold_E2E_ms')
    plt.figure(figsize=(10,4));plt.bar(x,pre,label='Measured native prefill phase')
    plt.bar(x,warm,bottom=pre,label='Measured warm decode + refresh phase')
    plt.bar(x,np.maximum(wall-pre-warm,0),bottom=pre+warm,label='Other request wall')
    plt.xticks(x,keys);plt.xlabel('Logical prefix tokens');plt.ylabel('ms / request')
    plt.title('Native TP4,8 calibration requests/P; P512/16384 eager prefill');plt.legend(fontsize=8);save('latency_breakdown.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(x,values('warm_target_superset_fraction')*100,'o-',label='Warm attention/KV superset ESTIMATE')
    axes[0].plot(x,values('cold_repeated_target_fraction')*100,'s-',label='Cold repeated-target ESTIMATE')
    axes[0].set(xticks=x,xticklabels=keys,ylabel='Fraction (%)',xlabel='Prefix')
    for speed in ['1.5','2','4','infinite']:
        axes[1].plot(x,[s[k]['cold_repeated_target_fraction_Amdahl'][speed]['mean']*100 for k in keys],'o-',label=f'{speed}x target')
    axes[1].set(xticks=x,xticklabels=keys,ylabel='Optimistic cold latency reduction (%)',xlabel='Prefix')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Stationarity/superset Amdahl bounds — NOT measured achievable gains');save('attention_fraction_amdahl.png')
    plt.figure(figsize=(9,4));p=np.arange(3)
    for i,(name,label) in enumerate([('native','Native full-cache copy'),('local','Exact local canvas update'),('distributed','Exact CP incl. all collectives')]):
        plt.bar(p+(i-1)*.25,[r[name+'_ms']/60 for r in micro],.25,label=label)
    plt.xticks(p,[r['prefix'] for r in micro]);plt.ylabel('Measured ms / attention call');plt.xlabel('Logical prefix')
    plt.title('SYNTHETIC physical four-GPU component test\nPrefix preparation measured separately');plt.legend(fontsize=8);save('native_vs_candidate_microbench.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(x,values('KV_per_rank_bytes_proxy')/1024**2,'o-',label='One padded KV copy/rank')
    axes[0].set(xticks=x,xticklabels=keys,ylabel='Logical MiB',xlabel='Prefix',title='All20 full-attention layers; no sliding cap')
    axes[1].plot(x,values('peak_HBM_allocated')/1024**3,'o-',label='Measured peak allocated')
    axes[1].plot(x,values('peak_HBM_reserved')/1024**3,'s-',label='Measured peak reserved')
    axes[1].set(xticks=x,xticklabels=keys,ylabel='GiB/rank',xlabel='Prefix',title='Co-resident clean/event graph buffers; not minimal deployment')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    save('kv_hbm_profile.png')
    print('FINAL NO-GO; native parity comparisons',correctness['tested_comparisons'])
    for r in micro:print('B',r['prefix'],'native/local/CP',r['native_ms'],r['local_ms'],r['distributed_ms'],
        'speedups',r['speedup_vs_native'],r['speedup_vs_local'])


if __name__=='__main__':main()
