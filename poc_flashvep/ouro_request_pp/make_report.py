"""Generate result tables from preserved physical measurements only."""
import collections
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contract import ROOT,write
from .analyze import read


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def median(rows,key):return float(np.median([r[key] for r in rows]))


def main():
    raw=json.loads((ROOT/'results_raw.json').read_text())
    primary=[r for r in raw if '/primary_' in r['file'] and '_v5_' in r['file']]
    paired=json.loads((ROOT/'paired_comparison.json').read_text())
    assert len(primary)==162 and len(paired)==27 and all(r['N_repeats']==3 for r in paired)
    checks=[r for f in ROOT.glob('correctness_checks_graph_v5*.json') for r in json.loads(f.read_text())]
    assert all(r['PASS'] and r['KV_bitwise'] and r['logit_error']['bitwise'] for r in checks)
    groups=collections.defaultdict(list)
    for r in primary:groups[(r['depth'],r['C'],r['P'],r['O'],r['placement'])].append(r)
    rows=[]
    for r in paired:
        ci=r['CI95'];rows.append([r['depth'],r['C'],r['P'],r['O'],f"{r['layer_median_s']:.4f}",
            f"{r['loop_median_s']:.4f}",f"{r['layer_over_loop_speedup']:.3f}×",
            f"[{ci[0]:.3f}, {ci[1]:.3f}]"])
    text='''# Benchmark summary

## Primary physical comparison

Evidence: LIVE_MEASURED. Each row contains three paired complete-workload repeats.
Both arms use physical GPUs 4,5,6,7, BF16, the same FIFO scheduler, and the same frozen token IDs.
Each stage handles one entire prompt or one decode token from one request.
The runtime forces output length. These are hardware comparisons, not accuracy measurements.
Graph preparation and common warm-up precede the serving clock.
All primary runs contain zero graph captures inside the serving clock.

Speedup is median Layer-PP makespan divided by median Loop-PP makespan.
A value below 1.000× favors Layer-PP.
Intervals use 2000 paired workload bootstrap resamples.
Only three repeats support each interval. They do not establish broad statistical significance.
P=-1 denotes frozen mixed prompt lengths 128/512/2048. It does not denote prompt chunking.

'''+table(['R','C','Input tokens','Output tokens/request','Layer-PP s','Loop-PP s','Loop/Layer speedup','95% CI'],rows)
    rows=[]
    for key,g in sorted(groups.items()):
        R,C,P,O,arm=key
        rows.append([R,C,P,O,arm,f"{median(g,'requests_per_s'):.3f}",f"{median(g,'tokens_per_s'):.2f}",
            f"{median(g,'TTFT_P50_ms'):.2f} / {median(g,'TTFT_P90_ms'):.2f}",
            f"{median(g,'TPOT_P50_ms'):.2f} / {median(g,'TPOT_P90_ms'):.2f}",
            f"{median(g,'E2E_P50_ms'):.2f} / {median(g,'E2E_P90_ms'):.2f}"])
    text+='''
## Serving metrics

Table entries are medians across three runs of each run's request percentile.
TTFT includes request queueing and complete prefill.
TPOT excludes the first token and includes contention from other requests.
E2E ends at the last token ID available to the controller.
Requests/s and output tokens/s describe a finite cohort with startup and drain.
They are not an independently measured infinite-backlog capacity.

'''+table(['R','C','P','O','Arm','Requests/s','Output tokens/s','TTFT P50/P90 ms','TPOT P50/P90 ms','E2E P50/P90 ms'],rows)
    rows=[]
    for R,C,P,O in [(4,4,128,32),(4,16,512,128),(2,16,128,32),(3,16,128,32)]:
        for arm in ['layer_pp4','loop_pp4']:
            g=groups[(R,C,P,O,arm)];s=g[0]
            rows.append([R,C,P,O,arm,s['hidden_handoffs'],f"{s['hidden_bytes']/2**20:.3f}",
                s['token_return_count'],s['token_return_bytes'],
                '/'.join(f'{v:.3f}' for v in s['rank_weights_GiB']),
                '/'.join(f'{v:.3f}' for v in s['rank_KV_GiB']),f"{median(g,'max_peak_HBM_GiB'):.3f}"])
    text+='''
## Transfer and memory examples

Hidden transfer counts and tensor bytes come from executed send paths.
They exclude token return, initial input transfer, and scalar control transfer.
Owner-local KV is physical cache allocation, including reserved output capacity.
Peak HBM is the PyTorch peak allocation, not a whole-driver NVML peak.
Graph scratch, graph pools, and allocator storage are additional to the listed KV.
Earlier graph shapes can remain resident in the same campaign process.
No OOM occurred in the executed matrix. A maximum-concurrency capacity search was not performed.

'''+table(['R','C','P','O','Arm','Hidden transfers','Hidden MiB','Token returns','Token bytes',
           'Weights GiB GPU4/5/6/7','KV GiB GPU4/5/6/7','Max peak GiB'],rows)
    (ROOT/'benchmark_summary.md').write_text(text)

    profiles=[]
    for f in sorted((ROOT/'profiles').glob('*/summary.json')):
        p=json.loads(f.read_text());p['summary_file']=str(f.relative_to(ROOT));profiles.append(p)
    write(ROOT/'profile_index.json',profiles)
    observer=[]
    for arm in ['layer_pp4','loop_pp4']:
        clean=[r for r in raw if f'observer_{arm}_clean_v5/' in r['file']]
        trace=[r for r in raw if f'observer_{arm}_trace_v5/' in r['file']]
        if clean and trace:
            observer.append(dict(arm=arm,clean_median_s=median(clean,'wall_s'),trace_median_s=median(trace,'wall_s'),
                                 trace_over_clean=median(trace,'wall_s')/median(clean,'wall_s')))
    write(ROOT/'observer_overhead.json',observer)
    correctness='''# Correctness

## Reference and acceptance rule

The reference loads the original pretrained HF model on physical GPU4.
It changes only the configured executed loop count for fixed R2/R3.
Layer hooks assert `24 * R * generated_tokens` actual decoder calls.
The reference retains separate KV for every executed loop and physical layer.
Greedy generation feeds each selected token into the next forward pass.
This is free-running generation, not teacher forcing.

Numerical limits were frozen before measurements in [preflight.json](preflight.json).
They require zero greedy-token mismatches.
They permit logit max/mean absolute errors of 0.5/0.05.
They permit hidden-state and KV relative L2 errors of 0.02.
No held-out failure caused a tolerance increase.

## Final runtime results

'''+f'{len(checks)} request-arm-depth-shape checks passed. Every checked terminal logit, hidden state, and valid KV element matched bitwise.\n\n'
    shape=collections.Counter((r['P'],r['C'],r['depth']) for r in checks)
    correctness+=table(['Prompt tokens','Concurrent requests','Executed depth','Request-arm checks','Result'],
                       [[P,C,R,n,'PASS, bitwise'] for (P,C,R),n in sorted(shape.items())])
    correctness+='''
These checks reuse a small fixed prompt pool. They are not independent benchmark-quality samples.
All 81 primary paired cohorts also produce identical greedy token IDs across placements.
Each primary output contains exactly the forced token count.
Cache ownership and length checks cover every primary request.
The final cache length is `prompt_length + generated_tokens - 1`.
The last emitted token does not enter the cache until another forward pass consumes it.
All cache slots stay on their executing GPU.
The slot key is `24 * loop_index + physical_layer_index`.

## Preserved failures and changes

1. A masked-capacity SDPA prototype changed a near-tied greedy token. The exact attention length replaced that path.
2. A CUDA Graph prototype failed interleaved requests because its position tensor lacked an explicit lifetime owner.
3. The final graph tuple retains that tensor. The same interleaving regression then passed.
4. An early campaign stopped during graph setup. It produced no headline measurements.

The failed outputs, commands, and logs remain under `failures/`, `logs/`, and versioned correctness JSON files.
The final comparison uses only `primary_*_v5_r*` records.
Older eager and graph prototypes do not contribute to headline latency.

## Evaluation boundaries

Benchmark accuracy: NOT_RUN. Forced-length hardware runs do not establish GSM8K or MATH accuracy.
True dynamic-depth/ragged-KV correctness: NOT_RUN.
Fixed R2/R3 truncate both prefill and decode. They do not reproduce full-depth prefill followed by adaptive CDB decode.
The official HF gate collects loop states before its exit selection. That path is not proof of online compute skipping.
No gate, threshold, router, expert, precision, or pretrained tensor changed in this experiment.
This dense model has no MoE router or expert path.
'''
    extra=[r for f in ROOT.glob('correctness_checks_graph_receive_pool*.json') for r in json.loads(f.read_text())]
    if extra:
        assert all(r['PASS'] and r['KV_bitwise'] and r['logit_error']['bitwise'] for r in extra)
        correctness+=f'\nThe optional backend controls add {len(extra)} reference checks. They also pass with bitwise terminal logits and KV.\n'
    eos=[r for r in raw if '/eos_' in r['file']]
    if eos:
        correctness+='\n## EOS-enabled smoke\n\n'
        for r in eos:
            d=read(ROOT/r['file']);stops=collections.Counter(v['finish_reason'] for v in d['outputs'])
            correctness+=f"- {r['placement']}: {dict(stops)}, lengths {[len(v['ids']) for v in d['outputs']]}.\n"
        a,b=[read(ROOT/r['file']) for r in eos]
        parity={v['id']:(v['ids'],v['finish_reason']) for v in a['outputs']}=={v['id']:(v['ids'],v['finish_reason']) for v in b['outputs']}
        correctness+=f'\nEOS-enabled arm-to-arm token/stop parity: {parity}.\n'
        correctness+='If every request reaches its cap, this smoke tests enabled stopping but not an actual EOS transition.\n'
    (ROOT/'correctness.md').write_text(correctness)

    # Small plots retain separate units and do not turn rank sums into E2E shares.
    figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for arm in ['layer_pp4','loop_pp4']:
        rr=[next(v for v in groups[(R,16,128,32,arm)]) for R in [2,3,4]]
        axs[0].plot([2,3,4],[v['hidden_handoffs'] for v in rr],marker='o',label=arm)
        axs[1].plot([2,3,4],[median(groups[(R,16,128,32,arm)],'tokens_per_s') for R in [2,3,4]],marker='o',label=arm)
    axs[0].set(xlabel='Fixed executed loops',ylabel='Measured hidden transfer count / cohort')
    axs[1].set(xlabel='Fixed executed loops',ylabel='Output tokens/s, C16 P128 O32')
    for ax in axs:ax.legend()
    fig.tight_layout();fig.savefig(figs/'depth_sweep.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for j,arm in enumerate(['layer_pp4','loop_pp4']):
        d=read(ROOT/groups[(4,16,512,128,arm)][0]['file'])
        w=np.array([m['weights_bytes']['total'] for m in d['memory']])/2**30
        kv=np.array([m['KV_bytes'] for m in d['memory']])/2**30
        other=np.array([m['allocated_bytes'] for m in d['memory']])/2**30-w-kv
        axs[j].bar(range(4),w,label='Weights');axs[j].bar(range(4),kv,bottom=w,label='Owner-local KV')
        axs[j].bar(range(4),other,bottom=w+kv,label='Other allocation, incl. graphs')
        axs[j].set(xticks=range(4),xticklabels=['GPU4','GPU5','GPU6','GPU7'],ylabel='PyTorch allocated GiB',title=arm)
    axs[1].legend();fig.suptitle('R4 C16 P512 O128, repeat 0. Not cold allocator capacity.')
    fig.tight_layout();fig.savefig(figs/'memory.png',dpi=170);plt.close(fig)
    write(ROOT/'report_facts.json',dict(primary_runs=len(primary),paired_cases=len(paired),correctness_checks=len(checks),
        profiles=len(profiles),loop_wins=sum(r['layer_over_loop_speedup']>1 for r in paired),
        layer_wins=sum(r['layer_over_loop_speedup']<1 for r in paired),
        speedup_min=min(r['layer_over_loop_speedup'] for r in paired),speedup_max=max(r['layer_over_loop_speedup'] for r in paired)))
    print('REPORT_TABLES',len(primary),len(checks),len(profiles))


if __name__=='__main__':main()
