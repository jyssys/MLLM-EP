"""Run fixed-depth correctness before paired physical pipeline measurements."""
import argparse
import gc
import gzip
import json
import subprocess
import time
import torch
import transformers
from .contract import *
from .reference import golden, errors
from .runtime import Pipeline


def rows(length, count):
    groups=json.loads((ROOT/'workload.json').read_text())['groups']
    if length==-1:
        return [groups[str([128,512,2048][i%3])][i] for i in range(count)]
    return groups[str(length)][:count]


def environment():
    validate_mask()
    assert torch.cuda.device_count() == 4
    peers = [[torch.cuda.can_device_access_peer(a,b) if a!=b else True for b in range(4)] for a in range(4)]
    assert all(all(r) for r in peers)
    write(ROOT/'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        CUDA=torch.version.cuda, NCCL=torch.cuda.nccl.version(), peers=peers, mask=os.environ['CUDA_VISIBLE_DEVICES'],
        logical_physical_mapping=dict(enumerate(GPUS)),
        devices=subprocess.check_output(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        source_reuse='Official pinned AutoModel decoder. Existing CDB checkpoint/mask audit used for inspection, not model math.'))


def correctness(depths=(4,2,3),graphs=False,prompt_length=16,count=4,label='v5',preallocated_recv=False,precompute_rotary=False):
    checks = []
    for depth in depths:
        inputs = rows(prompt_length,count)
        gold = golden(inputs, depth, 8)
        for arm in ['layer_pp4','loop_pp4']:
            engine = Pipeline(arm,preallocated_recv=preallocated_recv,precompute_rotary=precompute_rotary)
            result = engine.run(inputs, depth, 8, trace=True, correctness=True,graphs=graphs)
            for out in result['outputs']:
                rid = out['id']; ref = gold[rid]
                endpoint = engine.path[-1][0]
                snapshot = engine.parts[endpoint].snapshots[rid]
                logit_error = errors(snapshot['logits'], ref['logits'])
                hidden_error = errors(snapshot['hidden'], ref['hidden'])
                cache = {}
                for p in engine.parts:
                    if rid in p.caches:
                        local = p.snapshot(rid)
                        assert not (cache.keys() & local.keys()), 'KV ownership overlaps'
                        cache.update(local)
                assert set(cache) == set(ref['cache'])
                numerator = denominator = 0.
                worst = 0.; exact = True
                for slot in cache:
                    for a,b in zip(cache[slot], ref['cache'][slot]):
                        assert a.shape == b.shape
                        numerator += float((a.float()-b.float()).square().sum())
                        denominator += float(b.float().square().sum())
                        worst = max(worst, float((a.float()-b.float()).abs().max()))
                        exact &= torch.equal(a,b)
                kv_rel = (numerator/max(denominator,1e-30))**.5
                check = dict(arm=arm, depth=depth, C=count,P=prompt_length,id=rid, token_exact=out['ids']==ref['ids'],
                    golden_ids=ref['ids'], observed_ids=out['ids'], logit_error=logit_error,
                    hidden_error=hidden_error, KV_relative_L2=kv_rel, KV_max_abs=worst, KV_bitwise=exact,
                    KV_slots=len(cache), golden_layer_calls=ref['layer_calls'],
                    expected_cache_length=len(out['input_ids'])+len(out['ids'])-1)
                check['PASS'] = (check['token_exact'] and kv_rel<=TOLERANCE['KV_relative_L2']
                    and hidden_error['relative_L2']<=TOLERANCE['hidden_relative_L2']
                    and logit_error['max_abs']<=TOLERANCE['logit_max_abs']
                    and logit_error['mean_abs']<=TOLERANCE['logit_mean_abs'])
                checks.append(check)
                suffix='' if prompt_length==16 and count==4 else f'_P{prompt_length}_C{count}'
                write(ROOT/f'correctness_checks_{"graph" if graphs else "eager"}_{label}{suffix}.json', checks)
                assert check['PASS'], check
            write(ROOT/f'traces/correctness_{label}_{"graph" if graphs else "eager"}_{arm}_R{depth}_P{prompt_length}_C{count}.json', result)
            engine.close(); del engine
            gc.collect()
            for g in range(4):
                with torch.cuda.device(g): torch.cuda.empty_cache()
            print('CORRECTNESS_PASS', arm, depth, flush=True)
        del gold


def run_cases(args):
    # Separate process invocation for each placement avoids retained CUDA allocations.
    engine = Pipeline(args.placement,preallocated_recv=args.preallocated_recv,precompute_rotary=args.precompute_rotary)
    engine.run(rows(16, 4), 4, 4)  # Both arms use the same warm-up workload.
    outdir = ROOT/'raw'/args.label
    outdir.mkdir(parents=True, exist_ok=False)
    if args.case_file:
        cases = json.loads(Path(args.case_file).read_text())['cases']
    else:
        cases = [dict(depth=args.depth, C=args.concurrency, P=args.prompt_len, O=args.output_len, repeat=0)]
    for i, c in enumerate(cases):
        start = time.perf_counter()
        if args.graphs:
            warm_start=time.perf_counter()
            # Exact-length graphs are shared across request IDs, not their KV state.
            for length in ([128,512,2048] if c['P']==-1 else [c['P']]):
                engine.run(rows(length,1),c['depth'],c['O'],graphs=True)
            warmup_wall_s=time.perf_counter()-warm_start
        else:
            warmup_wall_s=0.
        if args.nsys:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push('MEASURED_WORKLOAD')
        result = engine.run(rows(c['P'], c['C']), c['depth'], c['O'], trace=args.trace,
                            stop_eos=args.stop_eos,graphs=args.graphs)
        torch.cuda.nvtx.range_pop()
        if args.nsys:
            torch.cuda.cudart().cudaProfilerStop()
        result['case'] = c
        result['runner_wall_s'] = time.perf_counter()-start
        result['exact_length_warmup_wall_s']=warmup_wall_s
        with gzip.open(outdir/f'case_{i:04d}.json.gz','wt') as f:
            json.dump(result,f,separators=(',',':'))
        print('MEASURED', args.placement, c, round(result['wall_s'],4), flush=True)
    engine.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['correctness','benchmark'], required=True)
    p.add_argument('--placement', choices=['layer_pp4','loop_pp4'], default='layer_pp4')
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--prompt-len', type=int, default=128)
    p.add_argument('--output-len', type=int, default=8)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--graphs', action='store_true')
    p.add_argument('--nsys', action='store_true')
    p.add_argument('--stop-eos', action='store_true')
    p.add_argument('--preallocated-recv', action='store_true')
    p.add_argument('--precompute-rotary', action='store_true')
    p.add_argument('--label', default='probe')
    p.add_argument('--case-file')
    p.add_argument('--correctness-prompt',type=int,default=16)
    p.add_argument('--correctness-count',type=int,default=4)
    p.add_argument('--correctness-depths',default='4,2,3')
    p.add_argument('--runtime-label',default='v5')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    environment()
    if args.mode == 'correctness': correctness(graphs=args.graphs,prompt_length=args.correctness_prompt,
        count=args.correctness_count,depths=list(map(int,args.correctness_depths.split(','))),label=args.runtime_label,
        preallocated_recv=args.preallocated_recv,precompute_rotary=args.precompute_rotary)
    else: run_cases(args)


if __name__ == '__main__':
    main()
