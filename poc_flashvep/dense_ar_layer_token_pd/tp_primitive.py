"""Replay the first differing row projection with identical captured operands."""
import json
import sys
import time
from .common import ROOT, write


def main():
    cfg = json.load(open(sys.argv[1]))
    folder = ROOT / 'raw' / cfg['name']
    write(folder / 'status.json', dict(status='RUNNING', time=time.time()))
    import torch
    from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    from ..dense_ar_slo_layer_pd.watch import GPUWatch
    from .tensor_analysis import errors
    torch.set_num_threads(1)
    enable_batch_invariant_mode()
    torch.backends.cuda.matmul.allow_tf32 = False
    monitor = GPUWatch(folder, cfg['gpus'], cfg['name'])
    monitor.start()
    normal = False
    try:
        roots = [ROOT / 'raw' / n for n in (cfg['TP1_witness'], cfg['TP2_witness'])]
        a = torch.load(roots[0] / 'tensors_rank0.pt', weights_only=True)
        bs = [torch.load(roots[1] / f'tensors_rank{r}.pt', weights_only=True) for r in range(2)]
        wa = torch.load(roots[0] / 'weights_rank0.pt', weights_only=True)
        wb = [torch.load(roots[1] / f'weights_rank{r}.pt', weights_only=True) for r in range(2)]
        name = 'model.layers.0.self_attn.o_proj'
        weight = wa[name + '.weight'].cuda()
        shards = [w[name + '.weight'].cuda() for w in wb]
        assert torch.equal(weight, torch.cat(shards, dim=1)), 'Weight shard mismatch'
        results, tensors = [], {}
        for step in (0, 1, 6):
            key = str(step) + ':' + name
            inputs = a[key + ':input.0'].cuda()
            assert torch.equal(inputs, torch.cat([b[key + ':input.0'].cuda() for b in bs], dim=-1))
            for shape in sorted({len(inputs), 8000 if step == 0 else len(inputs)}):
                x = inputs.repeat((shape + len(inputs) - 1) // len(inputs), 1)[:shape]
                full = torch.nn.functional.linear(x, weight)[-len(inputs):]
                partials = [torch.nn.functional.linear(v, w)[-len(inputs):]
                            for v, w in zip(x.chunk(2, dim=-1), shards)]
                reduced = partials[0] + partials[1]
                fp32_full = torch.nn.functional.linear(x.float(), weight.float())[-len(inputs):]
                fp32_parts = sum(torch.nn.functional.linear(v.float(), w.float())[-len(inputs):]
                                 for v, w in zip(x.chunk(2, dim=-1), shards))
                result = dict(step=step, replay_M=shape,
                    TP1_replay_vs_actual=errors(full.cpu(), a[key + ':output.0']),
                    TP2_partials_replay_vs_actual=[errors(p.cpu(), b[key + ':collective_input']) for p, b in zip(partials, bs)],
                    TP2_replay_vs_actual=errors(reduced.cpu(), bs[0][key + ':output.0']),
                    BF16_full_vs_split=errors(full.cpu(), reduced.cpu()),
                    FP32_full_vs_split=errors(fp32_full.cpu(), fp32_parts.cpu()),
                    full_BF16_vs_FP32_RMS=float((full.float() - fp32_full).square().mean().sqrt()),
                    split_BF16_vs_FP32_RMS=float((reduced.float() - fp32_full).square().mean().sqrt()),
                    collective_actual_vs_manual_sum=errors(
                        bs[0][key + ':collective_input'] + bs[1][key + ':collective_input'],
                        bs[0][key + ':collective_output']))
                results.append(result)
                tensors[f'{step}_{shape}'] = dict(full=full.cpu(), partials=[p.cpu() for p in partials],
                    reduced=reduced.cpu(), fp32_full=fp32_full.cpu(), fp32_split=fp32_parts.cpu())
        torch.save(tensors, folder / 'primitive_tensors.pt')
        write(folder / 'primitive_result.json', dict(results=results, weight_shards_exact=True,
            inputs_exact=True, BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            FP32_replay_allow_TF32=torch.backends.cuda.matmul.allow_tf32,
            evidence='CORRECTNESS_ONLY', scope='Captured actual operands. Two BF16 local projection outputs are rounded before their sum. FP32 replay is diagnostic only. Repeated independent rows preserve per-row math; replay_M=8000 tests the original GEMM shape. No serving timing claim.'))
        normal = True
        write(folder / 'status.json', dict(status='COMPLETE', time=time.time()))
    finally:
        monitor.stop()
        write(folder / 'worker_exits.json', dict(passed=normal))


if __name__ == '__main__':
    main()
