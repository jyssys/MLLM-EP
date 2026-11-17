"""Compare logical TP tensors without mixing different greedy prefixes."""
import json
import re
from .common import ROOT, digest, write


def gather(key, values):
    import torch
    if len(values) == 1:
        return values[0]
    if 'qkv_proj:output' in key:
        chunks = [v.split((2048, 512, 512), dim=-1) for v in values]
        return torch.cat([torch.cat([v[i] for v in chunks], dim=-1) for i in range(3)], dim=-1)
    if 'gate_up_proj:output' in key or 'act_fn:input' in key:
        chunks = [v.chunk(2, dim=-1) for v in values]
        return torch.cat([torch.cat([v[i] for v in chunks], dim=-1) for i in range(2)], dim=-1)
    if any(t in key for t in ('q_norm:', 'k_norm:')):
        return torch.cat(values, dim=-2)
    if 'rotary_emb:' in key and ':input.0' not in key:
        return torch.cat(values, dim=-1)
    if '.attn:' in key:
        return torch.cat(values, dim=-1)
    if 'o_proj:input' in key or 'act_fn:output' in key or 'down_proj:input' in key:
        return torch.cat(values, dim=-1)
    return values[0]


def errors(a, b):
    import torch
    assert a.shape == b.shape, (a.shape, b.shape)
    x, y = a.float(), b.float()
    delta = x - y
    byte_a, byte_b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    out = dict(shape=list(a.shape), equal=bool(torch.equal(byte_a, byte_b)),
        equal_numeric=bool(torch.equal(a, b)), differing_bytes=int((byte_a != byte_b).sum()),
        max_absolute=float(delta.abs().max()), RMS=float(delta.square().mean().sqrt()),
        relative_L2=float(delta.norm() / x.norm().clamp_min(1e-30)),
        differing_elements=int((a != b).sum()), elements=a.numel())
    if a.dtype == b.dtype == torch.bfloat16:
        def ordered(v):
            u = v.contiguous().view(torch.int16).to(torch.int32)
            return torch.where(u < 0, -32768 - u, u)
        ulp = (ordered(a) - ordered(b)).abs()
        out.update(max_BF16_ULP=int(ulp.max()), mean_BF16_ULP=float(ulp.float().mean()))
    return out


def analyze(tp1_name, tp2_name):
    import torch
    torch.set_num_threads(1)
    aroot, broot = [ROOT / 'raw' / n for n in (tp1_name, tp2_name)]
    a = torch.load(aroot / 'tensors_rank0.pt', map_location='cpu', weights_only=True)
    bs = [torch.load(broot / f'tensors_rank{r}.pt', map_location='cpu', weights_only=True) for r in range(2)]
    outputs = [json.loads((p / 'greedy_output.json').read_text()) for p in (aroot, broot)]
    assert outputs[0]['input_ids'] == outputs[1]['input_ids']
    flip = next((i for i, (x, y) in enumerate(zip(outputs[0]['output_ids'], outputs[1]['output_ids'])) if x != y), None)
    common_steps = range((flip + 1) if flip is not None else len(outputs[0]['output_ids']))
    comparisons, first = [], {}
    for key, value in a.items():
        step = int(key.split(':')[0])
        if step not in common_steps or any(key not in b for b in bs):
            continue
        other = gather(key, [b[key] for b in bs])
        result = dict(key=key, step=step, **errors(value, other))
        comparisons.append(result)
        if not result['equal']:
            first.setdefault(str(step), result)
    logits = []
    for step in common_steps:
        key = str(step) + ':logits'
        if key not in a or key not in bs[0]:
            continue
        pair = []
        for values in (a[key], bs[0][key]):
            top = values[0].float().topk(2)
            pair.append(dict(token_ids=top.indices.tolist(), logits=top.values.tolist(),
                greedy_argmax=int(values[0].argmax()),
                top1_top2_margin=float(top.values[0] - top.values[1])))
        logits.append(dict(step=step, TP1=pair[0], TP2=pair[1]))
    assert comparisons and first and logits, 'Incomplete tensor witness'
    result = dict(TP1=tp1_name, TP2=tp2_name, first_output_flip=flip,
        common_prefix_steps=list(common_steps), first_differences=first,
        comparisons=comparisons, logits=logits, evidence='CORRECTNESS_ONLY',
        tensor_archives={str(p.relative_to(ROOT)): digest(p) for root in (aroot, broot) for p in root.glob('*.pt')},
        scope='Logical shards gathered by Q/K/V, gate/up, attention-head axes. Only steps with identical input token prefixes are compared. No timing claim. This is tensor localization, not yet a primitive-level causal proof.')
    write(ROOT / 'analysis/TP1_TP2_first_divergence.json', result)
    print(json.dumps(dict(first_output_flip=flip, first_differences=first, logits=logits), indent=2))
    return result


if __name__ == '__main__':
    import sys
    analyze(*sys.argv[1:])
