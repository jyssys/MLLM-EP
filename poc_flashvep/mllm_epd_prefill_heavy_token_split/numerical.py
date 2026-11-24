"""Correctness metrics with FP64 accumulation for large BF16 feature tensors."""
import math


def embedding_metrics(actual, target):
    import torch
    assert actual.shape == target.shape and actual.numel()
    a, b = actual.detach().float(), target.detach().float()
    d = a - b
    a2 = float(a.square().sum(dtype=torch.float64))
    b2 = float(b.square().sum(dtype=torch.float64))
    d2 = float(d.square().sum(dtype=torch.float64))
    ab = float((a * b).sum(dtype=torch.float64))
    cosine = ab / math.sqrt(a2 * b2) if a2 and b2 else float(a2 == b2)
    return dict(relative_L2=math.sqrt(d2 / b2) if b2 else (0.0 if not d2 else float('inf')),
                cosine=cosine, max_abs=float(d.abs().max()), mean_abs=float(d.abs().mean(dtype=torch.float64)),
                reference_RMS=math.sqrt(b2 / b.numel()), actual_RMS=math.sqrt(a2 / a.numel()),
                strict=torch.equal(a, b), accumulation='FP64 reductions of FP32 products',
                finite=all(math.isfinite(x) for x in (a2, b2, d2, ab, cosine)))
