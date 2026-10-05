"""Exact independent reference for global softmax state merging."""
import torch


def partial_state(q, k, v, mask=None):
    shape = (*q.shape[:-1], 1)
    if k.shape[-2] == 0:
        return (torch.full(shape, -torch.inf, device=q.device, dtype=torch.float32),
                torch.zeros(shape, device=q.device, dtype=torch.float32),
                torch.zeros_like(q, dtype=torch.float32))
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / q.shape[-1]**.5
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    maximum = scores.amax(-1, keepdim=True)
    safe = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = (scores - safe).exp()
    return maximum, weights.sum(-1, keepdim=True), weights @ v.float()


def merge_states(a, b):
    ma, za, ua = a
    mb, zb, ub = b
    maximum = torch.maximum(ma, mb)
    safe = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    wa, wb = (ma-safe).exp(), (mb-safe).exp()
    denominator = wa*za + wb*zb
    numerator = wa*ua + wb*ub
    return torch.where(denominator > 0, numerator/denominator.clamp_min(1e-30), torch.zeros_like(numerator))


def valid_prefix(current_start, current_rows, positions, allowed, cache_length):
    # All fresh rows, including accepted positions, remain in current C.
    # P contains only valid native keys strictly before this invocation span.
    return positions[:current_start], allowed[..., :current_start], current_rows
