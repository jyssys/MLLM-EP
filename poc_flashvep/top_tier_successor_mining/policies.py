"""Paper-baseline primitives only; no successor optimization.

SERE: Algorithm 2 / official rerouting_kernel.cu.
MoDES: normalized importance times original normalized top-k probability;
       skip contribution without post-skip renormalization.
Libra: next-layer gate on current MoE input, not previous selected expert IDs.
"""
import torch


def sere_route(ids, similarity, retain=2, threshold=0.5, valid=None):
    if retain <= 0 or retain >= ids.shape[-1]:
        return ids.clone()
    n = similarity.shape[0]
    primary = torch.zeros(n, device=ids.device, dtype=torch.bool)
    selected = ids if valid is None else ids[valid]
    if selected.numel() == 0:
        return ids.clone()
    primary.scatter_(0, selected[:, :retain].reshape(-1).long(), True)
    allowed = similarity.masked_fill(~primary.unsqueeze(0), -float("inf"))
    best_score, best_id = allowed.max(dim=1)
    mapping = torch.arange(n, device=ids.device)
    can_replace = ~primary
    if threshold > 0:
        can_replace = can_replace & (best_score >= threshold)
    mapping = torch.where(can_replace, best_id, mapping)
    out = ids.clone()
    out[:, retain:] = mapping[ids[:, retain:].long()].to(ids.dtype)
    return out


def modes_weights(weights, text_mask, vision_mask, alpha_text, alpha_vision,
                  tau_text, tau_vision):
    drop = ((text_mask[:, None] & (weights * alpha_text < tau_text)) |
            (vision_mask[:, None] & (weights * alpha_vision < tau_vision)))
    return weights.masked_fill(drop, 0.0), drop


def libra_predict(hidden, next_gate_weight, top_k=8):
    # Top-k of logits equals top-k of softmax; use softmax for reported confidence.
    logits = torch.nn.functional.linear(hidden, next_gate_weight)
    probs = logits.float().softmax(-1)
    values, ids = probs.topk(top_k, dim=-1)
    return ids, values


def set_overlap(a, b):
    return (a[:, :, None] == b[:, None, :]).any(-1).float().mean(-1)
