"""Structural one-handoff frontiers. These counts are not GPU measurements."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Frontier:
    tokens: int
    loops: int
    shallow_loops: int
    deep_prefix: int

    def __post_init__(self):
        if not (self.tokens > 0 and self.loops > 0
                and 0 <= self.shallow_loops <= self.loops
                and 0 <= self.deep_prefix <= self.tokens):
            raise ValueError('Invalid one-handoff frontier')

    def p_owned(self, depth, token):
        return depth < self.shallow_loops or token < self.deep_prefix

    def verify(self):
        # Check coarse block endpoints instead of materializing T² edges.
        # A P node must have all previous-depth and same-depth-prefix inputs on P.
        for d in range(self.loops):
            for t in {0, self.tokens - 1, max(0, self.deep_prefix - 1), min(self.tokens - 1, self.deep_prefix)}:
                if self.p_owned(d, t):
                    if d and not self.p_owned(d - 1, t):
                        return False
                    if t and not self.p_owned(d, t - 1):
                        return False
        return True

    def accounting(self, model):
        if model not in ('ouro', 'huginn'):
            raise ValueError('Unknown model')
        t, r, c, k = self.tokens, self.loops, self.shallow_loops, self.deep_prefix
        suffix, remaining = t - k, r - c
        rows = suffix * remaining
        pairs = remaining * (t * (t + 1) - k * (k + 1)) // 2
        # BF16 K and V. Ouro has 24 recurrent layers at 16 KV heads × 128.
        # Huginn has 4 recurrent layers at 55 KV heads × 96, plus 2+2 boundary layers.
        per_recurrent_row = 196608 if model == 'ouro' else 84480
        per_boundary_token = 0 if model == 'ouro' else 42240
        full = not rows
        dlocal = c == 0 and k == 0
        recurrent_kv = per_recurrent_row * (r * t - rows) if not dlocal else 0
        prelude_tokens = t if c else k
        coda_tokens = t - 1 if full else min(k, t - 1)
        boundary_kv = per_boundary_token * (prelude_tokens + coda_tokens) if model == 'huginn' else 0
        width = 2048 if model == 'ouro' else 10560
        # Minimal state contract. The current implementation can register/copy
        # more storage. Actual transport bytes must be reconciled separately.
        hidden = (width * 2 if full or c == 0 else suffix * width * 2)
        canonical_suffix = suffix * 5280 * 2 if model == 'huginn' and c == 0 and not dlocal else 0
        if dlocal:
            hidden = boundary_kv = 0
        return dict(**asdict(self), DAG_closed_P_partition=self.verify(),
            D_recurrent_token_rows=rows, full_recurrent_token_rows=r * t,
            D_recurrent_row_fraction=rows / (r * t),
            D_causal_pairs_per_recurrent_attention_layer=pairs,
            recurrent_KV_bytes=recurrent_kv, boundary_KV_bytes=boundary_kv,
            minimum_hidden_bytes=hidden, canonical_suffix_initial_state_bytes=canonical_suffix,
            minimum_state_payload_bytes=recurrent_kv + boundary_kv + hidden + canonical_suffix,
            D_Huginn_prelude_tokens=suffix if model == 'huginn' and c == 0 else 0,
            D_Huginn_Coda_KV_tokens=max(0, t - 1 - k) if model == 'huginn' and not full else 0,
            runtime_status='STRUCTURAL_ONLY_NOT_GPU_QUALIFIED',
            evidence='STRUCTURAL_ACCOUNTING_NOT_FLOPS_OR_SPEEDUP')


def invalid_checker(owners, loops, tokens):
    """Reject a P partition that reads a D-owned dependency before handoff."""
    for d, t in owners:
        if d and (d - 1, t) not in owners:
            return False
        if any((d, j) not in owners for j in range(t)):
            return False
    return all(0 <= d < loops and 0 <= t < tokens for d, t in owners)
