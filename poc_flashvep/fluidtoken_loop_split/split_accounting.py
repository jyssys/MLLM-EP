"""Structural recurrent-row and causal-attention accounting, not GPU timing."""


def account(length, depth, cut, prefix):
    if not (length > 0 and depth > 0 and 0 <= cut <= depth and 0 <= prefix <= length):
        raise ValueError('Invalid partition boundary')
    pairs = length * (length + 1) // 2
    prefix_pairs = prefix * (prefix + 1) // 2
    total_rows = length * depth
    total_pairs = pairs * depth
    return dict(
        evidence='STRUCTURAL_ACCOUNTING_NOT_MEASURED_FLOPS_OR_TIME',
        full_recurrent_rows=total_rows, full_causal_attention_pairs=total_pairs,
        loop_D_rows=length * (depth - cut), token_D_rows=(length - prefix) * depth,
        loop_D_pairs=pairs * (depth - cut), token_D_pairs=(pairs - prefix_pairs) * depth,
        loop_D_row_fraction=(depth - cut) / depth,
        token_D_row_fraction=(length - prefix) / length,
        loop_D_pair_fraction=(depth - cut) / depth,
        token_D_pair_fraction=(pairs - prefix_pairs) / pairs,
        scope='Full causal recurrent attention only. Excludes Prelude/Coda, kernel padding, softmax details, migration, and scheduler costs.')


def fixed_projection_rows(length, depth, cut, prefix, tile=256):
    """Count padded rows for one fixed-tile projection per physical layer.

    Assume one complete prompt/suffix execution per depth, without co-batching.
    This does not count actual kernels, other projections, or FLOPs.
    """
    account(length, depth, cut, prefix)
    if tile <= 0:
        raise ValueError('Projection tile must be positive')
    padded = lambda rows: ((rows + tile - 1) // tile) * tile
    return dict(loop_rows=padded(length) * (depth - cut),
                token_rows=padded(length - prefix) * depth,
                evidence='STRUCTURAL_FIXED_PROJECTION_ROW_ACCOUNTING',
                tile=tile)
