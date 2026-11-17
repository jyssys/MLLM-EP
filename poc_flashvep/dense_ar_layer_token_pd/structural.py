"""Count exact causal pairs and dense projection work. Do not infer latency."""
from .common import ROOT, write
from .token_axis import matched_token_plan


def counts(length, layers, suffix):
    hidden, intermediate, q_heads, kv_heads, head = 4096, 12288, 32, 8, 128
    per_row_GEMM = (2 * hidden * (q_heads + 2 * kv_heads) * head +
                   2 * q_heads * head * hidden + 6 * hidden * intermediate)
    full_pairs = length * (length + 1) // 2
    prefix = length - suffix
    suffix_pairs = full_pairs - prefix * (prefix + 1) // 2
    return dict(layer_D_GEMM_rows=length * layers, token_D_GEMM_rows=suffix * 36,
        layer_D_GEMM_FLOPs=length * layers * per_row_GEMM,
        token_D_GEMM_FLOPs=suffix * 36 * per_row_GEMM,
        layer_D_attention_pairs=full_pairs * layers,
        token_D_attention_pairs=suffix_pairs * 36,
        layer_D_QK_PV_FLOPs=full_pairs * layers * 4 * q_heads * head,
        token_D_QK_PV_FLOPs=suffix_pairs * 36 * 4 * q_heads * head,
        layer_valid_KV_bytes=length * (36 - layers) * 2 * kv_heads * head * 2,
        token_valid_KV_bytes=prefix * 36 * 2 * kv_heads * head * 2,
        layer_hidden_residual_bytes=length * hidden * 2 * 2,
        token_hidden_residual_bytes=0)


def main():
    results = []
    for layers in (2, 4, 8, 12):
        plan = matched_token_plan(layers, 144)
        values = [counts(8000, layers, suffix) for suffix in plan]
        total = {key: sum(row[key] for row in values) for key in values[0]}
        assert total['layer_D_GEMM_rows'] == total['token_D_GEMM_rows']
        assert total['layer_valid_KV_bytes'] == total['token_valid_KV_bytes']
        results.append(dict(layers=layers, requests=144, token_suffix_histogram={
            str(size): plan.count(size) for size in sorted(set(plan))}, totals=total,
            token_to_layer_attention_pairs=total['token_D_attention_pairs'] / total['layer_D_attention_pairs']))
    write(ROOT / 'analysis/matched_GEMM_rows_structural_accounting.json', dict(results=results,
        evidence='STRUCTURAL_ACCOUNTING',
        scope='Multiply-add counts as two FLOPs. QK/PV exact causal pair count. Excludes softmax, norms, RoPE, launch, allocator, protocol headers and padding. Equal GEMM rows imply equal logical prefix KV bytes, not equal attention work, transfer completion or GPU time.'))
    print('STRUCTURAL_ACCOUNTING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
