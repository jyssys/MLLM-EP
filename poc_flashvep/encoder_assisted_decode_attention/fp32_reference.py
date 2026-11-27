"""Check physical BF16 FA3 against independent FP32 GQA on post-RoPE inputs."""
import argparse
import json
import math
import os
from pathlib import Path

import torch

from .operator_bench import compare, fixture_data, generated_data, pack, partition
from .peer_attention import AssistedAttention, PeerAccess


LIMITS = {"relative_l2": .008, "cosine": .9999, "lse_max_abs": 2e-4}


def dense_reference(query, rows):
    """One latest query per sequence accesses every supplied valid KV row."""
    outputs, normalizers = [], []
    for q, (k, v) in zip(query, rows, strict=True):
        kv_heads = k.shape[1]
        groups = q.shape[0] // kv_heads
        q = q.float().reshape(kv_heads, groups, q.shape[-1])
        k = k.float().permute(1, 2, 0)
        v = v.float().transpose(0, 1)
        scores = torch.bmm(q, k) * (q.shape[-1] ** -.5)
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        output = torch.bmm((scores - lse).exp(), v)
        outputs.append(output.flatten(0, 1))
        normalizers.append(lse.flatten())
    return torch.stack(outputs), torch.stack(normalizers, dim=1)


def fp64_normalizer(query, rows):
    normalizers = []
    for q, (keys, _) in zip(query, rows, strict=True):
        q = q.double().reshape(keys.shape[1], -1, q.shape[-1])
        scores = torch.einsum("ghd,lgd->ghl", q, keys.double()) / math.sqrt(q.shape[-1])
        normalizers.append(torch.logsumexp(scores, dim=-1).flatten())
    return torch.stack(normalizers, dim=1)


def ulp_distance(a, b):
    """Distances between finite FP32 values, with both zero signs at rank zero."""
    def rank(x):
        bits = x.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(bits < 0, -(bits & 0x7fffffff), bits)

    return (rank(a) - rank(b)).abs()


def lse_diagnostic(actual, fp32, native, fp64):
    def gap(other):
        distances = ulp_distance(actual, other)
        return {"max_abs": float((actual - other).abs().max()), "max_ulp": int(distances.max()),
                "count_different": int((distances != 0).sum())}

    error = (actual - fp32).abs()
    failing = error > LIMITS["lse_max_abs"]
    distances = ulp_distance(actual, fp32)
    head, row = divmod(int(error.argmax()), fp32.shape[1])
    worst = (head, row)
    return {
        "vs_fp32": gap(fp32), "vs_native_fa3": gap(native),
        "vs_fp64_rounded_to_fp32": gap(fp64.float()),
        "vs_fp64_max_abs": float((actual.double() - fp64).abs().max()),
        "strict_fp32_failure_entries": int(failing.sum()),
        "strict_fp32_failure_entries_one_ulp": int((failing & (distances == 1)).sum()),
        "worst_fp32_entry": {"query_head": head, "batch_row": row, "actual": float(actual[worst]),
                              "native_fa3": float(native[worst]), "fp32": float(fp32[worst]),
                              "fp64": float(fp64[worst]), "fp64_rounded_to_fp32": float(fp64.float()[worst]),
                              "ulp_distance": int(distances[worst])},
    }


def check(reference, expected_lse, actual, actual_lse, *, lse_comparators=None):
    metrics = compare(reference, actual)
    metrics.pop("bitwise")
    lse_error = (expected_lse - actual_lse).abs()
    metrics["lse_max_abs"] = float(lse_error.max())
    metrics["pass"] = (
        metrics["relative_l2"] <= LIMITS["relative_l2"]
        and metrics["cosine"] >= LIMITS["cosine"]
        and math.isfinite(metrics["lse_max_abs"])
        and metrics["lse_max_abs"] <= LIMITS["lse_max_abs"]
    )
    if lse_comparators is not None:
        metrics["lse_diagnostic"] = lse_diagnostic(actual_lse, expected_lse, *lse_comparators)
    return metrics


def evaluate(geometry, provenance, decode, helper, peer, ratios, *, diagnose_lse=False):
    q, data, visual = geometry
    reference, expected_lse = dense_reference(q, data)
    op = AssistedAttention(tuple(q.shape), decode_device=decode, helper_device=helper, peer=peer)
    full = pack(data, decode, "paged")
    native, native_lse = op.local(q, full, lse=True)
    lse_comparators, lse_context = None, None
    if diagnose_lse:
        fp64 = fp64_normalizer(q, data)
        lse_comparators = native_lse, fp64
        lse_context = {"entries": expected_lse.numel(), "query_abs_max": float(q.abs().max()),
                       "key_abs_max": max(float(keys.abs().max()) for keys, _ in data),
                       "ranges": {name: [float(values.min()), float(values.max())] for name, values in
                                  (("fp32", expected_lse), ("fp64", fp64), ("native_fa3", native_lse))},
                       "fp32_vs_fp64_max_abs": float((expected_lse.double() - fp64).abs().max())}
    checked = {"local_paged_fa3": check(reference, expected_lse, native, native_lse, lse_comparators=lse_comparators)}

    policies = ("generic_prefix", "visual_first", "equal_count_prefix") if visual else ("generic_prefix",)
    partitions = []
    for ratio in ratios:
        for policy in policies:
            local_rows, helper_rows = partition(data, ratio, visual, policy)
            local = pack(local_rows, decode, "paged")
            remote = pack(helper_rows, helper, "paged")
            if not min(remote.lengths):
                raise ValueError("This physical witness requires at least one helper KV row per query")
            torch.cuda.synchronize(helper)
            split = op.split(q, local, remote)
            _, local_lse = op.local_part(q, local)
            eager = check(reference, expected_lse, split, torch.logaddexp(local_lse, op.remote_lse),
                          lse_comparators=lse_comparators)

            graph = op.graph(q, local, remote)
            try:
                split = graph()
                # The native merge does not expose its LSE. Recompose it from
                # the captured D normalizer and the E normalizer copied to D.
                graph_lse = torch.logaddexp(graph.buffers[1], op.remote_lse)
                captured = check(reference, expected_lse, split, graph_lse, lse_comparators=lse_comparators)
            finally:
                torch.cuda.synchronize(decode)
                graph.close()
            partitions.append({"policy": policy, "requested_ratio": ratio, "decode_rows": local.lengths,
                               "helper_rows": remote.lengths, "eager": eager, "cuda_graph_dma": captured})
    result = {"input": provenance, "kv_rows": full.lengths, "local": checked, "partitioned": partitions}
    if lse_context is not None:
        result["lse_diagnostic_context"] = lse_context
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--controlled", action="store_true", help="Include a B8 × 32K controlled shape")
    parser.add_argument("--decode-device", type=int, default=1)
    parser.add_argument("--helper-device", type=int, default=0)
    parser.add_argument("--ratios", type=float, nargs="+", default=[.5])
    parser.add_argument("--diagnose-lse", action="store_true", help="Compare LSE magnitudes, native FA3, and independent FP64")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.fixture or args.controlled) or not all(0 < r <= 1 for r in args.ratios):
        parser.error("Select an input and ratios in (0, 1]")
    if args.decode_device == args.helper_device:
        parser.error("Use two distinct physical GPUs")

    torch.set_float32_matmul_precision("highest")
    decode = torch.device("cuda", args.decode_device)
    helper = torch.device("cuda", args.helper_device)
    outcomes = []
    with torch.cuda.device(decode):
        peer = PeerAccess(decode.index, helper.index)
        if args.fixture:
            geometry, provenance = fixture_data(args.fixture, decode)
            outcomes.append(evaluate(geometry, provenance, decode, helper, peer, args.ratios, diagnose_lse=args.diagnose_lse))
            del geometry
        if args.controlled:
            seed = 20261128
            geometry = generated_data(8, 32768, (28, 4, 128), decode, torch.Generator(device=decode).manual_seed(seed))
            outcomes.append(evaluate(geometry, {"source": "controlled_bf16", "seed": seed}, decode, helper, peer, args.ratios,
                                     diagnose_lse=args.diagnose_lse))
    checks = [metrics for row in outcomes for metrics in row["local"].values()]
    checks.extend(item[mode] for row in outcomes for item in row["partitioned"] for mode in ("eager", "cuda_graph_dma"))
    success = all(row["pass"] for row in checks)
    with args.out.open("x") as raw:
        json.dump({"evidence": "CORRECTNESS_ONLY", "passed": success, "checks": len(checks),
                   "reference": "FP32 QK scale, logsumexp normalization and probability V from the same BF16 inputs",
                   "normalizer": "local FA3 LSE or D/E LSE recomposed on D with logaddexp",
                   "thresholds": LIMITS, "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "decode_device": str(decode), "helper_device": str(helper), "results": outcomes,
                   "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES")}, raw, indent=2, allow_nan=False)
        raw.write("\n")
    print(f"FP32 operator witness checks {sum(row['pass'] for row in checks)}/{len(checks)}", flush=True)
    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
