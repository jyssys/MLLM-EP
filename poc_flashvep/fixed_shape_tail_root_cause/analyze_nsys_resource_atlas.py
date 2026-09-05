#!/usr/bin/env python3
"""Version-tolerant SQLite summary for a bounded Nsight Systems capture.

The script intentionally uses only table/column discovery plus the stable
StringIds join.  It does not assume a particular Nsight report schema.
"""
from __future__ import annotations
import argparse, csv, json, sqlite3
from pathlib import Path


def tables(con):
    return [r[0] for r in con.execute("select name from sqlite_master where type='table'")]


def classify(name: str) -> str:
    n = name.lower()
    if "deep_ep::intranode::dispatch" in n or "notify_dispatch" in n:
        return "DEEPEP_DISPATCH"
    if "deep_ep::intranode::combine" in n or "notify_combine" in n or "cached_notify_combine" in n:
        return "DEEPEP_COMBINE"
    if "deep_ep::layout::get_dispatch_layout" in n:
        return "DEEPEP_LAYOUT"
    if "fused_moe_kernel" in n or "cutlass" in n and "gemm" in n:
        return "EXPERT"
    if "topkgating" in n or "topk" in n or "count_and_sort_expert" in n:
        return "ROUTER_TOPK"
    if "nccl" in n or "cross_device_reduce" in n or "allgather" in n or "reducescatter" in n:
        return "TP_OR_COLLECTIVE"
    if "flashattn" in n or "flash::" in n and "fwd" in n:
        return "ATTENTION"
    if "memcpy" in n or "copy" in n:
        return "COPY"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(a.sqlite)
    ts = tables(con)
    (a.out / "schema_tables.json").write_text(json.dumps(ts, indent=2) + "\n")
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in ts or "StringIds" not in ts:
        raise SystemExit("kernel/StringIds tables not present")
    cols = [r[1] for r in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    required = {"start", "end", "demangledName"}
    if not required.issubset(cols):
        raise SystemExit(f"kernel schema lacks {required - set(cols)}; columns={cols}")
    q = """select s.value, count(*) n, sum(k.end-k.start) total_ns,
                   min(k.start) first_ns, max(k.end) last_ns
            from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s
              on k.demangledName=s.id group by s.value order by total_ns desc"""
    rows = []
    for name, n, total, first, last in con.execute(q):
        rows.append({"kernel": name, "count": int(n), "kernel_ms": float(total or 0)/1e6,
                     "first_ns": int(first), "last_ns": int(last),
                     "category": classify(name)})
    with (a.out / "kernel_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kernel", "count", "kernel_ms", "first_ns", "last_ns", "category"])
        w.writeheader(); w.writerows(rows)
    sums = {}
    for r in rows:
        s = sums.setdefault(r["category"], {"category": r["category"], "kernel_count": 0, "kernel_ms": 0.0, "dominant_kernel": r["kernel"], "_max_kernel_ms": -1.0})
        s["kernel_count"] += r["count"]; s["kernel_ms"] += r["kernel_ms"]
        if r["kernel_ms"] > s["_max_kernel_ms"]:
            s["dominant_kernel"] = r["kernel"]; s["_max_kernel_ms"] = r["kernel_ms"]
    with (a.out / "phase_kernel_mapping.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["phase", "kernel_count", "kernel_ms", "dominant_kernel"])
        def clean(k, v):
            return {"phase": k, "kernel_count": v["kernel_count"], "kernel_ms": v["kernel_ms"], "dominant_kernel": v["dominant_kernel"]}
        w.writeheader(); w.writerows(sorted((clean(k, v) for k,v in sums.items()), key=lambda x: -x["kernel_ms"]))
    (a.out / "summary.json").write_text(json.dumps({"kernel_rows": len(rows), "categories": sums, "tables": ts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
