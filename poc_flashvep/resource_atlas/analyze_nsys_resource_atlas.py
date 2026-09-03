#!/usr/bin/env python3
"""Version-tolerant, GUI-free analysis for an Nsight Systems SQLite export.

The profiler schema changes between Nsight releases.  This script therefore
discovers tables/columns at runtime, resolves StringIds when available, and
never invents hardware utilisation numbers that are not present in the trace.
The main capture in this PoC is an exact-route DeepEP replay; absent NVTX
records are reported explicitly rather than being treated as full-serving
phase measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PHASES = [
    ("VISION_PATCH", "prefill"), ("VISION_ATTN", "prefill"),
    ("VISION_MLP", "prefill"), ("VISION_MERGER", "prefill"),
    ("LLM_PREFILL", "prefill"), ("LLM_QKV", "prefill"),
    ("LLM_ATTN", "prefill"), ("LLM_O_PROJ", "prefill"),
    ("TP_COMM", "prefill"), ("ROUTER_TOPK", "prefill"),
    ("DEEPEP_LAYOUT", "prefill"), ("DEEPEP_DISPATCH", "prefill"),
    ("EXPERT_GEMM", "prefill"), ("DEEPEP_COMBINE", "prefill"),
    ("LLM_DECODE", "decode"), ("DECODE_ATTN", "decode"),
    ("DECODE_DISPATCH", "decode"), ("DECODE_EXPERT", "decode"),
    ("DECODE_COMBINE", "decode"),
]


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_info(conn: sqlite3.Connection) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for (name,) in conn.execute("select name from sqlite_master where type='table'"):
        out[name] = [r[1] for r in conn.execute(f"pragma table_info({ident(name)})")]
    return out


def choose_table(tables: dict[str, list[str]], required: set[str], hint: str) -> str | None:
    choices = []
    for name, cols in tables.items():
        low = {c.lower() for c in cols}
        if required <= low:
            score = (3 if hint in name.upper() else 0) + len(required)
            choices.append((score, name))
    return sorted(choices, reverse=True)[0][1] if choices else None


def resolve_names(conn: sqlite3.Connection, tables: dict[str, list[str]]) -> dict[int, str]:
    for table in ("StringIds", "StringIDs", "StringId"):
        if table in tables:
            cols = {c.lower(): c for c in tables[table]}
            if "id" in cols and "value" in cols:
                return {int(i): str(v) for i, v in conn.execute(
                    f"select {ident(cols['id'])},{ident(cols['value'])} from {ident(table)}")}
    return {}


def name_column(cols: list[str]) -> str | None:
    for c in ("demangledName", "shortName", "name", "kernelName", "mangledName"):
        if c in cols:
            return c
    lower = {c.lower(): c for c in cols}
    for c in ("demangledname", "shortname", "kernelname", "name", "mangledname"):
        if c in lower:
            return lower[c]
    return None


def load_kernel_rows(conn: sqlite3.Connection, tables: dict[str, list[str]], names: dict[int, str]) -> tuple[str | None, list[dict[str, Any]]]:
    table = choose_table(tables, {"start", "end"}, "KERNEL")
    if table is None:
        return None, []
    cols = tables[table]
    ncol = name_column(cols)
    select_cols = [c for c in ("start", "end", "deviceId", "streamId", "correlationId", ncol) if c and c in cols]
    rows = []
    for rec in conn.execute(f"select {','.join(ident(c) for c in select_cols)} from {ident(table)}"):
        d = dict(zip(select_cols, rec))
        raw = d.get(ncol) if ncol else None
        d["name"] = names.get(int(raw), str(raw)) if isinstance(raw, (int, float)) else str(raw or "unknown_kernel")
        d["duration_ms"] = (float(d.get("end", 0)) - float(d.get("start", 0))) / 1e6
        rows.append(d)
    return table, rows


def load_memcpy_rows(conn: sqlite3.Connection, tables: dict[str, list[str]]) -> tuple[str | None, list[dict[str, Any]]]:
    table = choose_table(tables, {"start", "end"}, "MEMCPY")
    if table is None:
        return None, []
    cols = tables[table]
    pick = [c for c in ("start", "end", "bytes", "copyKind", "streamId") if c in cols]
    out = []
    for rec in conn.execute(f"select {','.join(ident(c) for c in pick)} from {ident(table)}"):
        d = dict(zip(pick, rec)); d["duration_ms"] = (float(d["end"])-float(d["start"])) / 1e6
        out.append(d)
    return table, out


def classify(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("nvshmem", "deep_ep", "deepep", "alltoall", "nccl", "ucx", "rma", "signal", "notify")):
        return "COMMUNICATION_MIXED"
    if any(x in n for x in ("memcpy", "memset", "copy")):
        return "COPY"
    if any(x in n for x in ("gemm", "matmul", "mm_", "triton", "cutlass", "flash", "attn", "silu", "rmsnorm", "fused")):
        return "COMPUTE_HEAVY"
    return "UNKNOWN"


def dominant(rows: list[dict[str, Any]], limit: int = 3) -> str:
    c = Counter(r["name"] for r in rows)
    return "; ".join(f"{n} ({k})" for n, k in c.most_common(limit))


def replay_phase(name: str) -> str:
    """Conservative name-based phase mapping for the bounded replay."""
    n = name.lower()
    if "layout" in n:
        return "DEEPEP_LAYOUT"
    if "dispatch" in n or "notify_dispatch" in n:
        return "DEEPEP_DISPATCH"
    if "combine" in n or "notify_combine" in n:
        return "DEEPEP_COMBINE"
    if "nccl" in n or "alltoall" in n or "nvshmem" in n:
        return "DEEPEP_COMM_UNKNOWN_PHASE"
    return "BOUNDED_REPLAY_AUXILIARY"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def markdown_table(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n|" + "|".join("---" for _ in fields) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |\n")


def replay_summary(out: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(out.glob("replay_rank*.json")):
        try:
            d = json.loads(p.read_text())
            for metric, vals in d.get("stats", {}).items():
                rows.append({"rank": d.get("rank"), "physical_gpu": d.get("physical_gpu"), "metric": metric, **vals})
        except Exception:
            continue
    return rows


def make_figures(out: Path, kernels: list[dict[str, Any]], sig: list[dict[str, Any]], compat: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out / "figures_unavailable.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return
    # Timeline: observed replay kernels, intentionally labelled as bounded replay.
    if kernels:
        k = sorted(kernels, key=lambda x: x.get("start", 0))
        base = float(k[0].get("start", 0))
        fig, ax = plt.subplots(figsize=(11, 3.5))
        for i, r in enumerate(k[:120]):
            ax.barh(0, r["duration_ms"], left=(float(r.get("start", base))-base)/1e6, height=.6,
                    color="#d95f02" if classify(r["name"]) == "COMMUNICATION_MIXED" else "#1b9e77")
        ax.set(xlabel="relative time (ms)", yticks=[], title="Bounded exact-route DeepEP replay kernel timeline")
        fig.tight_layout(); fig.savefig(out / "figure_phase_timeline.png", dpi=160); plt.close(fig)
    observed = [r for r in sig if float(r.get("cuda_kernel_time_ms", 0) or 0) > 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    if observed:
        ax.bar([r["phase"] for r in observed], [float(r["cuda_kernel_time_ms"]) for r in observed], color="#7570b3")
        ax.tick_params(axis="x", rotation=75)
    ax.set_ylabel("CUDA kernel time (ms)"); ax.set_title("Observed phase duration (missing full-serving phases remain unmeasured)")
    fig.tight_layout(); fig.savefig(out / "figure_phase_duration_breakdown.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["compute", "communication", "copy", "unknown"]
    counts = [sum(1 for r in kernels if classify(r["name"]) == x.upper()+"_HEAVY" if x == "compute") if x == "compute" else
              sum(1 for r in kernels if classify(r["name"]) == ("COMMUNICATION_MIXED" if x == "communication" else "COPY" if x == "copy" else "UNKNOWN")) for x in labels]
    ax.bar(labels, counts, color=["#1b9e77", "#d95f02", "#e6ab02", "#999999"])
    ax.set_ylabel("kernel count"); ax.set_title("Observed resource classes")
    fig.tight_layout(); fig.savefig(out / "figure_resource_class_map.png", dpi=160); plt.close(fig)
    # Compatibility heatmap, with explicit numeric encoding only for the categorical table.
    if compat:
        pairs = [r["pair"] for r in compat]; vals = [{"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}.get(r["RESOURCE_COMPATIBILITY"], 0) for r in compat]
        fig, ax = plt.subplots(figsize=(10, max(3, len(pairs)*.32)))
        ax.imshow([[v] for v in vals], aspect="auto", cmap="RdYlGn", vmin=0, vmax=3)
        ax.set(xticks=[0], xticklabels=["compatibility"], yticks=range(len(pairs)), yticklabels=pairs)
        fig.tight_layout(); fig.savefig(out / "figure_compatibility_heatmap.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(a.sqlite)
    tables = table_info(conn)
    (a.output / "schema_inventory.json").write_text(json.dumps(tables, indent=2) + "\n")
    with (a.output / "sqlite_schema.txt").open("w") as f:
        for name, cols in tables.items(): f.write(f"{name}: {', '.join(cols)}\n")
    names = resolve_names(conn, tables)
    kernel_table, kernels = load_kernel_rows(conn, tables, names)
    memcpy_table, memcpys = load_memcpy_rows(conn, tables)
    nvtx_tables = [n for n, c in tables.items() if "NVTX" in n.upper() or "nvtx" in " ".join(c).lower()]
    replay = replay_summary(a.output)
    (a.output / "replay_rank_summary.json").write_text(json.dumps(replay, indent=2) + "\n")
    # Aggregate only what the trace observes. Main replay has no NVTX records, so
    # source/command evidence is kept separate from observed kernel evidence.
    observed_classes = Counter(classify(r["name"]) for r in kernels)
    comm_rows = [r for r in kernels if classify(r["name"]) == "COMMUNICATION_MIXED"]
    copy_rows = [r for r in kernels if classify(r["name"]) == "COPY"]
    total_cuda = sum(max(0, r["duration_ms"]) for r in kernels)
    fields = ["phase", "prefill_or_decode", "wall_ms", "cuda_kernel_time_ms", "kernel_count", "dominant_kernels",
              "communication_present", "communication_type", "memcpy_present", "resource_class", "evidence_type", "confidence", "notes"]
    sig = []
    phase_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kernels:
        phase_groups[replay_phase(row["name"])].append(row)
    for phase, mode in PHASES:
        rows_for_phase = phase_groups.get(phase, [])
        if rows_for_phase:
            d = sum(r["duration_ms"] for r in rows_for_phase)
            sig.append({"phase": phase, "prefill_or_decode": mode, "wall_ms": 0.0,
                        "cuda_kernel_time_ms": d, "kernel_count": len(rows_for_phase),
                        "dominant_kernels": dominant(rows_for_phase), "communication_present": "YES",
                        "communication_type": "DeepEP kernel-name evidence", "memcpy_present": "NO",
                        "resource_class": "COMMUNICATION_MIXED", "evidence_type": "OBSERVED_BOUNDED_REPLAY",
                        "confidence": "HIGH", "notes": "Name-resolved exact-route replay; not full-serving NVTX."})
        else:
            sig.append({"phase": phase, "prefill_or_decode": mode, "wall_ms": 0.0, "cuda_kernel_time_ms": 0.0,
                        "kernel_count": 0, "dominant_kernels": "", "communication_present": "UNKNOWN",
                        "communication_type": "UNKNOWN", "memcpy_present": "UNKNOWN", "resource_class": "UNKNOWN",
                        "evidence_type": "SOURCE_INFERRED", "confidence": "LOW",
                        "notes": "Full-serving NVTX range not present in this SQLite; no phase latency assigned."})
    # Keep kernels which cannot be safely attributed to dispatch vs combine
    # separate, rather than smearing their duration across both phases.
    for phase in ("DEEPEP_COMM_UNKNOWN_PHASE", "BOUNDED_REPLAY_AUXILIARY"):
        rows_for_phase = phase_groups.get(phase, [])
        if rows_for_phase:
            sig.append({"phase": phase, "prefill_or_decode": "prefill", "wall_ms": 0.0,
                        "cuda_kernel_time_ms": sum(r["duration_ms"] for r in rows_for_phase),
                        "kernel_count": len(rows_for_phase), "dominant_kernels": dominant(rows_for_phase),
                        "communication_present": "YES" if phase == "DEEPEP_COMM_UNKNOWN_PHASE" else "UNKNOWN",
                        "communication_type": "NCCL/collective (phase unresolved)" if phase == "DEEPEP_COMM_UNKNOWN_PHASE" else "UNKNOWN",
                        "memcpy_present": "NO", "resource_class": "COMMUNICATION_MIXED" if phase == "DEEPEP_COMM_UNKNOWN_PHASE" else "UNKNOWN",
                        "evidence_type": "OBSERVED_BOUNDED_REPLAY", "confidence": "MEDIUM",
                        "notes": "Kernel names observed in replay; exact logical sub-phase unresolved."})
    write_csv(a.output / "resource_signature.csv", sig, fields)
    markdown_table(a.output / "resource_signature.md", fields, sig)
    mapping = [{"kernel_name": r["name"], "kernel_count": 1, "duration_ms": round(r["duration_ms"], 6),
                "observed_phase": replay_phase(r["name"]), "mapping_basis": "kernel-name mapping; NVTX absent",
                "resource_class": classify(r["name"])} for r in kernels]
    write_csv(a.output / "phase_kernel_mapping.csv", mapping,
              ["kernel_name", "kernel_count", "duration_ms", "observed_phase", "mapping_basis", "resource_class"])
    pairs = [
        ("VISION_ENCODER+DEEPEP_DISPATCH", "LOW", "CROSS_REQUEST_INDEPENDENT", "MEASURED_NEGATIVE", "Prior paired real run: wall slowdown 12.4%, communication slowdown 19.0%; not a positive candidate."),
        ("VISION_ENCODER+DEEPEP_COMBINE", "LOW", "CROSS_REQUEST_INDEPENDENT", "MEASURED_NEGATIVE", "Prior paired real run: wall slowdown 5.0%, communication slowdown 14.0%; not a positive candidate."),
        ("VISION_ENCODER+EXPERT_GEMM", "LOW", "CROSS_REQUEST_INDEPENDENT", "MEASURED_NEGATIVE", "Prior paired real run: wall slowdown 8.9%; compute contention."),
        ("VISION_ATTN+DEEPEP_DISPATCH", "MEDIUM", "CROSS_REQUEST_INDEPENDENT", "INFERRED", "Different request and communication phase; resource complementarity is plausible but unvalidated."),
        ("VISION_MLP+DEEPEP_DISPATCH", "MEDIUM", "CROSS_REQUEST_INDEPENDENT", "INFERRED", "Compute plus communication may be complementary; HBM/SM contention is unknown."),
        ("LLM_ATTN+DEEPEP_DISPATCH", "MEDIUM", "HARD_DEPENDENCY", "INFERRED", "Same request has ordering dependency; cross-request only is conditional."),
        ("LLM_ATTN+DEEPEP_COMBINE", "MEDIUM", "HARD_DEPENDENCY", "INFERRED", "Same request combine follows expert/dispatch dependencies."),
        ("TP_COMM+DEEPEP_DISPATCH", "LOW", "HARD_DEPENDENCY", "INFERRED", "Potentially shared communication fabric and ordering."),
        ("CPU_SCHEDULER+DEEPEP_DISPATCH", "MEDIUM", "CROSS_REQUEST_INDEPENDENT", "INFERRED", "CPU orchestration can overlap only if it does not introduce queueing jitter."),
        ("DECODE_ATTN+DEEPEP_DISPATCH", "MEDIUM", "CONDITIONAL", "INFERRED", "Cross-request independent, but both may compete for GPU memory/SM resources."),
        ("VISION_MERGER+DEEPEP_COMBINE", "MEDIUM", "CROSS_REQUEST_INDEPENDENT", "INFERRED", "Short projector/merger unit; requires bounded validation."),
    ]
    crows = []
    for pair, comp, dep, evidence, reason in pairs:
        crows.append({"pair": pair, "RESOURCE_COMPATIBILITY": comp, "DEPENDENCY": dep,
                      "OVERLAP_CANDIDATE": "NO" if evidence == "MEASURED_NEGATIVE" or dep == "HARD_DEPENDENCY" else ("MAYBE" if comp == "MEDIUM" else "NO"),
                      "reason": reason, "confidence": "HIGH" if evidence == "MEASURED_NEGATIVE" else "LOW", "evidence": evidence})
    cf = ["pair", "RESOURCE_COMPATIBILITY", "DEPENDENCY", "OVERLAP_CANDIDATE", "reason", "confidence", "evidence"]
    write_csv(a.output / "resource_compatibility_matrix.csv", crows, cf)
    markdown_table(a.output / "resource_compatibility_matrix.md", cf, crows)
    dep = {"phase_dependency_classes": {
        "VISION_ENCODER->LLM_PREFILL": "HARD_DEPENDENCY",
        "LLM_PREFILL->DEEPEP_DISPATCH": "HARD_DEPENDENCY",
        "DEEPEP_DISPATCH->EXPERT_GEMM": "HARD_DEPENDENCY",
        "EXPERT_GEMM->DEEPEP_COMBINE": "HARD_DEPENDENCY",
        "pending_request_VISION_ENCODER->current_request_DEEPEP_COMM": "CROSS_REQUEST_INDEPENDENT",
        "image_i_encoder->image_j_encoder": "CROSS_IMAGE_POSSIBLE",
    }, "notes": "Classes reflect source audit and cross-request semantics; absent full-serving NVTX prevents timing-derived dependency proof."}
    (a.output / "dependency_graph.json").write_text(json.dumps(dep, indent=2) + "\n")
    (a.output / "dependency_graph.md").write_text("# Dependency graph\n\n" + "\n".join(f"- `{k}`: **{v}**" for k, v in dep["phase_dependency_classes"].items()) + "\n\n" + dep["notes"] + "\n")
    (a.output / "overlap_candidate_shortlist.md").write_text("""# Overlap candidate shortlist\n\nCandidates below are conditional/inferred only; prior encoder+DeepEP and encoder+expert measurements are explicitly negative.\n\n1. **CPU scheduler/request preparation + DeepEP dispatch** — CPU-side work is cross-request independent and has low direct GPU-resource overlap risk. Evidence: INFERRED. Risk: host scheduling may not cover the communication window. Cheapest next PoC: NVTX-marked CPU preparation with one exact-route dispatch.\n2. **Small Vision merger/projector + DeepEP combine** — short memory-oriented encoder tail versus communication. Evidence: INFERRED. Risk: the merger may share HBM/L2 and the prior full encoder+combine pair was negative. Cheapest next PoC: one natural merger unit with the exact replay.\n3. **Decode attention + DeepEP dispatch** — cross-request independent but conditional. Evidence: INFERRED. Risk: decode attention is latency-sensitive and can contend for HBM. Cheapest next PoC: one mixed prefill/decode trace.\n\nNo candidate is labelled HIGH: the available real paired measurements are MEASURED_NEGATIVE, and the bounded replay has no full-serving NVTX ranges.\n""")
    (a.output / "visual_streaming_feasibility.md").write_text("""# Visual streaming feasibility\n\n**POSSIBLE_WITH_RUNTIME_CHANGE**. Qwen3-VL concatenates image patch sequences with `grid_thw`/`cu_seqlens` and runs the vision transformer before merging embeddings into the language sequence. Image sequences are structurally separated, but the current vLLM path returns the complete visual embedding tensor before LM execution and has no per-image ready callback. Exposing image-level completion would require runtime scheduling/interface work; no implementation was added.\n""")
    summary = {"sqlite": str(a.sqlite), "kernel_table": kernel_table, "kernel_rows": len(kernels), "memcpy_table": memcpy_table,
               "memcpy_rows": len(memcpys), "nvtx_tables": nvtx_tables, "nvtx_present": bool(nvtx_tables),
               "string_ids_resolved": len(names), "total_kernel_ms": total_cuda, "observed_resource_classes": dict(observed_classes),
               "replay_rank_files": len(replay), "full_serving_phase_mapping": "NOT_AVAILABLE_NO_NVTX",
               "replay_phase_groups": {k: len(v) for k, v in phase_groups.items()}}
    (a.output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figures(a.output, kernels, sig, crows)


if __name__ == "__main__":
    main()
