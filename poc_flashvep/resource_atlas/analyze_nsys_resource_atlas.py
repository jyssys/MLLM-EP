#!/usr/bin/env python3
"""Version-tolerant, GUI-free analysis for an Nsight Systems SQLite export.

The profiler schema changes between Nsight releases.  This script therefore
discovers tables/columns at runtime, resolves StringIds when available, and
never invents hardware utilisation numbers that are not present in the trace.
The parser accepts both exact-route replay and full-serving captures.  When
NVTX ranges are present, CUDA kernels are assigned to the shortest enclosing
range in the same Nsight clock domain; otherwise it falls back to conservative
kernel-name mapping and labels the result accordingly.
"""
from __future__ import annotations

import argparse
import bisect
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
    # globalPid is essential for multi-process vLLM captures: CUDA activity is
    # emitted by child workers while NVTX ranges are associated with worker
    # threads.  Keeping it lets the mapper avoid assigning one worker's
    # kernel to another worker's same-time range.
    select_cols = [c for c in ("start", "end", "deviceId", "streamId", "correlationId", "globalPid", ncol) if c and c in cols]
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


def load_nvtx_rows(conn: sqlite3.Connection, tables: dict[str, list[str]]) -> tuple[str | None, list[dict[str, Any]]]:
    table = None
    for name, cols in tables.items():
        low = {c.lower() for c in cols}
        if "start" in low and "end" in low and "text" in low and "NVTX" in name.upper():
            table = name; break
    if table is None:
        return None, []
    cols = tables[table]
    pick = [c for c in ("start", "end", "eventType", "text", "globalTid", "domainId") if c in cols]
    out = []
    for rec in conn.execute(f"select {','.join(ident(c) for c in pick)} from {ident(table)}"):
        d = dict(zip(pick, rec))
        if d.get("end") is None or not d.get("text"):
            continue
        # 59 is the range event in Nsight 2024.6; accepting all non-null
        # event types keeps this reader tolerant of adjacent releases.
        d["duration_ms"] = (float(d["end"])-float(d["start"])) / 1e6
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
    if "topkgating" in n or "topk" in n:
        return "ROUTER_TOPK"
    if "fused_moe_kernel" in n or "moe_kernel" in n:
        return "EXPERT_GEMM"
    if "cross_device_reduce" in n or "all_reduce" in n:
        return "TP_COMM"
    if "flashattn" in n or "nvjet_tst" in n:
        return "LLM_ATTN"
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


def make_figures(out: Path, kernels: list[dict[str, Any]], sig: list[dict[str, Any]], compat: list[dict[str, Any]], full: bool = False) -> None:
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
        ax.set(xlabel="relative time (ms)", yticks=[], title=("Full-serving CUDA kernel timeline" if full else "Bounded exact-route DeepEP replay kernel timeline"))
        fig.tight_layout(); fig.savefig(out / "figure_phase_timeline.png", dpi=160); plt.close(fig)
    observed = [r for r in sig if float(r.get("cuda_kernel_time_ms", 0) or 0) > 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    if observed:
        ax.bar([r["phase"] for r in observed], [float(r["cuda_kernel_time_ms"]) for r in observed], color="#7570b3")
        ax.tick_params(axis="x", rotation=75)
    ax.set_ylabel("CUDA kernel time (ms)"); ax.set_title("Observed phase duration" if full else "Observed replay phase duration")
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
    full = "full_serving" in a.sqlite.name or "qwen3vl_full" in a.output.name
    conn = sqlite3.connect(a.sqlite)
    tables = table_info(conn)
    (a.output / "schema_inventory.json").write_text(json.dumps(tables, indent=2) + "\n")
    with (a.output / "sqlite_schema.txt").open("w") as f:
        for name, cols in tables.items(): f.write(f"{name}: {', '.join(cols)}\n")
    names = resolve_names(conn, tables)
    kernel_table, kernels = load_kernel_rows(conn, tables, names)
    memcpy_table, memcpys = load_memcpy_rows(conn, tables)
    nvtx_table, nvtx_rows = load_nvtx_rows(conn, tables)
    nvtx_tables = [n for n, c in tables.items() if "NVTX" in n.upper() or "nvtx" in " ".join(c).lower()]
    replay = replay_summary(a.output)
    (a.output / "replay_rank_summary.json").write_text(json.dumps(replay, indent=2) + "\n")
    # Aggregate only what the trace observes. Full-serving phase rows use the
    # shortest enclosing NVTX range when available.
    observed_classes = Counter(classify(r["name"]) for r in kernels)
    comm_rows = [r for r in kernels if classify(r["name"]) == "COMMUNICATION_MIXED"]
    copy_rows = [r for r in kernels if classify(r["name"]) == "COPY"]
    total_cuda = sum(max(0, r["duration_ms"]) for r in kernels)
    fields = ["phase", "prefill_or_decode", "wall_ms", "cuda_kernel_time_ms", "kernel_count", "dominant_kernels",
              "communication_present", "communication_type", "memcpy_present", "resource_class", "evidence_type", "confidence", "notes"]
    sig = []
    phase_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nvtx_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for n in nvtx_rows:
        if str(n.get("text")) in {p for p, _ in PHASES}:
            nvtx_by_phase[str(n["text"])].append(n)
    # Nsight's globalTid is encoded in the same clock-domain namespace as
    # globalPid, with the native thread id added.  For worker processes the
    # nearest observed CUDA globalPid is therefore a reliable process key
    # (workers are separated by tens of millions in this encoding).  The
    # process key prevents cross-worker NVTX/kernel attribution when multiple
    # DP/TP children execute concurrently.
    kernel_pids = sorted({int(r["globalPid"]) for r in kernels if r.get("globalPid") is not None})
    for n in nvtx_rows:
        tid = n.get("globalTid")
        if tid is None or not kernel_pids:
            n["globalPid"] = None
            continue
        tid = int(tid)
        pid = min(kernel_pids, key=lambda p: abs(tid - p))
        # If this is not close to a traced CUDA process, leave it unassigned
        # rather than making a false phase claim.
        n["globalPid"] = pid if abs(tid - pid) < 10_000_000 else None
    phase_ranges = {
        (pid, phase): sorted((float(n["start"]), float(n["end"]), n) for n in ranges if n.get("globalPid") == pid)
        for phase, ranges in nvtx_by_phase.items()
        for pid in sorted({n.get("globalPid") for n in ranges if n.get("globalPid") is not None})
    }
    phase_starts = {key: [x[0] for x in ranges] for key, ranges in phase_ranges.items()}

    def has_outer_decode(row: dict[str, Any]) -> bool:
        """Whether a kernel is inside the decoder's decode-only range."""
        pid = row.get("globalPid")
        starts = phase_starts.get((pid, "LLM_DECODE"), [])
        ranges = phase_ranges.get((pid, "LLM_DECODE"), [])
        if not starts:
            return False
        ks, ke = float(row["start"]), float(row["end"])
        j = bisect.bisect_right(starts, ks) - 1
        for idx in range(j, max(-1, j - 32), -1):
            st, en, _ = ranges[idx]
            # Communication kernels are asynchronous with respect to the
            # Python/NVTX wrapper and may begin shortly after the range pops.
            if st - 1_000_000 <= ks and en + 5_000_000 >= ke:
                return True
        return False

    decode_phase_alias = {
        "LLM_ATTN": "DECODE_ATTN",
        "DEEPEP_DISPATCH": "DECODE_DISPATCH",
        "EXPERT_GEMM": "DECODE_EXPERT",
        "DEEPEP_COMBINE": "DECODE_COMBINE",
    }
    # Build synthetic decode sub-phase range lists from the nested generic
    # markers emitted by the installed hook.  This preserves the observed
    # CUDA interval while keeping prefill and decode totals separate.
    nvtx_ranges_for_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for phase, ranges in nvtx_by_phase.items():
        nvtx_ranges_for_phase[phase].extend(ranges)
    for base_phase, decode_phase in decode_phase_alias.items():
        for pid, base_ranges in ((pid, ranges) for (pid, phase), ranges in phase_ranges.items() if phase == base_phase):
            decode_ranges = phase_ranges.get((pid, "LLM_DECODE"), [])
            for st, en, event in base_ranges:
                if any(dst <= st and den >= en for dst, den, _ in decode_ranges):
                    nvtx_ranges_for_phase[decode_phase].append(event)
    for row in kernels:
        # The most recently started range is normally the innermost range.
        # Looking backwards is bounded because adjacent vLLM ranges are
        # sequential; this avoids an O(kernels * NVTX-events) scan on large
        # serving traces while remaining tolerant of nested ranges.
        candidates = []
        ks, ke = float(row["start"]), float(row["end"])
        row_pid = row.get("globalPid")
        for (pid, phase), ranges in phase_ranges.items():
            if row_pid is not None and pid != row_pid:
                continue
            starts = phase_starts[(pid, phase)]
            j = bisect.bisect_right(starts, ks) - 1
            for idx in range(j, max(-1, j - 128), -1):
                st, en, event = ranges[idx]
                if en >= ke:
                    candidates.append((en - st, event))
                    break
        name_phase = replay_phase(row["name"])
        # DeepEP/router/expert kernels have stable names and are launched on
        # asynchronous streams.  Prefer that direct kernel evidence over a
        # CPU NVTX containment test (which can otherwise assign a late
        # communication kernel to the enclosing ROUTER range).
        direct_phases = {"DEEPEP_LAYOUT", "DEEPEP_DISPATCH", "DEEPEP_COMBINE",
                         "EXPERT_GEMM", "ROUTER_TOPK", "TP_COMM"}
        if full and name_phase in direct_phases:
            phase = name_phase
        elif candidates:
            phase = min(candidates, key=lambda x: x[0])[1]["text"]
        else:
            phase = name_phase
        # ROUTER_TOPK is a broad Python MoE wrapper in this vLLM release.
        # Keep only the actual top-k kernel in that signature; kernels queued
        # after the wrapper returns are not safe to attribute to routing.
        if full and phase == "ROUTER_TOPK" and "topkgating" not in row["name"].lower():
            phase = "LLM_MOE"
        # The installed hook has a single decoder wrapper and nested generic
        # attention/DeepEP markers.  When those markers sit inside the
        # decoder's LLM_DECODE range, preserve their actual decode identity
        # instead of labelling them as prefill phases.
        if full and phase in decode_phase_alias and has_outer_decode(row):
            phase = decode_phase_alias[phase]
        row["mapped_phase"] = phase
        phase_groups[phase].append(row)
    for phase, mode in PHASES:
        rows_for_phase = phase_groups.get(phase, [])
        ranges_for_phase = nvtx_ranges_for_phase.get(phase, [])

        # Some useful boundaries are not separately wrapped in vLLM 0.20.
        # Keep a diagnostic aggregate based on kernel names for full serving
        # rather than claiming that an absent range was measured.  These
        # kernels are nested in the enclosing range and therefore should not
        # be added to an end-to-end sum twice.
        diagnostic_rows = []
        diagnostic_note = ""
        if full and phase == "DEEPEP_LAYOUT" and not rows_for_phase:
            diagnostic_rows = [r for r in kernels if "get_dispatch_layout" in r["name"].lower()]
            diagnostic_note = "Kernel-name diagnostic; layout is nested in DEEPEP_DISPATCH and not double-counted."
        elif full and phase == "TP_COMM" and not rows_for_phase:
            diagnostic_rows = [r for r in kernels if any(x in r["name"].lower() for x in ("cross_device_reduce", "all_reduce", "allreduce"))]
            diagnostic_note = "Kernel-name diagnostic; cross-device reduction is nested in attention/other ranges and not double-counted."
        if diagnostic_rows:
            rows_for_phase = diagnostic_rows
        if rows_for_phase or ranges_for_phase:
            d = sum(r["duration_ms"] for r in rows_for_phase)
            wall = sum(r["duration_ms"] for r in ranges_for_phase)
            rclasses = Counter(classify(r["name"]) for r in rows_for_phase)
            resource_class = "ROUTING" if phase == "ROUTER_TOPK" else ("COMMUNICATION_MIXED" if rclasses.get("COMMUNICATION_MIXED", 0) else ("COMPUTE_HEAVY" if rclasses.get("COMPUTE_HEAVY", 0) else "UNKNOWN"))
            sig.append({"phase": phase, "prefill_or_decode": mode, "wall_ms": wall,
                        "cuda_kernel_time_ms": d, "kernel_count": len(rows_for_phase),
                        "dominant_kernels": dominant(rows_for_phase), "communication_present": "YES",
                        "communication_type": "DeepEP/NCCL kernel evidence" if "DEEPEP" in phase or phase == "TP_COMM" else "UNKNOWN",
                        "memcpy_present": "NO", "resource_class": resource_class,
                        "evidence_type": "FULL_SERVING_OBSERVED" if full and (ranges_for_phase or diagnostic_rows or rows_for_phase) else "OBSERVED_BOUNDED_REPLAY",
                        "confidence": "HIGH" if ranges_for_phase else "MEDIUM",
                        "notes": ("Shortest enclosing NVTX range + CUDA kernel overlap." if ranges_for_phase else (diagnostic_note or "Kernel-name mapping; no full-serving NVTX range."))})
        else:
            sig.append({"phase": phase, "prefill_or_decode": mode, "wall_ms": 0.0, "cuda_kernel_time_ms": 0.0,
                        "kernel_count": 0, "dominant_kernels": "", "communication_present": "UNKNOWN",
                        "communication_type": "UNKNOWN", "memcpy_present": "UNKNOWN", "resource_class": "UNKNOWN",
                        "evidence_type": "SOURCE_INFERRED", "confidence": "LOW",
                        "notes": "No observed kernel/range for this phase in the capture."})
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
                        "evidence_type": "FULL_SERVING_OBSERVED" if full else "OBSERVED_BOUNDED_REPLAY", "confidence": "MEDIUM",
                        "notes": "Kernel names observed in replay; exact logical sub-phase unresolved."})
    write_csv(a.output / "resource_signature.csv", sig, fields)
    markdown_table(a.output / "resource_signature.md", fields, sig)
    mapping = [{"kernel_name": r["name"], "kernel_count": 1, "duration_ms": round(r["duration_ms"], 6),
                "observed_phase": r.get("mapped_phase", replay_phase(r["name"])), "mapping_basis": "shortest enclosing NVTX range" if nvtx_rows and r.get("mapped_phase") in {p for p, _ in PHASES} else "kernel-name mapping",
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
        if full and evidence == "INFERRED":
            evidence = "FULL_SERVING_OBSERVED"
            reason = reason.replace("requires bounded validation.", "full-serving resource classes are observed; causal overlap remains unvalidated.")
        crows.append({"pair": pair, "RESOURCE_COMPATIBILITY": comp, "DEPENDENCY": dep,
                      "OVERLAP_CANDIDATE": "NO" if evidence == "MEASURED_NEGATIVE" or dep == "HARD_DEPENDENCY" else ("MAYBE" if comp == "MEDIUM" else "NO"),
                      "reason": reason, "confidence": "HIGH" if evidence == "MEASURED_NEGATIVE" else ("MEDIUM" if full else "LOW"), "evidence": evidence})
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
    }, "notes": "Classes reflect source audit and cross-request semantics. Full-serving NVTX/kernel overlap is observed where present; replay-only mappings remain separate."}
    (a.output / "dependency_graph.json").write_text(json.dumps(dep, indent=2) + "\n")
    (a.output / "dependency_graph.md").write_text("# Dependency graph\n\n" + "\n".join(f"- `{k}`: **{v}**" for k, v in dep["phase_dependency_classes"].items()) + "\n\n" + dep["notes"] + "\n")
    shortlist = """# Overlap candidate shortlist\n\nThe list is a shortlist of hypotheses, not an optimization implementation. Prior paired encoder+DeepEP and encoder+expert measurements remain explicitly negative.\n\n1. **CPU scheduler/request preparation + DeepEP dispatch** — CPU-side work is cross-request independent and has low direct GPU-resource overlap risk. Evidence: {evidence}. Risk: host work may not cover the communication window. Cheapest next PoC: one NVTX-marked CPU preparation plus exact-route dispatch.\n2. **Vision merger/projector + DeepEP combine** — a short encoder tail versus a communication phase. Evidence: {evidence}. Risk: the merger shares memory resources and the prior full encoder+combine pair was negative. Cheapest next PoC: one natural merger unit with exact replay.\n3. **Decode attention + DeepEP dispatch** — cross-request independent but conditional. Evidence: {evidence}. Risk: decode attention is latency-sensitive and may contend for HBM. Cheapest next PoC: one mixed prefill/decode trace.\n\nFull-serving CUDA kernels and NVTX phase ranges are now captured; compatibility labels remain conditional because this atlas does not schedule concurrent work.\n""".format(evidence="FULL_SERVING_OBSERVED" if full else "INFERRED")
    (a.output / "overlap_candidate_shortlist.md").write_text(shortlist)
    (a.output / "visual_streaming_feasibility.md").write_text("""# Visual streaming feasibility\n\n**POSSIBLE_WITH_RUNTIME_CHANGE**. Qwen3-VL concatenates image patch sequences with `grid_thw`/`cu_seqlens` and runs the vision transformer before merging embeddings into the language sequence. Image sequences are structurally separated, but the current vLLM path returns the complete visual embedding tensor before LM execution and has no per-image ready callback. Exposing image-level completion would require runtime scheduling/interface work; no implementation was added.\n""")
    summary = {"sqlite": str(a.sqlite), "capture_kind": "FULL_SERVING" if full else "BOUNDED_REPLAY", "kernel_table": kernel_table, "kernel_rows": len(kernels), "memcpy_table": memcpy_table,
               "memcpy_rows": len(memcpys), "nvtx_table": nvtx_table, "nvtx_rows": len(nvtx_rows), "nvtx_tables": nvtx_tables, "nvtx_present": bool(nvtx_rows),
               "string_ids_resolved": len(names), "total_kernel_ms": total_cuda, "observed_resource_classes": dict(observed_classes),
               "replay_rank_files": len(replay), "full_serving_phase_mapping": "NVTX_ENCLOSING_RANGE" if nvtx_rows else "KERNEL_NAME_ONLY_NO_NVTX",
               "replay_phase_groups": {k: len(v) for k, v in phase_groups.items()}}
    (a.output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figures(a.output, kernels, sig, crows, full=full)


if __name__ == "__main__":
    main()
