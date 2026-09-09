"""Same-device kernel interval unions inside parent request-step NVTX ranges.

Nsight correlates CPU and GPU clocks. Never subtract timestamps from different
devices. These instrumented overlap diagnostics are not clean serving timings.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import statistics


def union(spans):
    out = []
    for start, end in sorted(spans):
        assert end >= start
        if out and start <= out[-1][1]:
            out[-1][1] = max(end, out[-1][1])
        else:
            out.append([start, end])
    return out


def length(spans):
    return sum(end-start for start, end in spans)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    p = argparse.ArgumentParser()
    p.add_argument("database", type=Path)
    args = p.parse_args()
    db = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    devices = [dict(row) for row in db.execute("SELECT id,uuid,busLocation,cuDevice FROM TARGET_INFO_GPU")]
    markers = [dict(row) for row in db.execute(
        "SELECT n.start,n.end,COALESCE(n.text,s.value) AS label FROM NVTX_EVENTS n "
        "LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value) "
        "LIKE 'COHORT_TARGET_STEP_%' AND n.end IS NOT NULL ORDER BY n.start")]
    assert len(markers) == 32, len(markers)
    assert all(a["end"] <= b["start"] for a, b in zip(markers, markers[1:]))
    # 0 = target prefill; 1 = decode-plan creation/capture. Exclude both.
    markers = [m for m in markers if int(m["label"].rsplit("_", 1)[-1]) >= 2]
    low, high = markers[0]["start"], markers[-1]["end"]
    kernels = [dict(row) for row in db.execute(
        "SELECT k.start,k.end,k.deviceId,k.streamId,k.greenContextId,k.graphNodeId,s.value AS name "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.end>? AND k.start<? ORDER BY k.start", (low, high))]
    assert kernels and {k["deviceId"] for k in kernels} == {0, 1, 2, 3}
    rows = []
    for marker in markers:
        start, end = marker["start"], marker["end"]
        for device in range(4):
            selected = [k for k in kernels if k["deviceId"] == device and k["end"] > start and k["start"] < end]
            comm = union((max(start, k["start"]), min(end, k["end"])) for k in selected if "nccl" in k["name"].lower())
            comp = union((max(start, k["start"]), min(end, k["end"])) for k in selected if "nccl" not in k["name"].lower())
            combined = union(comm+comp)
            overlap = length(comm)+length(comp)-length(combined)
            rows.append({"step": int(marker["label"].rsplit("_", 1)[-1]), "device_id": device,
                         "request_step_ms_profiled": (end-start)/1e6,
                         "kernel_union_ms": length(combined)/1e6,
                         "compute_union_ms": length(comp)/1e6, "nccl_union_ms": length(comm)/1e6,
                         "compute_nccl_overlap_ms": overlap/1e6,
                         "nccl_overlap_fraction": overlap/length(comm) if length(comm) else None,
                         "uncovered_by_kernels_ms": (end-start-length(combined))/1e6,
                         "reported_graph_node_stream_count": len({k["streamId"] for k in selected}),
                         "kernel_count": len(selected),
                         "graph_kernel_count": sum(bool(k["graphNodeId"]) for k in selected)})
    stem = args.database.with_suffix("")
    with stem.with_suffix(".overlap.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = [k for k in rows[0] if k not in {"step", "device_id"}]
    summary = {"scope": "PROFILED_NATIVE_OVERLAP_NOT_E2E_ORACLE", "physical_gpu_inventory": devices,
               "kernel_device_id_scope": "process-visible ordinals 0-3 under enforced CUDA_VISIBLE_DEVICES=4,5,6,7; inventory IDs are physical and must not be directly joined",
               "steady_steps": len(markers), "same_device_rows": len(rows),
               "medians": {m: statistics.median(r[m] for r in rows) for m in metrics},
               "limitations": ["excludes first prefill and decode capture",
                               "kernel gaps include host scheduling, memcopy, event waits and profiling overhead",
                               "overlap intersection is not removable request latency",
                               "CUDA graph node stream IDs are not counts of source-level independent streams",
                               "manual full-SM plan, not searched optimum"]}
    stem.with_suffix(".overlap.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
