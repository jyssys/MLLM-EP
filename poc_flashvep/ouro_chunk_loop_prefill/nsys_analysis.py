"""Export scoped Nsight events without process environment or private metadata."""

import csv
import hashlib
import json
import sqlite3
from .common import ROOT, write


def export(connection, name, sql):
    cursor = connection.execute(sql)
    target = ROOT / name
    with target.open("w") as output:
        writer = csv.writer(output)
        writer.writerow([column[0] for column in cursor.description])
        count = 0
        for row in cursor:
            writer.writerow(row)
            count += 1
    return count


def union_ns(intervals, begin, end):
    covered = 0
    previous = begin
    for start, stop in sorted(intervals):
        start = max(start, begin, previous)
        stop = min(stop, end)
        if stop > start:
            covered += stop - start
            previous = stop
    return covered


def main():
    database = ROOT / "raw/nsys_grouped.sqlite"
    if not database.exists():
        write(ROOT / "nsys_summary.json", dict(status="NOT_AVAILABLE"))
        return
    c = sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True)
    count = export(
        c,
        "nsys_kernel_timeline.csv",
        "select k.start,k.end,k.deviceId,k.streamId,k.correlationId,k.graphNodeId,s.value as kernel_name "
        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start",
    )
    copies = export(
        c,
        "nsys_copy_timeline.csv",
        "select start,end,deviceId,streamId,correlationId,bytes,copyKind,srcKind,dstKind "
        "from CUPTI_ACTIVITY_KIND_MEMCPY order by start",
    )
    runtime = export(
        c,
        "nsys_runtime_timeline.csv",
        "select r.start,r.end,r.correlationId,s.value as API "
        "from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId order by r.start",
    )
    export(
        c,
        "nsys_kernel_service.csv",
        "select s.value as kernel_name,count(*) as instances,sum(k.end-k.start)/1000000.0 as summed_service_ms "
        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName "
        "group by s.value order by 3 desc",
    )
    markers = c.execute(
        "select start,end,text from NVTX_EVENTS where text like 'CHUNK_LOOP%' order by start"
    ).fetchall()
    raw = json.loads(
        (ROOT / "raw/nsys_grouped_v1/grouped_C1024_P2048_N4_T2.json").read_text()
    )
    assert len(markers) == len(raw["quantums"])
    aligned = []
    for i, ((start, end, text), quantum) in enumerate(zip(markers, raw["quantums"])):
        aligned.append(
            dict(
                quantum=i,
                start_ns=start,
                end_ns=end,
                NVTX=text,
                ids="|".join(s["id"] for s in quantum["spans"]),
                phases="|".join(s["phase"] for s in quantum["spans"]),
                group=max(s["group"] for s in quantum["spans"]),
                active_rows=sum(s["tokens"] for s in quantum["spans"]),
                status="HOST_NVTX_SUBMISSION_NOT_GPU_COMPLETION",
            )
        )
    from .analyze import csvfile

    csvfile("nsys_quantum_alignment.csv", aligned)
    begin, end = markers[0][0], max(row[1] for row in markers)
    intervals = c.execute(
        "select start,end from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?",
        (begin, end),
    ).fetchall()
    mem = c.execute(
        "select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?",
        (begin, end),
    ).fetchall()
    nccl = c.execute(
        "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where lower(s.value) like '%nccl%'"
    ).fetchone()[0]
    summary = dict(
        status="LIVE_MEASURED_HEAVY_DIAGNOSTIC",
        nsys_version="2024.6.2",
        kernels=count,
        copies=copies,
        runtime_APIs=runtime,
        NVTX_quanta=len(markers),
        NCCL_kernels=nccl,
        logical_devices=[
            x[0]
            for x in c.execute(
                "select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL"
            )
        ],
        physical_GPU=4,
        NVTX_host_window_ms=(end - begin) / 1e6,
        kernel_union_ms_in_NVTX_window=union_ns(intervals, begin, end) / 1e6,
        kernel_or_memcpy_union_ms_in_NVTX_window=union_ns(intervals + mem, begin, end)
        / 1e6,
        note="Heavy cold probe only. Union occupancy is not a clean E2E bottleneck attribution. CUDA-event durations include gaps.",
        private_raw_artifacts=[],
    )
    for file in (database, ROOT / "raw/nsys_grouped.nsys-rep"):
        summary["private_raw_artifacts"].append(
            dict(
                path=str(file),
                bytes=file.stat().st_size,
                sha256=hashlib.file_digest(file.open("rb"), "sha256").hexdigest(),
                storage="SERVER_LOCAL_NOT_PUSHED; sanitized complete event CSV exports are archived",
            )
        )
    write(ROOT / "nsys_summary.json", summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(
        row
        for row in aligned
        if "long_" in row["ids"] and "prefilling_recurrent" in row["phases"]
    )
    left = first["start_ns"] - 100_000_000
    right = first["end_ns"] + 350_000_000
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), layout="constrained", sharex=True)
    for row in aligned:
        if row["end_ns"] < left or row["start_ns"] > right:
            continue
        long = "long_" in row["ids"]
        axes[0].broken_barh(
            [((row["start_ns"] - left) / 1e6, (row["end_ns"] - row["start_ns"]) / 1e6)],
            (int(long) - 0.3, 0.6),
            facecolors="#d37b30" if long else "#3760a0",
        )
    streams = sorted(
        {
            row[0]
            for row in c.execute(
                "select distinct streamId from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<?",
                (left, right),
            )
        }
    )
    for stream in streams:
        items = c.execute(
            "select start,end from CUPTI_ACTIVITY_KIND_KERNEL where streamId=? and end>? and start<? order by start",
            (stream, left, right),
        ).fetchall()
        axes[1].broken_barh(
            [((start - left) / 1e6, (stop - start) / 1e6) for start, stop in items],
            (streams.index(stream) - 0.35, 0.7),
            facecolors="#3760a0",
        )
    axes[0].set(
        yticks=[0, 1],
        yticklabels=["Background submit", "Long submit"],
        title="Actual Nsight host ranges and GPU kernels — heavy cold probe, not headline timing",
    )
    axes[1].set(
        yticks=list(range(len(streams))),
        yticklabels=[f"CUDA stream {s}" for s in streams],
        xlabel="Nsight time from selected window (ms)",
    )
    axes[1].set_xlim(0, (right - left) / 1e6)
    fig.savefig(ROOT / "figures/nsys_physical_timeline.png", dpi=160)
    plt.close(fig)
    print("NSYS", count, "kernels", len(markers), "matched quanta")


if __name__ == "__main__":
    main()
