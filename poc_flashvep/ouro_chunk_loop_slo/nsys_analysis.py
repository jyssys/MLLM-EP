"""Heavy profiler timeline coverage. Never convert service sums to E2E shares."""

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import ROOT, write
from .history import table


def read(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(row, choices):
    for key in choices:
        if key in row and row[key] not in ("", None, "N/A"):
            return float(row[key].replace(",", ""))
    raise ValueError("Missing numeric profiler field: " + repr(list(row)))


def bucket(name):
    name = name.lower()
    if any(v in name for v in ("memcpy", "memset", "copy_", "scatter", "gather")):
        return "MEMORY_OR_LAYOUT_NAME"
    if any(v in name for v in ("flash", "attention", "fmha")):
        return "ATTENTION_NAME"
    if any(v in name for v in ("gemm", "matmul", "cublas", "mma")):
        return "GEMM_NAME"
    if any(v in name for v in ("norm", "rope", "rotary")):
        return "NORM_OR_ROPE_NAME"
    return "OTHER_NAME"


def main():
    summary = []
    names = []
    outputs = ROOT / "figures"
    outputs.mkdir(exist_ok=True)
    for folder in sorted((ROOT / "raw").glob("nsys_*")):
        status = (
            json.loads((folder / "nsys_export_status.json").read_text())
            if (folder / "nsys_export_status.json").exists()
            else {}
        )
        traces = read(folder / "nsys_cuda_gpu_trace.csv")
        if not traces:
            summary.append(
                dict(
                    job=folder.name,
                    status="NO_PORTABLE_GPU_TRACE",
                    export_status=status,
                    interpretation="NOT_AVAILABLE. Read preserved profiler logs. No inferred kernel timing.",
                )
            )
            continue
        identity = json.loads((folder / "identity.json").read_text())
        events = []
        service = defaultdict(float)
        count = defaultdict(int)
        for row in traces:
            start = number(row, ("Start (ns)", "Start")) / 1e6
            duration = number(row, ("Duration (ns)", "Duration")) / 1e6
            name = row.get("Name", row.get("Kernel Name", "UNKNOWN"))
            stream = row.get("Strm", row.get("Stream", "UNKNOWN"))
            events.append((start, start + duration, str(stream), name))
            service[bucket(name)] += duration
            count[bucket(name)] += 1
        events.sort()
        merged = []
        for begin, end, *_ in events:
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([begin, end])
        union = sum(end - begin for begin, end in merged)
        span = events[-1][1] - events[0][0]
        span = max(end for _, end, *_ in events) - events[0][0]
        summary.append(
            dict(
                job=folder.name,
                GPU=identity["physical_GPU"],
                status="HEAVY_TRACE_AVAILABLE",
                events=len(events),
                sum_kernel_and_memory_service_ms=sum(
                    end - begin for begin, end, *_ in events
                ),
                all_stream_union_busy_ms=union,
                first_to_last_GPU_activity_ms=span,
                holes_inside_GPU_activity_span_ms=span - union,
                coverage_fraction=union / span,
                interpretation="Heavy GPU activity-window coverage, not clean E2E critical-path or removable-time percentage.",
            )
        )
        for key, value in service.items():
            names.append(
                dict(
                    job=folder.name,
                    GPU=identity["physical_GPU"],
                    heuristic_name_bucket=key,
                    events=count[key],
                    sum_service_ms=value,
                    scope="Name heuristic only. Overlapping service times cannot be added into E2E.",
                )
            )
        streams = sorted({event[2] for event in events})
        fig, ax = plt.subplots(figsize=(13, 4))
        # Display a real one-second window after the first activity, not a fabricated schedule.
        origin = events[0][0]
        for stream in streams:
            intervals = [
                (begin - origin, min(end - origin, 1000) - (begin - origin))
                for begin, end, s, _ in events
                if s == stream and begin - origin < 1000
            ]
            ax.broken_barh(intervals, (streams.index(stream) - 0.3, 0.6))
        ax.set(
            yticks=range(len(streams)),
            yticklabels=streams,
            xlim=(0, 1000),
            xlabel="Profiler GPU time from first activity (ms)",
            ylabel="CUDA stream ID",
            title=f"HEAVY measured stream timeline: {folder.name}, physical GPU{identity['physical_GPU']}\nFirst 1000ms of captured activity. Not clean E2E or a model-parallel schedule.",
        )
        fig.tight_layout()
        fig.savefig(outputs / (folder.name + "_timeline.png"), dpi=160)
        fig.savefig(outputs / (folder.name + "_timeline.svg"))
        plt.close(fig)
    table("nsys_timeline_coverage.csv", summary)
    table("nsys_kernel_name_buckets.csv", names)
    write(
        ROOT / "nsys_summary.json",
        dict(
            cases=len(summary),
            with_portable_trace=sum(
                r["status"] == "HEAVY_TRACE_AVAILABLE" for r in summary
            ),
            source="Nsight CSV exports. Full local captures are separate.",
            limitations="No service-sum/E2E equivalence. No fabricated critical-path decomposition or bandwidth.",
        ),
    )
    print("NSYS_ANALYSIS", len(summary))


if __name__ == "__main__":
    main()
