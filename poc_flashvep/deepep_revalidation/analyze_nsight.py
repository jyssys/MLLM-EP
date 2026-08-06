#!/usr/bin/env python3
"""Extract compact actual-overlap evidence from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


PAIR_QUERY = """
WITH ranges AS (
  SELECT (globalTid & 0xFFFFFFFFFF000000) AS gp, start, end,
         row_number() OVER (
           PARTITION BY (globalTid & 0xFFFFFFFFFF000000) ORDER BY start DESC
         ) AS rn
  FROM NVTX_EVENTS
  WHERE text LIKE 'FLASHVEP%' AND end IS NOT NULL
),
window AS (
  SELECT gp, min(start) AS start, max(end) AS end
  FROM ranges WHERE rn <= 18 GROUP BY gp
),
kernels AS (
  SELECT k.*, s.value AS name
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON s.id = k.demangledName
  JOIN window w ON w.gp = k.globalPid
  WHERE k.start >= w.start AND k.end <= w.end
),
comm AS (
  SELECT *, CASE WHEN name LIKE '%::dispatch<%' THEN 'dispatch' ELSE 'combine' END AS stage
  FROM kernels
  WHERE name LIKE '%deep_ep::intranode::dispatch<%'
     OR name LIKE '%deep_ep::intranode::combine<%'
),
expert AS (
  SELECT * FROM kernels WHERE name = 'fused_moe_kernel'
)
SELECT p.pid, c.stage, c.streamId AS comm_stream, e.streamId AS expert_stream,
       c.start AS comm_start, c.end AS comm_end,
       e.start AS expert_start, e.end AS expert_end,
       min(c.end, e.end) - max(c.start, e.start) AS overlap_ns
FROM comm c
JOIN expert e ON c.globalPid = e.globalPid
             AND c.start < e.end AND e.start < c.end
JOIN PROCESSES p ON p.globalPid = c.globalPid
ORDER BY p.pid, c.stage, c.start
"""


SLOWDOWN_QUERY = """
WITH nvtx AS (
  SELECT *, (globalTid & 0xFFFFFFFFFF000000) AS gp,
         row_number() OVER (
           PARTITION BY (globalTid & 0xFFFFFFFFFF000000) ORDER BY start
         ) AS rn
  FROM NVTX_EVENTS
  WHERE text LIKE 'FLASHVEP%' AND end IS NOT NULL
),
tagged AS (
  SELECT *,
         CASE WHEN rn BETWEEN 19 AND 36 THEN 'serial'
              WHEN rn BETWEEN 43 AND 60 THEN 'overlap' END AS variant,
         CASE WHEN text LIKE '%_D_%' THEN 'D'
              WHEN text LIKE '%_E_%' THEN 'E' ELSE 'C' END AS stage
  FROM nvtx
),
attributed AS (
  SELECT t.gp, t.variant, t.stage, k.start, k.end
  FROM tagged t
  JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
    ON (r.globalTid & 0xFFFFFFFFFF000000) = t.gp
   AND r.start BETWEEN t.start AND t.end
  JOIN CUPTI_ACTIVITY_KIND_KERNEL k
    ON k.globalPid = t.gp AND k.correlationId = r.correlationId
  WHERE t.variant IS NOT NULL
),
sums AS (
  SELECT gp, variant, stage, sum(end - start) AS ns
  FROM attributed GROUP BY gp, variant, stage
),
pivot AS (
  SELECT gp, stage,
         max(CASE WHEN variant = 'serial' THEN ns END) AS serial_ns,
         max(CASE WHEN variant = 'overlap' THEN ns END) AS overlap_ns
  FROM sums GROUP BY gp, stage
)
SELECT p.pid, stage, serial_ns, overlap_ns,
       1.0 * overlap_ns / serial_ns AS slowdown
FROM pivot JOIN PROCESSES p ON p.globalPid = pivot.gp
WHERE serial_ns IS NOT NULL AND overlap_ns IS NOT NULL
ORDER BY p.pid, stage
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    connection = sqlite3.connect(args.sqlite)
    connection.row_factory = sqlite3.Row
    pairs = [dict(row) for row in connection.execute(PAIR_QUERY)]
    slowdowns = [dict(row) for row in connection.execute(SLOWDOWN_QUERY)]
    connection.close()

    workers: dict[int, dict[str, Any]] = {}
    for row in pairs:
        worker = workers.setdefault(
            int(row["pid"]),
            {
                "pid": int(row["pid"]),
                "dispatch_expert_pairs": 0,
                "combine_expert_pairs": 0,
                "overlap_ns": 0,
                "max_overlap_ns": 0,
                "comm_streams": set(),
                "expert_streams": set(),
            },
        )
        worker[f"{row['stage']}_expert_pairs"] += 1
        worker["overlap_ns"] += int(row["overlap_ns"])
        worker["max_overlap_ns"] = max(
            worker["max_overlap_ns"], int(row["overlap_ns"])
        )
        worker["comm_streams"].add(int(row["comm_stream"]))
        worker["expert_streams"].add(int(row["expert_stream"]))

    compact_workers = []
    for pid, worker in sorted(workers.items()):
        compact_workers.append(
            {
                "pid": pid,
                "dispatch_expert_pairs": worker["dispatch_expert_pairs"],
                "combine_expert_pairs": worker["combine_expert_pairs"],
                "total_overlap_ms": worker["overlap_ns"] / 1e6,
                "max_single_overlap_us": worker["max_overlap_ns"] / 1e3,
                "comm_streams": sorted(worker["comm_streams"]),
                "expert_streams": sorted(worker["expert_streams"]),
                "streams_are_distinct": worker["comm_streams"].isdisjoint(
                    worker["expert_streams"]
                ),
            }
        )

    slowdown_rows = [
        {
            "pid": int(row["pid"]),
            "stage": row["stage"],
            "serial_kernel_sum_ms": int(row["serial_ns"]) / 1e6,
            "overlap_kernel_sum_ms": int(row["overlap_ns"]) / 1e6,
            "slowdown": float(row["slowdown"]),
        }
        for row in slowdowns
    ]
    actual = len(compact_workers) == 4 and all(
        worker["dispatch_expert_pairs"] > 0
        and worker["combine_expert_pairs"] > 0
        and worker["streams_are_distinct"]
        for worker in compact_workers
    )
    summary = {
        "status": "PASS" if actual else "FAIL",
        "actual_kernel_overlap": actual,
        "configuration": {
            "batch_equivalent": 128,
            "communication_sms": 8,
            "microbatches": 2,
            "warmups": 1,
            "iterations": 3,
        },
        "kernel_classes": {
            "communication": [
                "deep_ep::intranode::dispatch",
                "deep_ep::intranode::combine",
            ],
            "expert": ["fused_moe_kernel"],
        },
        "workers": compact_workers,
        "nvtx_attributed_kernel_sum_slowdown": slowdown_rows,
        "max_nsight_D_kernel_sum_slowdown": max(
            row["slowdown"] for row in slowdown_rows if row["stage"] == "D"
        ),
        "max_nsight_E_kernel_sum_slowdown": max(
            row["slowdown"] for row in slowdown_rows if row["stage"] == "E"
        ),
        "C_kernel_sum_slowdown": None,
        "C_kernel_sum_note": "DeepEP combine launches were asynchronous to the host NVTX range, so C slowdown uses corrected CUDA-event stage timing in operator_matrix.json.",
        "command": "BATCHES=128 SMS_VALUES=8 WARMUPS=1 ITERATIONS=3 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --trace-fork-before-exec=true ... run_operator_replay.sh",
        "large_trace_committed": False,
        "interpretation_limit": "Concurrent kernel intervals prove residency overlap; HBM/L2 contention is not directly established by this trace.",
    }
    (output_dir / "nsight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "nsight_overlap_pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    lines = [
        "Nsight Systems actual overlap: PASS" if actual else "Nsight Systems actual overlap: FAIL",
        "Configuration: B_eq=128, communication SMs=8, K=2",
    ]
    for worker in compact_workers:
        lines.append(
            "PID {pid}: D||E pairs={dispatch_expert_pairs}, C||E pairs={combine_expert_pairs}, "
            "total intersection={total_overlap_ms:.6f} ms, max intersection={max_single_overlap_us:.3f} us, "
            "comm streams={comm_streams}, expert streams={expert_streams}".format(**worker)
        )
    lines.extend(
        [
            f"Max Nsight-attributed D kernel-sum slowdown: {summary['max_nsight_D_kernel_sum_slowdown']:.6f}x",
            f"Max Nsight-attributed E kernel-sum slowdown: {summary['max_nsight_E_kernel_sum_slowdown']:.6f}x",
            "C kernel-sum slowdown: unavailable from host NVTX attribution; see corrected CUDA-event result.",
            summary["interpretation_limit"],
        ]
    )
    (output_dir / "nsight_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"actual_kernel_overlap": actual, "workers": compact_workers}, indent=2))


if __name__ == "__main__":
    main()
