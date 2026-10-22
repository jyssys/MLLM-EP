"""Relate old output divergence to recorded batch geometry, without a causal claim."""

import json
from .common import ROOT, OLD, write
from .history import table


def coda_sequence(data, rid):
    return [
        q
        for q in data["quantums"]
        if any(s["id"] == rid and s["phase"] == "coda" for s in q["spans"])
    ]


def prior_loops(data, rid, start, end):
    return [
        dict(
            depth=s["depth"],
            rows=sum(v["tokens"] for v in q["spans"]),
            requests=len(q["spans"]),
            ids=[v["id"] for v in q["spans"]],
        )
        for q in data["quantums"]
        if start <= q["host_begin"] <= end
        for s in q["spans"]
        if s["id"] == rid and s["phase"] == "recurrent"
    ]


def main():
    audit = json.loads((ROOT / "baseline_audit.json").read_text())
    result = []
    for item in audit["unresolved_batched_mismatches"]:
        p, n, t = int(item["P"]), int(item["N"]), int(item["trial"])
        folder = OLD / "raw" / item["run"]
        a = json.loads((folder / f"split_C1024_P{p}_N{n}_T{t}.json").read_text())
        b = json.loads((folder / f"grouped_C1024_P{p}_N{n}_T{t}.json").read_text())
        rid = item["id"]
        k = int(item["first_divergent_token"])
        ca, cb = coda_sequence(a, rid), coda_sequence(b, rid)
        ra = next(r for r in a["requests"] if r["id"] == rid)
        rb = next(r for r in b["requests"] if r["id"] == rid)
        if len(ca) != len(ra["output_ids"]) or len(cb) != len(rb["output_ids"]):
            result.append(dict(**item, status="CODA_ALIGNMENT_UNRESOLVED"))
            continue
        aa = prior_loops(a, rid, ca[max(0, k - 1)]["host_begin"], ca[k]["host_begin"])
        bb = prior_loops(b, rid, cb[max(0, k - 1)]["host_begin"], cb[k]["host_begin"])
        result.append(
            dict(
                **item,
                status="TRACE_ALIGNED_NOT_NUMERICAL_CAUSAL_PROOF",
                shared_prefix_exact=ra["output_ids"][:k] == rb["output_ids"][:k],
                split_coda_batch=len(ca[k]["spans"]),
                grouped_coda_batch=len(cb[k]["spans"]),
                split_recurrent_rows=json.dumps([v["rows"] for v in aa]),
                grouped_recurrent_rows=json.dumps([v["rows"] for v in bb]),
                same_recurrent_row_counts=[v["rows"] for v in aa]
                == [v["rows"] for v in bb],
                same_coda_ids=[s["id"] for s in ca[k]["spans"]]
                == [s["id"] for s in cb[k]["spans"]],
                affected_request="Background decode; not the newly admitted long prefill request",
                conclusion="Batch geometry can differ. BF16 numeric, graph, and lifecycle causes require controlled GPU probes.",
            )
        )
    table("historical_mismatch_batch_audit.csv", result)
    write(ROOT / "historical_mismatch_batch_audit.json", result)
    print(
        "MISMATCH_AUDIT",
        len(result),
        "cases",
        sum(not r.get("same_recurrent_row_counts", True) for r in result),
        "different row histories",
    )


if __name__ == "__main__":
    main()
