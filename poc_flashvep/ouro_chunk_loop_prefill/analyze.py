"""Request/cohort-level analysis. Never bootstrap tokens as independent requests."""

import csv
import json
import numpy as np
from .common import ROOT, write


def percent(values, q):
    return float(np.percentile(values, q)) if values else None


def scan():
    for file in sorted((ROOT / "raw").glob("*/*.json")):
        data = json.loads(file.read_text())
        if "requests" in data and "mode" in data:
            yield file, data


def metrics(file, d):
    rows = d["requests"]
    background = [r for r in rows if r["background"]]
    long = next(r for r in rows if not r["background"])
    itls = [v for r in background for v in r["measured_ITLs_s"]]
    span = d["end"] - min(r["arrival"] for r in rows)
    slo = [
        r["TTFT_s"] <= 5
        and (not r["decode_ITLs_s"] or max(r["decode_ITLs_s"]) <= 0.100)
        and r["finish_reason"] != "abort"
        for r in rows
    ]
    return dict(
        run=file.parent.name,
        file=file.name,
        mode=d["mode"],
        C=d["chunk_tokens"],
        G=d["loop_group"],
        P=d["prompt_bin"],
        N=d["concurrency"],
        trial=d["trial"],
        trace=d["heavy"],
        warmup=d.get("warmup", False),
        graphs=d["graphs"],
        evidence="LIVE_MEASURED",
        requests=len(rows),
        tokens=sum(len(r["output_ids"]) for r in rows),
        interference_span_s=d["measured_cohort_s"],
        client_cohort_makespan_s=span,
        long_TTFT_s=long["TTFT_s"],
        ITL_p95_s=percent(itls, 95),
        ITL_p99_s=percent(itls, 99),
        ITL_max_s=max(itls) if itls else None,
        E2E_p50_s=percent([r["E2E_s"] for r in rows], 50),
        E2E_p95_s=percent([r["E2E_s"] for r in rows], 95),
        TTFT_p50_s=percent([r["TTFT_s"] for r in rows], 50),
        TTFT_p95_s=percent([r["TTFT_s"] for r in rows], 95),
        TPOT_p50_s=percent([r["TPOT_s"] for r in rows if r["TPOT_s"] is not None], 50),
        completed_requests_s=len(rows) / span,
        output_tokens_s=sum(len(r["output_ids"]) for r in rows) / span,
        cohort_SLO_qualified_requests_s=sum(slo) / span,
        SLO_qualified=sum(slo),
        goodput_note="Drained-cohort SLO-qualified completion rate, not steady-state open-loop goodput.",
        truncated=sum(r["finish_reason"] == "length" for r in rows),
        EOS=sum(r["finish_reason"] == "stop" for r in rows),
        live_background=d["live_decode_at_long_admission"],
        peak_HBM_bytes=d["memory"]["peak"],
        activation_bytes=d["memory"]["activation_bytes"],
        KV_pool_bytes=d["memory"]["KV_bytes"],
        quantum_count=len(d["quantums"]),
        graph_entries=d["graph_entries"],
        graph_captures=d.get("graph_stats", {}).get("captures"),
        graph_replays=d.get("graph_stats", {}).get("replays"),
    )


def csvfile(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT / name).open("w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def compare_outputs(records):
    groups = {}
    for file, d in records:
        if d["heavy"]:
            continue
        key = (
            file.parent.name,
            d["prompt_bin"],
            d["concurrency"],
            d["trial"],
            d["chunk_tokens"],
        )
        groups.setdefault(key, []).append((file, d))
    out = []
    for key, values in groups.items():
        reference = next((d for _, d in values if d["mode"] == "split"), None)
        if reference is None:
            continue
        ref = {r["id"]: r for r in reference["requests"]}
        for file, d in values:
            for r in d["requests"]:
                a, b = ref[r["id"]]["output_ids"], r["output_ids"]
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                out.append(
                    dict(
                        run=key[0],
                        P=key[1],
                        N=key[2],
                        trial=key[3],
                        C=key[4],
                        mode=d["mode"],
                        id=r["id"],
                        tokens_exact=a == b,
                        length_equal=len(a) == len(b),
                        reference_tokens=len(a),
                        candidate_tokens=len(b),
                        depths_exact=ref[r["id"]]["depths"] == r["depths"],
                        first_divergent_token=first,
                        same_index_token_agreement=sum(x == y for x, y in zip(a, b))
                        / max(len(a), len(b)),
                        reference_EOS=ref[r["id"]]["finish_reason"],
                        candidate_EOS=r["finish_reason"],
                    )
                )
    return out


def main():
    records = list(scan())
    rows = [metrics(f, d) for f, d in records]
    csvfile("results_raw.csv", rows)
    csvfile("natural_output_parity.csv", compare_outputs(records))
    request_rows = []
    with (
        (ROOT / "request_events.jsonl").open("w") as rf,
        (ROOT / "scheduler_quantum_trace.jsonl").open("w") as qf,
    ):
        for file, d in records:
            common = dict(
                run=file.parent.name,
                file=file.name,
                mode=d["mode"],
                C=d["chunk_tokens"],
                G=d["loop_group"],
                P=d["prompt_bin"],
                concurrency=d["concurrency"],
                trial=d["trial"],
                traced=d["heavy"],
                evidence="LIVE_MEASURED",
            )
            for r in d["requests"]:
                event = dict(common, **r)
                rf.write(json.dumps(event) + "\n")
                request_rows.append(
                    dict(
                        common,
                        id=r["id"],
                        background=r["background"],
                        output_tokens=len(r["output_ids"]),
                        E2E_s=r["E2E_s"],
                        TTFT_s=r["TTFT_s"],
                        TPOT_s=r["TPOT_s"],
                        finish_reason=r["finish_reason"],
                    )
                )
            gpu = {g["quantum"]: g for g in d["gpu_quantums"] if "quantum" in g}
            for i, q in enumerate(d["quantums"]):
                qf.write(
                    json.dumps(
                        dict(
                            common,
                            quantum=i,
                            **q,
                            GPU=gpu.get(i),
                            GPU_status="LIVE_MEASURED"
                            if i in gpu
                            else "NOT_AVAILABLE_CLEAN_RUN",
                        )
                    )
                    + "\n"
                )
    csvfile("request_metrics.csv", request_rows)
    frontier = []
    for family in ("static_grid_v1", "fa4_grid_v1"):
        for axis in ("ITL_p95_s", "ITL_p99_s"):
            for p, n in ((2048, 4), (2048, 8), (8192, 4), (8192, 8)):
                points = [
                    r
                    for r in rows
                    if r["run"] == family and r["P"] == p and r["N"] == n
                ]
                for r in points:
                    dominates = [
                        q
                        for q in points
                        if q is not r
                        and q["long_TTFT_s"] <= r["long_TTFT_s"]
                        and q[axis] <= r[axis]
                        and q["cohort_SLO_qualified_requests_s"]
                        >= r["cohort_SLO_qualified_requests_s"]
                        and (q["long_TTFT_s"] < r["long_TTFT_s"] or q[axis] < r[axis])
                    ]
                    frontier.append(
                        dict(
                            run=family,
                            ITL_axis=axis,
                            P=p,
                            N=n,
                            C=r["C"],
                            G=r["G"],
                            mode=r["mode"],
                            nondominated=not dominates,
                            dominated_by="|".join(
                                q["mode"] + ":" + str(q["C"]) for q in dominates
                            ),
                        )
                    )
    csvfile("static_frontier.csv", frontier)
    write(
        ROOT / "analysis_progress.json",
        dict(
            completed_cells=len(rows),
            frontier=frontier,
            limits="Pilot frontier. No token-level independence or steady-state goodput claim.",
        ),
    )
    print("ANALYZED", len(rows), "cells", len(request_rows), "requests", flush=True)


if __name__ == "__main__":
    main()
