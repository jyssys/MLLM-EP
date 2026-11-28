"""Reduce measured whole-request routes with connector and correctness receipts."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def jsonl(path):
    if path.is_file():
        with path.open() as stream:
            yield from map(json.loads, stream)


def read(path):
    return json.loads(path.read_text())


def request_rows(path):
    fields = (
        "index", "request_id", "runtime_request_id", "status", "scheduled_arrival", "token_ids",
        "TTFT_ms", "TPOT_ms", "ITL_ms", "E2E_ms", "prompt_token_ids_SHA256",
    )
    for item in jsonl(path):
        compact = {name: item[name] for name in fields if name in item}
        if (tokens := item.get("prompt_token_ids")) is not None:
            payload = json.dumps(tokens, separators=(",", ":")).encode()
            compact["prompt_token_ids_SHA256"] = hashlib.sha256(payload).hexdigest()
        yield compact


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lo = int(index)
    return values[lo] + (index - lo) * (values[min(lo + 1, len(values) - 1)] - values[lo])


def counter_delta(before_after, role, name):
    def value(point):
        prefix = f"vllm:{name}{{"
        return sum(float(line.split()[-1]) for line in point["roles"].get(role, ())
                   if line.startswith(prefix))
    return value(before_after["after"]) - value(before_after["before"])


def observed(epochs):
    summary = []
    pairs = []
    groups = defaultdict(list)
    for epoch in epochs:
        if not (epoch/"configs"/"cohorts.jsonl").exists():
            continue
        proxy = {}
        for path in (epoch/"raw").glob("proxy_*.jsonl"):
            proxy.update((item["runtime_request_id"], item) for item in jsonl(path))
        for setup in jsonl(epoch/"configs"/"cohorts.jsonl"):
            if setup["observer"] != "clean":
                continue
            name = setup["tag"]
            metrics_path = epoch/"analysis"/f"{name}_metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = read(metrics_path)
            rows = list(request_rows(epoch/"raw"/f"{name}_requests.jsonl"))
            routes = [proxy[item["runtime_request_id"]] for item in rows if item["runtime_request_id"] in proxy]
            local = [event for event in routes if event.get("deflected") and event["status"] == "MEASURED_ONLINE"]
            assist = [event for event in routes if event.get("encoder_assisted") and event["status"] == "MEASURED_ONLINE"]
            complete = [item for item in rows if item.get("status") == "MEASURED_ONLINE"]
            state = read(epoch/"state"/f"{name}_complete.json")
            last = state["monotonic_start"] + state["duration_seconds"]
            ipc = defaultdict(int)
            for path in (epoch/"raw").glob("ec_cuda_ipc_*.jsonl"):
                role = path.name.removeprefix("ec_cuda_ipc_").split("_gpu", 1)[0]
                for it in jsonl(path):
                    if state["monotonic_start"] <= it["begin"] < last:
                        ipc[f"{role}_bytes"] += it["logical_bytes"]
                        ipc[f"{role}_images"] += it["images"]
                        ipc[f"{role}_inline"] += it["inline_images"]
            prom_path = epoch/"raw"/f"{name}_prometheus.json"
            kv = read(prom_path) if prom_path.exists() else None
            joint = metrics["categories"].get("JOINT_PASS", 0)
            absolute = sum(item.get("TPOT_ms", 0) <= 33 and max(item["ITL_ms"], default=0) <= 34
                           for item in complete)
            record = {
                "epoch":epoch.name, "tag":name, "route":setup["route"], "source":setup["source"],
                "count":setup["count"], "observed":metrics["completed_count"], "qps":setup["qps"],
                "output":setup["output"], "p_budget":setup["p_budget"], "d_budget":setup["d_budget"],
                "TTFT_p50_ms":metrics["TTFT_p50_ms"], "TTFT_p95_ms":metrics["TTFT_p95_ms"],
                "TTFT_p99_ms":metrics["TTFT_p99_ms"], "TPOT_p95_ms":metrics["TPOT_p95_ms"],
                "ITL_p99_ms":metrics["ITL_p99_ms"],
                "E2E_p95_ms":percentile([item["E2E_ms"] for item in complete], .95),
                "duration_s":metrics["duration_seconds"], "abs_decode_pass":absolute,
                "joint_pass":joint, "goodput_req_s":joint / metrics["duration_seconds"],
                "throughput_req_s":metrics["completed_requests_per_second"],
                "output_tok_s":metrics["raw_output_tokens_per_second"],
                "local_n":len(local), "assist_n":len(assist),
                "D_local_prompt_tokens":sum(event["input_tokens"] for event in local),
                "D_local_p50_D_submit_to_first_ms":median([
                    1000 * (event["D_first_generated_token"] - event["D_submit"]) for event in local]) if local else None,
                "NIXL_count":counter_delta(kv, "decode", "nixl_bytes_transferred_count") if kv else None,
                "NIXL_bytes":counter_delta(kv, "decode", "nixl_bytes_transferred_sum") if kv else None,
                **{f"EC_{role}_{key}":ipc[f"{role}_{key}"] for role in ("prefill","decode")
                   for key in ("bytes","images","inline")},
            }
            summary.append(record)
            key = (setup["manifest_sha256"], setup["count"], setup["qps"], setup["output"],
                   setup.get("arrival", "poisson"))
            groups[key].append((record, rows, routes))
    for group in groups.values():
        for a_no, (a, a_rows, _) in enumerate(group):
            for b, b_rows, b_routes in group[a_no+1:]:
                aa = {row["index"]:row for row in a_rows}
                bb = {row["index"]:row for row in b_rows}
                ab = sorted(aa.keys() & bb.keys())
                aligned = [i for i in ab if aa[i]["request_id"] == bb[i]["request_id"]]
                matched = [i for i in aligned if aa[i]["status"] == bb[i]["status"] == "MEASURED_ONLINE"]
                if len(aligned) != a["count"]:
                    continue
                local_ids = {event["runtime_request_id"] for event in b_routes if event.get("deflected")}
                local = [i for i in matched if bb[i]["runtime_request_id"] in local_ids]
                pairs.append({
                    "a":a["epoch"]+"/"+a["tag"], "b":b["epoch"]+"/"+b["tag"],
                    "aligned":len(aligned), "completed_pairs":len(matched), "local_paired":len(local),
                    "token_exact":sum(aa[i]["token_ids"] == bb[i]["token_ids"] for i in matched),
                    "local_exact":sum(aa[i]["token_ids"] == bb[i]["token_ids"] for i in local),
                    "prompt_exact":sum(aa[i].get("prompt_token_ids_SHA256")
                                       == bb[i].get("prompt_token_ids_SHA256")
                                       and aa[i].get("prompt_token_ids_SHA256") is not None for i in matched),
                    "same_scheduled_offsets":all(abs((aa[i]["scheduled_arrival"]-aa[0]["scheduled_arrival"])
                                                     -(bb[i]["scheduled_arrival"]-bb[0]["scheduled_arrival"]))
                                                 < .00001 for i in matched) if 0 in aa and 0 in bb else False,
                })
    return summary, pairs


def figures(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapping = {
        ("deflection-inline","inline_native_n24"):"Native EPD",
        ("deflection-inline","inline_ondemand_n24"):"Matched route",
        ("deflection-inline","inline_naive_n24"):"Naive E→D",
        ("deflection-inline","inline_cost_n24"):"Cost E→D",
        ("dp3","colocated_long_n24"):"DP3 · B8192",
        ("dp3-retry","colocated_long_n24"):"DP3 · B8192",
        ("dp3-b2048","colocated_long_n24"):"DP3 · B2048",
    }
    selected = [(mapping[key], row) for row in rows
                if (key := (row["epoch"],row["tag"])) in mapping and row["observed"]]
    if len(selected) >= 2:
        metric_panels = (("TTFT_p95_ms","TTFT p95 (s)",.001),
                         ("TPOT_p95_ms","TPOT p95 (ms)",1),
                         ("ITL_p99_ms","ITL p99 (ms)",1),
                         ("goodput_req_s","Joint SLO goodput (req/s)",1))
        fig, axs = plt.subplots(2, 2, figsize=(10,6), layout="constrained")
        for ax, (key,label,scale) in zip(axs.flat, metric_panels, strict=True):
            ax.bar([n for n, _ in selected], [r[key]*scale for _,r in selected], color="#557599")
            ax.set(ylabel=label)
            ax.tick_params(axis="x", labelrotation=22)
        fig.suptitle("Natural very long · QPS 1.465 · Fixed16 · paired N24")
        fig.savefig(out/"whole_request_deflection.png", dpi=180)
        plt.close(fig)

    selected = [(row["tag"].removeprefix("mixed_").removesuffix("_n24"), row) for row in rows
                if row["epoch"] == "deflection-inline" and row["tag"].startswith("mixed_")
                and row["observed"]]
    if len(selected) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(8,3.6), layout="constrained")
        labels = [label for label, _ in selected]
        ax[0].bar(labels, [row["local_n"] for _,row in selected], color="#087e8b")
        ax[0].set(ylabel="D-local completed requests")
        ax[1].bar(labels, [row["TTFT_p95_ms"]/1000 for _,row in selected], color="#de6b35")
        ax[1].set(ylabel="TTFT p95 (s)")
        fig.suptitle("Controlled 16K · pairing uses the same visual-ratio mixture")
        fig.savefig(out/"prefill_route_modality_ablation.png", dpi=180)
        plt.close(fig)

    gate_receipt = out.parent/"decision_gates.json"
    if gate_receipt.is_file():
        plot_gates(read(gate_receipt), out, plt)


def plot_gates(gates, out, plt):
    from textwrap import fill

    fig, ax = plt.subplots(figsize=(11, 5.6), layout="constrained")
    ax.set_axis_off()
    content = [[row["gate"], row["status"], fill(row["measurement"], 63)] for row in gates]
    table = ax.table(cellText=content, colLabels=["Gate", "Decision", "Measurement"],
                     colWidths=[.24, .20, .56], cellLoc="left", colLoc="left", bbox=(0, .01, 1, .90))
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#234560")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f2f6f8" if row % 2 == 0 else "#ffffff")
        cell.set_edgecolor("#dce5ea")
    ax.text(0, .98, "EPD operating-region and deflection decisions", va="top", weight="bold", fontsize=15)
    fig.savefig(out/"candidate_motivation_verdict_matrix.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", action="append", type=Path, required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result, pairs = observed(args.epoch)
    if not result:
        raise ValueError("No completed online case receipts")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/"whole_request_routes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    (args.out/"whole_request_paired_tokens.json").write_text(json.dumps(pairs, indent=2) + "\n")
    out = args.out/"figures"
    out.mkdir(exist_ok=True)
    figures(result, out)


if __name__ == "__main__":
    main()
