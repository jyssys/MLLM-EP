"""Reduce independent stage calibration receipts and plot their measured scope."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def read(file):
    return json.loads(file.read_text())


def summarize(epochs):
    result = []
    observations = []
    for epoch in epochs:
        for path in sorted((epoch / "configs").glob("*_workload.json")):
            tag = path.name.removesuffix("_workload.json")
            if not (epoch / "analysis" / f"{tag}.json").exists():
                continue
            setup = read(path)
            value = read(epoch / "analysis" / f"{tag}.json")
            stage = setup["stage"]
            row = {
                "epoch": epoch.name, "tag": tag, "cohort": setup["cohort_id"], "stage": stage,
                "batch_token_budget_p": setup["p_budget"], "requests": value["requests"],
                "input_median": median(setup["input_tokens"]),
                "requests_per_second": value["requests_per_second"],
                "tokens_per_second": value["tokens_per_second"],
                "token_unit": {"encoder": "visual", "prefill": "LM input", "decode": "output"}[stage],
                "scope": value["scope"],
            }
            result.append(row)
            raw = epoch / "raw" / f"{tag}_rows.json"
            inputs = [raw] if raw.exists() else sorted((epoch / "raw" / tag).glob("*_rows.json"))
            for file in inputs:
                for item in read(file):
                    if "submit" not in item or "finish" not in item:
                        continue
                    observations.append({
                        "epoch": epoch.name, "cohort": setup["cohort_id"], "stage": stage,
                        "request_id": item["request_id"],
                        "images": item.get("image_count"), "input": item.get("input_tokens"),
                        "output": item.get("output_tokens"),
                        "stage_http_ms": 1000 * (item["finish"] - item["submit"]),
                    })
    return result, observations


def observe_d_service(epochs):
    result = []
    for epoch in epochs:
        timing = {}
        for path in (epoch / "raw").glob("proxy_*.jsonl"):
            with path.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if (event["status"] == "MEASURED_ONLINE" and "D_submit" in event
                            and "D_http_complete" in event and not event.get("deflected")):
                        timing[event["runtime_request_id"]] = event["D_http_complete"] - event["D_submit"]
        for path in (epoch / "raw").glob("*_requests.jsonl"):
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["runtime_request_id"] in timing and row["status"] == "MEASURED_ONLINE":
                        result.append({
                            "epoch":epoch.name, "run":path.stem.removesuffix("_requests"),
                            "request_id":row["request_id"], "output_tokens":row["output_tokens"],
                            "D_submit_to_completion_ms":1000*timing[row["runtime_request_id"]],
                        })
    return result


def plot_rates(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    selected = [(cohort, [row for row in rows if row["epoch"] == "calibration-sweep"
                          and row["cohort"] == cohort]) for cohort in ("mix", "upper", "long")]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    colors = {"encoder":"#087e8b", "prefill":"#de6b35", "decode":"#334f80"}
    stages = list(colors)
    for j, (cohort, values) in enumerate(selected):
        for cell in values:
            stage = cell["stage"]
            n = cell["requests_per_second"]
            pos = j + .25 * (stages.index(stage) - 1)
            ax.bar(pos, n, .23, color=colors[stage], label=stage.title() if j == 0 else None)
            ax.text(pos, n * 1.1, f"{n:.2f}", va="bottom", ha="center", fontsize=9)
    ax.set(yscale="log", ylim=(.8, 48), ylabel="Finite reference (requests/s)",
           xticks=range(len(selected)), xticklabels=["Natural mixed", "Natural upper", "Natural very long"],
           title="Independent E/P/D service references · H200 × 3")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.legend(title="Stage", frameon=False, loc="upper left")
    ax.text(0, -.22, "Decode uses pre-staged native KV at concurrency 12. It can exceed the Decode tail SLO.",
            transform=ax.transAxes, fontsize=9)
    fig.savefig(output / "independent_stage_capacity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(observations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specifications = (
        ("encoder", "images", "Image count", "Encoder", "encoder_image_count_vs_service"),
        ("prefill", "input", "LM input tokens", "Prefill", "input_length_vs_prefill_service"),
    )
    for stage, x, x_label, title, filename in specifications:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for cohort in sorted({row["cohort"] for row in observations if row["epoch"] == "calibration-sweep"}):
            pts = [(row[x], row["stage_http_ms"] / 1000) for row in observations
                   if row["epoch"] == "calibration-sweep" and row["stage"] == stage
                   and row["cohort"] == cohort and row[x] is not None]
            if pts:
                ax.scatter(*zip(*pts), s=20, alpha=.72, label=cohort)
        ax.set(xlabel=x_label, ylabel="Stage HTTP service (s)", title=f"Independent {title} burst reference")
        ax.legend(title="Natural cohort", frameon=False)
        ax.text(0, -.18, "Each point includes queueing at that stage under concurrent dispatch.",
                transform=ax.transAxes, fontsize=9)
        fig.savefig(output / f"{filename}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_d_geometry(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [item for item in rows if item["epoch"] == "calibration-sweep"
                and item["run"] in ("mix_q096", "mix_o64", "mix_o256")]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(6.8,4.0), layout="constrained")
    columns = sorted({item["output_tokens"] for item in selected})
    groups = [[item["D_submit_to_completion_ms"] for item in selected if item["output_tokens"] == count]
              for count in columns]
    ax.boxplot(groups, tick_labels=[str(column) for column in columns], showmeans=True)
    ax.set(xlabel="Actual emitted output tokens", ylabel="D submit → completion (ms)",
           title="Decode stage residence by output length")
    ax.text(0, -.18, "Native mixed Qwen2.5-VL inputs. Includes KV import and request queueing.",
            transform=ax.transAxes, fontsize=9)
    fig.savefig(output/"output_length_vs_decode_service.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result, observations = summarize(args.epoch)
    if not result:
        raise ValueError("No completed calibration receipts")
    with (args.out / "independent_stage_capacity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    (args.out / "stage_request_observations.json").write_text(
        json.dumps(observations, sort_keys=True, separators=(",", ":")) + "\n")
    decoded = observe_d_service(args.epoch)
    (args.out / "decode_request_observations.json").write_text(
        json.dumps(decoded, sort_keys=True, separators=(",", ":")) + "\n")
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    plot_rates(result, figures)
    plot_geometry(observations, figures)
    plot_d_geometry(decoded, figures)


if __name__ == "__main__":
    main()
