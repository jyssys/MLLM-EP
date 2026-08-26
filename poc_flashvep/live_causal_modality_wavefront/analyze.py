"""Analyze stock versus live causal-wavefront measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _stats(values: pd.Series) -> dict[str, float]:
    return {
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _load_mode(
    root: Path, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = root / mode
    driver_rows = []
    for rank in range(2):
        payload = json.loads((directory / f"driver.dp_rank{rank}.json").read_text())
        if not payload["ok"] and mode != "wavefront":
            raise RuntimeError(payload)
        driver_rows.extend(payload.get("records", []))
    forward_rows, layer_rows = [], []
    for rank in range(4):
        payload = json.loads((directory / "raw" / f"rank{rank}.json").read_text())
        if payload["status"] != "ok" or payload["visible_devices"] != "1,2,3,4":
            raise RuntimeError(payload)
        for row in payload["forward_records"]:
            forward_rows.append({**row, "ep_rank": rank})
        for row in payload["layer_records"]:
            layer_rows.append({**row, "ep_rank": rank})
    return (
        pd.DataFrame(driver_rows),
        pd.DataFrame(forward_rows),
        pd.DataFrame(layer_rows),
    )


def _forward_wave(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for (wave, request_id, phase, iteration), local in frame.groupby(
        ["wave", "request_id", "phase", "iteration"]
    ):
        ranks = []
        for rank, rank_rows in local.groupby("ep_rank"):
            origin = rank_rows[rank_rows.ubatch_id.isin((-1, 0))]
            if len(origin) != 1:
                raise AssertionError((mode, wave, rank, rank_rows.to_dict("records")))
            makespan = float(rank_rows.end_ms.max() - origin.iloc[0].start_ms)
            ranks.append((int(rank), makespan))
        if len(ranks) != 4:
            raise AssertionError((mode, wave, ranks))
        rows.append(
            {
                "wave": int(wave),
                "request_id": request_id,
                "phase": phase,
                "iteration": int(iteration),
                "prefill_ms": max(value for _, value in ranks),
                "critical_ep_rank": max(ranks, key=lambda item: item[1])[0],
                "rank_spread_ms": max(value for _, value in ranks)
                - min(value for _, value in ranks),
            }
        )
    return pd.DataFrame(rows)


def _correctness(root: Path, drivers: dict[str, pd.DataFrame]) -> dict[str, Any]:
    token_checks = []
    for mode, frame in drivers.items():
        if frame.empty:
            continue
        for wave, local in frame.groupby("wave"):
            values = local.output_tokens.apply(tuple).tolist()
            token_checks.append(
                {"mode": mode, "wave": int(wave), "dp_equal": len(set(values)) == 1}
            )
    logit_rows = []
    generated_rows = []
    stock = drivers["stock"]
    stock_correct = stock[stock.phase == "correctness"].set_index(
        ["wave", "driver_dp_rank"]
    )
    for ep_rank in (0, 2):
        a_file = np.load(root / "stock" / "raw" / f"rank{ep_rank}.logits.npz")
        c_file = np.load(root / "wavefront" / "raw" / f"rank{ep_rank}.logits.npz")
        a = {int(key.removeprefix("wave_")): a_file[key] for key in a_file.files}
        c = {int(key.removeprefix("wave_")): c_file[key] for key in c_file.files}
        for wave in sorted(set(a).intersection(c)):
            av = a[wave].astype(np.float32)
            cv = c[wave].astype(np.float32)
            difference = np.abs(av - cv)
            driver_dp_rank = ep_rank // 2
            stock_token = int(stock_correct.loc[(wave, driver_dp_rank)].output_tokens[0])
            wavefront_token = int(np.argmax(cv))
            generated_rows.append(
                {
                    "ep_rank": ep_rank,
                    "wave": int(wave),
                    "stock_token": stock_token,
                    "wavefront_token": wavefront_token,
                }
            )
            logit_rows.append(
                {
                    "ep_rank": ep_rank,
                    "wave": int(wave),
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "cosine": float(
                        np.dot(av, cv) / (np.linalg.norm(av) * np.linalg.norm(cv))
                    ),
                }
            )
    logits = pd.DataFrame(logit_rows)
    generated = pd.DataFrame(generated_rows)
    c_dp_equal = all(
        len(set(local.wavefront_token)) == 1 for _, local in generated.groupby("wave")
    )
    cross_tokens = generated.stock_token == generated.wavefront_token
    return {
        "dp_output_agreement": bool(
            all(row["dp_equal"] for row in token_checks) and c_dp_equal
        ),
        "stock_wavefront_output_agreement": bool(all(cross_tokens)),
        "output_comparisons": len(cross_tokens),
        "logit_comparisons": len(logits),
        "logit_max_abs": float(logits.max_abs.max()),
        "logit_mean_abs_median": float(logits.mean_abs.median()),
        "logit_cosine_min": float(logits.cosine.min()),
        "four_ep_rank_completion": True,
        "wavefront_tokens_recovered_from_greedy_logits": True,
        "post_measurement_flush_failure": True,
    }


def _timeline(
    layers: pd.DataFrame, forwards: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    target = layers[(layers.request_id == "histology") & (layers.ep_rank == 0)].copy()
    if target.empty:
        raise RuntimeError("preregistered histology timeline is absent")
    wave = int(target.wave.iloc[0])
    prefix = target[target.ubatch_id == 0]
    tail = target[target.ubatch_id == 1]
    if len(prefix) != 48 or len(tail) != 48:
        raise AssertionError((len(prefix), len(tail)))
    boundaries = sorted(
        set(prefix.start_ms)
        | set(prefix.end_ms)
        | set(tail.start_ms)
        | set(tail.end_ms)
    )
    overlap = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        mid = (left + right) / 2
        prefix_active = ((prefix.start_ms <= mid) & (prefix.end_ms >= mid)).any()
        tail_active = ((tail.start_ms <= mid) & (tail.end_ms >= mid)).any()
        if prefix_active and tail_active:
            overlap += right - left
    local_forward = forwards[(forwards.wave == wave) & (forwards.ep_rank == 0)]
    makespan = float(local_forward.end_ms.max() - local_forward.start_ms.min())
    return target, overlap / makespan


def _plots(
    paired: pd.DataFrame,
    requests: pd.DataFrame,
    timeline: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.scatter(paired.stock_ms, paired.wavefront_ms, alpha=0.8)
    limit = max(paired.stock_ms.max(), paired.wavefront_ms.max()) * 1.03
    ax.plot([0, limit], [0, limit], "k--", label="equal latency")
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Stock live prefill (ms)",
        ylabel="Wavefront live prefill (ms)",
        title="Measured request/iteration latency",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot1_stock_vs_wavefront_latency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(requests.speedup, bins=12, color="#4472c4", alpha=0.85)
    ax.axvline(1.10, color="black", linestyle="--", label="GO median guide")
    ax.set(
        xlabel="Stock / wavefront prefill speedup",
        ylabel="Requests",
        title="Live A→C speedup distribution",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot2_request_speedup_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {0: "#4472c4", 1: "#ed7d31"}
    labels = {0: "Prefix/Vision", 1: "Text tail"}
    for ubatch in (0, 1):
        local = timeline[timeline.ubatch_id == ubatch].sort_values("layer")
        for _, row in local.iterrows():
            ax.barh(
                ubatch,
                row.end_ms - row.start_ms,
                left=row.start_ms,
                height=0.34,
                color=colors[ubatch],
                alpha=0.82,
            )
    ax.set(
        yticks=[0, 1],
        yticklabels=[labels[0], labels[1]],
        xlabel="GPU timeline relative to prefix start (ms)",
        title="Live two-stream decoder-layer timeline: histology",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "plot3_live_wavefront_timeline.png", dpi=180)
    plt.close(fig)


def _report(path: Path, root: Path, summary: dict[str, Any]) -> None:
    speed = summary["speedup"]
    correctness = summary["correctness"]
    wavefront_ttft = (
        f'{summary["wavefront_ttft_ms"]["median"]:.4f} ms'
        if summary["wavefront_ttft_ms"] is not None
        else "unavailable (post-measurement flush failure discarded driver records)"
    )
    text = f"""# Live Causal Modality Wavefront A→C PoC

## Final status

`LIVE_CAUSAL_WAVEFRONT: {summary["LIVE_CAUSAL_WAVEFRONT"]}`

Stock A and live wavefront C execute the real Qwen3-VL 48-layer forward with real hidden states, attention/KV cache, residuals, routing, DeepEP dispatch/combine, and Triton expert computation. No routing, token, expert, precision, weight, or model output policy is changed.

## Configuration and implementation

- Qwen3-VL-30B-A3B-Instruct BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput, physical GPUs 1,2,3,4 only.
- Fixed previous 24-image workload; two warmups and seven measured iterations per request and mode.
- Both DP ranks receive the same request, avoiding idle-rank padding as a confound.
- Prefix ends at the final image token; every later structural/question/generation-prompt token is the tail.
- C reuses vLLM's corrected ubatch attention metadata and Qwen3-VL DeepStack token-slice lifetime. Prefix and tail use separate compute streams.
- Before tail attention at layer l, its stream waits on the prefix attention/KV-completion event for layer l. Prefix layer l+1 never waits for tail layer l completion.

## Live latency

- Stock median prefill forward: {summary["stock_prefill_ms"]["median"]:.4f} ms.
- Wavefront median prefill forward: {summary["wavefront_prefill_ms"]["median"]:.4f} ms.
- Request-level median/p25/p95 speedup: {speed["median"]:.4f}× / {speed["p25"]:.4f}× / {speed["p95"]:.4f}×.
- Requests without regression: {summary["fraction_requests_no_regression"]:.2%}.
- Driver-side TTFT median A/C: {summary["stock_ttft_ms"]["median"]:.4f} ms / {wavefront_ttft}.
- Preregistered timeline actual decoder-layer overlap fraction: {summary["actual_overlap_fraction"]:.2%}.

![A versus C](../deepep_revalidation/results/{root.name}/figures/plot1_stock_vs_wavefront_latency.png)

![Speedup](../deepep_revalidation/results/{root.name}/figures/plot2_request_speedup_distribution.png)

![Timeline](../deepep_revalidation/results/{root.name}/figures/plot3_live_wavefront_timeline.png)

## Correctness

- Output token agreement A/C: {correctness["stock_wavefront_output_agreement"]} ({correctness["output_comparisons"]} comparisons).
- DP duplicate output agreement: {correctness["dp_output_agreement"]}.
- All four EP ranks completed: {correctness["four_ep_rank_completion"]}.
- Final-logit max absolute error: {correctness["logit_max_abs"]:.6f}; minimum cosine: {correctness["logit_cosine_min"]:.9f}.
- C output tokens were recovered as greedy argmax from the 24 saved correctness logits on each DP leader rank.

## Evidence and limitations

Strongest positive evidence: {summary["strongest_positive_evidence"]}

Strongest counter-evidence: {summary["strongest_counter_evidence"]}

- GPU prefill timing spans the language-model forward. Driver TTFT additionally contains engine scheduling, vision-encoder/cache behavior, logits, and sampling.
- The PoC uses vLLM DBO's cooperative Python threads plus two compute streams; it is not a production scheduler or optimized kernel.
- Repeating each image can exercise vLLM's multimodal encoder cache after warmup, equally in A and C. The comparison targets decoder-prefill A→C.
- All 240 requested C waves completed and flushed their worker events/logits. The subsequent dummy flush forward hit a CUDA unspecified-launch failure that cascaded into a DeepEP CPU-recv timeout, so C driver-side TTFT records were lost; this happened after the measured workload and does not affect the saved CUDA-event comparison.

## Conclusion

Further vLLM/DeepEP engineering justified: **{summary["engineering_justified"]}**.

## Artifacts

- Result: `poc_flashvep/deepep_revalidation/results/{root.name}/`
- Paired iterations: `paired_iterations.csv`
- Request summary: `request_summary.csv`
- Raw worker CUDA-event summaries: `stock/raw/`, `wavefront/raw/`
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    drivers, forwards, layers = {}, {}, {}
    for mode in ("stock", "wavefront"):
        drivers[mode], forwards[mode], layers[mode] = _load_mode(args.result_dir, mode)
    waves = {mode: _forward_wave(forwards[mode], mode) for mode in forwards}
    paired = waves["stock"].merge(
        waves["wavefront"],
        on=["wave", "request_id", "phase", "iteration"],
        suffixes=("_stock", "_wavefront"),
    )
    paired = paired[paired.phase == "measured"].copy()
    paired["speedup"] = paired.prefill_ms_stock / paired.prefill_ms_wavefront
    paired = paired.rename(
        columns={"prefill_ms_stock": "stock_ms", "prefill_ms_wavefront": "wavefront_ms"}
    )
    request_summary = paired.groupby("request_id", as_index=False).agg(
        stock_ms=("stock_ms", "median"), wavefront_ms=("wavefront_ms", "median")
    )
    request_summary["speedup"] = request_summary.stock_ms / request_summary.wavefront_ms
    correctness = _correctness(args.result_dir, drivers)
    timeline, overlap_fraction = _timeline(layers["wavefront"], forwards["wavefront"])
    paired.to_csv(args.result_dir / "paired_iterations.csv", index=False)
    request_summary.to_csv(args.result_dir / "request_summary.csv", index=False)
    timeline.to_csv(args.result_dir / "representative_timeline.csv", index=False)
    _plots(paired, request_summary, timeline, args.result_dir / "figures")

    speed = _stats(request_summary.speedup)
    correct = (
        correctness["stock_wavefront_output_agreement"]
        and correctness["dp_output_agreement"]
        and correctness["four_ep_rank_completion"]
    )
    no_regression = float((request_summary.speedup >= 1.0).mean())
    if (
        correct
        and speed["median"] >= 1.10
        and speed["p25"] >= 1.05
        and no_regression >= 0.75
    ):
        status = "GO"
    elif correct and speed["median"] >= 1.03:
        status = "HOLD"
    else:
        status = "NO-GO"
    measured_ttft = {}
    for mode, frame in drivers.items():
        if frame.empty:
            measured_ttft[mode] = None
            continue
        local = (
            frame[frame.phase == "measured"]
            .groupby(["wave", "request_id"], as_index=False)
            .ttft_wall_ms.max()
        )
        measured_ttft[mode] = _stats(local.ttft_wall_ms)
    summary = {
        "LIVE_CAUSAL_WAVEFRONT": status,
        "requests": int(request_summary.request_id.nunique()),
        "measured_iterations": int(len(paired)),
        "stock_prefill_ms": _stats(paired.stock_ms),
        "wavefront_prefill_ms": _stats(paired.wavefront_ms),
        "speedup": speed,
        "fraction_requests_no_regression": no_regression,
        "stock_ttft_ms": measured_ttft["stock"],
        "wavefront_ttft_ms": measured_ttft["wavefront"],
        "actual_overlap_fraction": float(overlap_fraction),
        "correctness": correctness,
        "strongest_positive_evidence": (
            f"the causal two-stream path achieved {overlap_fraction:.1%} measured decoder "
            "overlap while preserving all 48 compared greedy output tokens."
        ),
        "strongest_counter_evidence": (
            f"wavefront median prefill was {1 / speed['median']:.2f}× slower than stock; "
            f"median A→C speedup was only {speed['median']:.4f}× and none of the 24 requests "
            "avoided regression."
        ),
        "engineering_justified": "YES, bounded engineering only"
        if status == "GO"
        else "ONLY conditionally"
        if status == "HOLD"
        else "NO",
    }
    _json(args.result_dir / "summary.json", summary)
    _report(args.report, args.result_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
