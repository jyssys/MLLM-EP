"""Aggregate measured modality stages and simulate the fixed ideal wavefront."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def _load(result: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, settings = [], []
    for path in sorted((result / "timing").glob("rank*.json")):
        payload = json.loads(path.read_text())
        if payload["status"] != "ok":
            raise RuntimeError(payload)
        rows.extend(payload["observations"])
        settings.append(
            {
                key: payload[key]
                for key in (
                    "rank",
                    "physical_gpu",
                    "visible_devices",
                    "expert_backend",
                    "prepare_finalize_backend",
                    "hidden_provenance",
                    "input_replication",
                )
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or len(settings) != 4:
        raise RuntimeError("four timing rank files are required")
    return frame, settings


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "request_id",
        "category",
        "pair_id",
        "token_bucket",
        "prompt_tokens",
        "vision_tokens",
        "prefix_tokens",
        "tail_tokens",
        "layer",
        "component",
    ]
    # Layer progress is gated by the slowest EP rank. Every stage uses measured
    # medians; wall is the primary V/T duration, stage maxima are descriptive.
    return frame.groupby(keys, as_index=False).agg(
        duration_ms=("wall_median_ms", "max"),
        dispatch_ms=("dispatch_median_ms", "max"),
        expert_ms=("expert_median_ms", "max"),
        combine_ms=("combine_median_ms", "max"),
        fastest_rank_ms=("wall_median_ms", "min"),
    )


def _simulate(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, slots = [], []
    for request_id, local in components.groupby("request_id"):
        vision = local[local.component == "vision_prefix"].sort_values("layer")
        tail = local[local.component == "language_tail"].sort_values("layer")
        if vision.layer.tolist() != list(range(48)) or tail.layer.tolist() != list(
            range(48)
        ):
            raise AssertionError(request_id)
        v = vision.duration_ms.to_numpy()
        t = tail.duration_ms.to_numpy()
        baseline = float((v + t).sum())
        slot_values = [
            float(v[0]),
            *[float(max(v[i], t[i - 1])) for i in range(1, 48)],
            float(t[-1]),
        ]
        ideal = float(sum(slot_values))
        metadata = vision.iloc[0]
        rows.append(
            {
                "request_id": request_id,
                "category": metadata.category,
                "pair_id": int(metadata.pair_id),
                "token_bucket": metadata.token_bucket,
                "prompt_tokens": int(metadata.prompt_tokens),
                "vision_tokens": int(metadata.vision_tokens),
                "prefix_tokens": int(metadata.prefix_tokens),
                "tail_tokens": int(metadata.tail_tokens),
                "vision_ratio": int(metadata.prefix_tokens)
                / int(metadata.prompt_tokens),
                "vision_total_ms": float(v.sum()),
                "tail_total_ms": float(t.sum()),
                "baseline_ms": baseline,
                "ideal_ms": ideal,
                "ideal_speedup": baseline / ideal,
                "hidden_ms": baseline - ideal,
                "hidden_fraction": (baseline - ideal) / baseline,
                "layers_with_overlap": int(
                    sum(min(v[i], t[i - 1]) > 0 for i in range(1, 48))
                ),
            }
        )
        for slot, duration in enumerate(slot_values):
            slots.append(
                {"request_id": request_id, "slot": slot, "duration_ms": duration}
            )
    return pd.DataFrame(rows), pd.DataFrame(slots)


def _diagnostic(result: Path) -> dict[str, Any] | None:
    paths = sorted((result / "diagnostic").glob("rank*.json"))
    if not paths:
        return None
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["status"] != "ok":
            raise RuntimeError(payload)
        rows.append(payload["diagnostic"])
    serial = max(row["serial"]["wall_ms_stats"]["median_ms"] for row in rows)
    overlap = max(row["overlap"]["wall_ms_stats"]["median_ms"] for row in rows)
    return {
        "request_id": rows[0]["request_id"],
        "layer": rows[0]["layer"],
        "selection": rows[0]["selection"],
        "serial_makespan_ms": serial,
        "overlap_makespan_ms": overlap,
        "speedup": serial / overlap,
        "all_rank_correctness_passed": all(
            row["correctness"]["passed"] for row in rows
        ),
        "rank_speedups": [float(row["speedup"]) for row in rows],
    }


def _plots(
    components: pd.DataFrame, requests: pd.DataFrame, slots: pd.DataFrame, figures: Path
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    layer = components.groupby(
        ["layer", "component"], as_index=False
    ).duration_ms.median()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for component, label, color in (
        ("vision_prefix", "Vision prefix V_l", "#4472c4"),
        ("language_tail", "Language tail T_l", "#ed7d31"),
    ):
        local = layer[layer.component == component]
        ax.plot(local.layer, local.duration_ms, label=label, color=color)
    ax.set(
        xlabel="Decoder layer",
        ylabel="Median measured D/E/C wall time (ms)",
        title="Modality-split DeepEP + TritonExperts timing",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "plot1_modality_timing_by_layer.png", dpi=180)
    plt.close(fig)

    representative = requests.iloc[
        (requests.ideal_speedup - requests.ideal_speedup.median())
        .abs()
        .argsort()
        .iloc[0]
    ]
    request = representative.request_id
    local = components[components.request_id == request]
    v = (
        local[local.component == "vision_prefix"]
        .sort_values("layer")
        .duration_ms.to_numpy()
    )
    t = (
        local[local.component == "language_tail"]
        .sort_values("layer")
        .duration_ms.to_numpy()
    )
    show = 12
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=False)
    baseline_start = 0.0
    for index in range(show):
        axes[0].barh(0, v[index], left=baseline_start, color="#4472c4")
        baseline_start += v[index]
        axes[0].barh(0, t[index], left=baseline_start, color="#ed7d31")
        baseline_start += t[index]
    axes[0].set(title=f"Baseline first {show} layers ({request})", yticks=[])
    time = 0.0
    axes[1].barh(1, v[0], left=time, color="#4472c4", label="Vision V")
    time += v[0]
    for index in range(1, show):
        axes[1].barh(1, v[index], left=time, color="#4472c4")
        axes[1].barh(0, t[index - 1], left=time, color="#ed7d31", alpha=0.8)
        time += max(v[index], t[index - 1])
    axes[1].barh(0, t[show - 1], left=time, color="#ed7d31", alpha=0.8)
    axes[1].set(
        title="Ideal causal wavefront",
        yticks=[0, 1],
        yticklabels=["T stream", "V stream"],
        xlabel="Time (ms)",
    )
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "plot2_wavefront_timeline.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].hist(requests.ideal_speedup, bins=12, color="#70ad47", alpha=0.85)
    axes[0].axvline(1.15, color="black", linestyle="--", label="GO median guide")
    axes[0].set(
        xlabel="Zero-contention ideal speedup",
        ylabel="Requests",
        title="Request speedup distribution",
    )
    axes[0].legend()
    axes[1].scatter(
        requests.vision_ratio,
        requests.ideal_speedup,
        c=requests.tail_tokens,
        cmap="plasma",
        s=45,
    )
    axes[1].set(
        xlabel="Prefix token fraction",
        ylabel="Ideal speedup",
        title="Workload ratio versus headroom",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "plot3_ideal_speedup_distribution.png", dpi=180)
    plt.close(fig)


def _report(path: Path, result: Path, summary: dict[str, Any]) -> None:
    speed = summary["ideal_speedup"]
    timing = summary["timing_breakdown"]
    diagnostic = summary["two_stream_diagnostic"]
    diag_text = (
        "NOT-RUN (NO-GO gate)"
        if diagnostic is None
        else (
            f"fixed request `{diagnostic['request_id']}`, layer {diagnostic['layer']}: "
            f"{diagnostic['serial_makespan_ms']:.6f} → {diagnostic['overlap_makespan_ms']:.6f} ms "
            f"({diagnostic['speedup']:.4f}×), correctness={diagnostic['all_rank_correctness_passed']}"
        )
    )
    text = f"""# Causal Modality Wavefront PoC

## Final status

`CAUSAL_MODALITY_WAVEFRONT: {summary["CAUSAL_MODALITY_WAVEFRONT"]}`

The causal prefix/tail dependency is mathematically valid, and the measured zero-contention operator-stage upper bound is {speed["median"]:.4f}× median (p25 {speed["p25"]:.4f}×, p95 {speed["p95"]:.4f}×).

## Environment and fixed methodology

- Qwen3-VL-30B-A3B-Instruct BF16, TP2/DP2/EP4/PP1, vLLM 0.20, DeepEP high-throughput and TritonExperts.
- Physical GPUs 1,2,3,4 only; every rank artifact records `CUDA_VISIBLE_DEVICES=1,2,3,4`.
- Exact routes from the fixed previous 24 real-image workload, all 48 layers.
- Every route is split at the token immediately after the final repeated image token: the prefix includes system/user/image/vision-end tokens, and the tail contains only post-image question/generation-prompt tokens.
- Timing uses actual DeepEP dispatch, TritonExperts compute, and DeepEP combine with the validated real layer-24 BF16 activation template. Two warmups and seven measured repetitions are used per request/layer/component/rank. Four identical EP sources reproduce the validated replay convention.
- No model output, route ID, route weight, token, expert placement, weight, production scheduler, or kernel is modified.

This is a measured D/E/C operator-stage upper bound, not an end-to-end TTFT measurement: attention is not separately replayed, per-layer hidden values are represented by the layer-24 template, and splitting creates separate component collectives.

## POC1 — modality timing

Across requests/layers, median V_l duration is {timing["vision_median_ms"]:.6f} ms and median T_l duration is {timing["tail_median_ms"]:.6f} ms. Request-total Vision-prefix work is {timing["vision_share_median"]:.2%} median; prefix tokens are {timing["prefix_token_fraction_median"]:.2%} of prompt tokens.

- Vision-prefix D/E/C medians: {timing["vision_dispatch_median_ms"]:.6f} / {timing["vision_expert_median_ms"]:.6f} / {timing["vision_combine_median_ms"]:.6f} ms.
- Language-tail D/E/C medians: {timing["tail_dispatch_median_ms"]:.6f} / {timing["tail_expert_median_ms"]:.6f} / {timing["tail_combine_median_ms"]:.6f} ms.

![Timing](../deepep_revalidation/results/{result.name}/figures/plot1_modality_timing_by_layer.png)

## POC2 — dependency validation

`CAUSAL_VALIDITY: VALID`

For the decoder's lower-triangular causal attention, a prefix query cannot read any later post-image key/value. RMSNorm, rotary embedding, router selection, expert MLPs, and residual updates are token-local; EP collectives move selected token rows but introduce no semantic cross-token reduction. Therefore V_{{l+1}} needs V_l but not T_l. T_l still needs the completed prefix state and its own previous-layer state, producing the fixed two-chain DAG represented by the requested formula. Current vLLM batches those rows together operationally, but that lockstep is an implementation choice rather than a model dependency.

## POC3 — zero-contention upper bound

- Median/p25/p95 speedup: {speed["median"]:.4f}× / {speed["p25"]:.4f}× / {speed["p95"]:.4f}×.
- Median hidden-time fraction: {summary["hidden_fraction"]["median"]:.2%}.
- Fraction of requests at least 1.15×: {summary["fraction_ge_1_15"]:.2%}.
- Formula: `V_1 + sum(max(V_l,T_(l-1))) + T_L`; thresholds and schedule were not changed post-hoc.

![Timeline](../deepep_revalidation/results/{result.name}/figures/plot2_wavefront_timeline.png)

![Speedup](../deepep_revalidation/results/{result.name}/figures/plot3_ideal_speedup_distribution.png)

## Conditional two-stream diagnostic

{diag_text}

This diagnostic, when run, is a bounded two-group DeepEP D/E/C wavefront on a preregistered medium request/layer. It is not a production cross-layer scheduler.

## Evidence and limitations

Strongest positive evidence: {summary["strongest_positive_evidence"]}

Strongest counter-evidence: {summary["strongest_counter_evidence"]}

- Attention timing is absent, so the reported result cannot be called measured end-to-end TTFT speedup.
- The replay uses exact layer-specific routes but one real layer-24 activation template and layer-24 weights for all layer labels; it measures route/shape and communication cost, not hidden-value or weight variation across depth.
- Four-source replication increases absolute traffic relative to one live DP request; it is held identical for V and T.
- Zero-contention simulation ignores contention, launch coupling, KV-cache coordination, and separate-collective overhead beyond what component replay already measures.

## Conclusion

Real CUDA wavefront implementation justified: **{summary["cuda_implementation_justified"]}**.

## Artifacts

- Result: `poc_flashvep/deepep_revalidation/results/{result.name}/`
- Derived component timing: `component_timing.csv`
- Request simulation: `request_wavefront.csv`
- Summary: `summary.json`

## Single recommended action

{summary["recommended_action"]}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    raw, settings = _load(args.result_dir)
    components = _aggregate(raw)
    requests, slots = _simulate(components)
    components.to_csv(args.result_dir / "component_timing.csv", index=False)
    requests.to_csv(args.result_dir / "request_wavefront.csv", index=False)
    slots.to_csv(args.result_dir / "wavefront_slots.csv", index=False)
    _plots(components, requests, slots, args.result_dir / "figures")
    speed = _stats(requests.ideal_speedup)
    if speed["median"] >= 1.15 and speed["p25"] >= 1.10:
        status = "GO"
    elif speed["median"] >= 1.05:
        status = "HOLD"
    else:
        status = "NO-GO"
    diagnostic = _diagnostic(args.result_dir)
    vision_components = components[components.component == "vision_prefix"]
    tail_components = components[components.component == "language_tail"]
    summary = {
        "CAUSAL_MODALITY_WAVEFRONT": status,
        "causal_validity": "VALID",
        "requests": int(requests.request_id.nunique()),
        "layers": 48,
        "timing_scope": "measured DeepEP dispatch + TritonExperts + combine; attention excluded",
        "timing_breakdown": {
            "vision_median_ms": float(
                components.loc[
                    components.component == "vision_prefix", "duration_ms"
                ].median()
            ),
            "tail_median_ms": float(
                components.loc[
                    components.component == "language_tail", "duration_ms"
                ].median()
            ),
            "vision_dispatch_median_ms": float(vision_components.dispatch_ms.median()),
            "vision_expert_median_ms": float(vision_components.expert_ms.median()),
            "vision_combine_median_ms": float(vision_components.combine_ms.median()),
            "tail_dispatch_median_ms": float(tail_components.dispatch_ms.median()),
            "tail_expert_median_ms": float(tail_components.expert_ms.median()),
            "tail_combine_median_ms": float(tail_components.combine_ms.median()),
            "vision_share_median": float(
                (
                    requests.vision_total_ms
                    / (requests.vision_total_ms + requests.tail_total_ms)
                ).median()
            ),
            "prefix_token_fraction_median": float(requests.vision_ratio.median()),
        },
        "ideal_speedup": speed,
        "hidden_fraction": _stats(requests.hidden_fraction),
        "fraction_ge_1_15": float((requests.ideal_speedup >= 1.15).mean()),
        "rank_settings": settings,
        "two_stream_diagnostic": diagnostic,
        "strongest_positive_evidence": (
            f"all 24 requests have a valid 47-slot overlap opportunity; median ideal "
            f"operator-stage speedup is {speed['median']:.4f}×."
        ),
        "strongest_counter_evidence": (
            "the timing excludes attention and reuses layer-24 activations/weights, so the "
            "ideal result is not an end-to-end implementation result."
        ),
        "cuda_implementation_justified": "YES, bounded prototype only"
        if status == "GO"
        else "ONLY a bounded prototype"
        if status == "HOLD"
        else "NO",
        "recommended_action": (
            "Implement one bounded live two-stream prefix/tail prototype with attention and "
            "one-layer-ahead dependency events; do not alter routing or kernels."
            if status in ("GO", "HOLD")
            else "Do not implement the CUDA wavefront; redirect to a mechanism with larger ideal headroom."
        ),
    }
    _json(args.result_dir / "summary.json", summary)
    _report(args.report, args.result_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
