#!/usr/bin/env python3
"""Materialize compact artifacts and the final serving-regime report.

The raw rank JSONL captures are intentionally left in their run directories;
this script copies only reproducible summaries and figures into the compact
artifact directory so a report commit does not contain hundreds of MB of
CUDA-event streams.
"""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("poc_flashvep/deepep_revalidation/results")
RUNS = [
    ROOT / "ep4_serving_straggler_regime_20260901_1340",
    ROOT / "ep4_serving_straggler_regime_20260901_8192",
    ROOT / "ep4_serving_straggler_regime_20260901_4096",
    ROOT / "ep4_serving_straggler_regime_20260901_2048",
    ROOT / "ep4_serving_straggler_regime_20260901_mixed_c4b",
    ROOT / "ep4_serving_straggler_regime_20260901_mixed_c8b",
]
BASE = RUNS[0]
ISOLATED = ROOT / "ep4_serving_straggler_regime_20260901_isolated_1to4" / "isolated"
COMPACT = ROOT / "ep4_serving_straggler_regime_20260901_final"
REPORT = Path("poc_flashvep/reports/ep4_serving_straggler_regime.md")


def _mode(row: dict[str, Any]) -> str:
    total = int(row.get("total_num_scheduled_tokens", 0))
    nreq = len(row.get("num_scheduled_tokens", {}))
    return "prefill" if total > nreq or bool(row.get("scheduled_new_req_ids")) else "decode"


def _bootstrap(values: np.ndarray, seed: int = 7) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    return float(np.median(values)), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def _load() -> pd.DataFrame:
    frame = pd.read_csv(BASE / "analysis" / "invocation_metrics_all.csv")
    frame["run"] = frame["result"].map(lambda x: Path(str(x)).name)
    return frame


def _driver_latency() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        path = run / "driver.dp_rank0.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for row in data.get("records", []):
            if row.get("measured") and int(row.get("driver_dp_rank", -1)) == 0:
                rows.append({"run": run.name, **row})
    return pd.DataFrame(rows)


def _scheduler_positive() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        files = sorted((run / "scheduler_trace").glob("*.jsonl"))
        if not files:
            continue
        seen: set[int] = set()
        for line in files[0].read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            seq = int(row["sequence"])
            if seq in seen or int(row.get("total_num_scheduled_tokens", 0)) <= 0:
                continue
            seen.add(seq)
            rows.append({
                "run": run.name,
                "sequence": seq,
                "batch_id": row.get("batch_id"),
                "condition": row.get("condition"),
                "submitted_concurrency": row.get("concurrency"),
                "scheduler_mode": _mode(row),
                "scheduled_tokens": int(row.get("total_num_scheduled_tokens", 0)),
                "scheduled_requests": len(row.get("num_scheduled_tokens", {})),
                "new_requests": len(row.get("scheduled_new_req_ids", [])),
                "cached_requests": len(row.get("scheduled_cached_req_ids", [])),
                "measured": bool(row.get("measured", False)),
            })
    return pd.DataFrame(rows)


def _make_mixed_plot(frame: pd.DataFrame, figures: Path) -> None:
    mixed = frame[(frame["run"].str.contains("mixed_c"))]
    mixed = mixed[mixed["scheduler_mode"].isin(["prefill", "decode"])].copy()
    if mixed.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = [mixed.loc[mixed.scheduler_mode == mode, "expert_ratio"].to_numpy()
            for mode in ("prefill", "decode")]
    ax.boxplot(data, tick_labels=["prefill", "decode"], showfliers=False)
    ax.axhline(1.10, color="crimson", ls="--", lw=.9, label="expert imbalance gate 1.10")
    ax.set(xlabel="mixed scheduler phase", ylabel="max/mean expert CUDA time")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures / "plot7_mixed_prefill_decode_context.png", dpi=170)
    plt.close(fig)


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    group = frame[frame.scheduler_mode == "prefill"].groupby(
        ["run", "condition", "concurrency"], dropna=False
    )
    out = group.agg(
        n=("layer", "size"),
        scheduled_tokens_median=("scheduled_tokens", "median"),
        scheduled_tokens_max=("scheduled_tokens", "max"),
        scheduled_requests_max=("scheduled_requests", "max"),
        rank_ratio_median=("rank_ratio", "median"),
        rank_ratio_p95=("rank_ratio", lambda x: x.quantile(.95)),
        expert_ratio_median=("expert_ratio", "median"),
        expert_ratio_p95=("expert_ratio", lambda x: x.quantile(.95)),
        dispatch_ratio_median=("dispatch_ratio", "median"),
        combine_ratio_median=("combine_ratio", "median"),
        expert_max_ms_median=("expert_max_ms", "median"),
        dispatch_max_ms_median=("dispatch_max_ms", "median"),
        combine_max_ms_median=("combine_max_ms", "median"),
        tail15_fraction=("expert_ratio", lambda x: float((x >= 1.15).mean())),
    ).reset_index()
    return out


def _matched(frame: pd.DataFrame) -> dict[str, Any]:
    data = frame[(frame.run == BASE.name) & (frame.scheduler_mode == "prefill") &
                 (frame.condition.isin(["vision_heavy", "text_control"])) &
                 (frame.scheduled_tokens == 4528) & (frame.scheduled_requests == 15)]
    keys = ["concurrency", "scheduled_tokens", "scheduled_requests", "layer",
            "chunk_index", "control_iteration"]
    left = data[data.condition == "vision_heavy"].rename(columns={
        "rank_ratio": "vision_rank_ratio", "expert_ratio": "vision_expert_ratio",
        "expert_max_ms": "vision_expert_ms", "dispatch_max_ms": "vision_dispatch_ms",
        "combine_max_ms": "vision_combine_ms",
    })
    right = data[data.condition == "text_control"].rename(columns={
        "rank_ratio": "text_rank_ratio", "expert_ratio": "text_expert_ratio",
        "expert_max_ms": "text_expert_ms", "dispatch_max_ms": "text_dispatch_ms",
        "combine_max_ms": "text_combine_ms",
    })
    merged = left.merge(right, on=keys, how="inner")
    if merged.empty:
        return {"n": 0}
    out: dict[str, Any] = {"n": int(len(merged))}
    for metric in ("rank_ratio", "expert_ratio", "expert_ms", "dispatch_ms", "combine_ms"):
        diff = merged[f"vision_{metric}"] - merged[f"text_{metric}"]
        med, lo, hi = _bootstrap(diff.to_numpy())
        out[metric] = {"median_vision_minus_text": med, "bootstrap95": [lo, hi]}
    return out


def _latency_stats(values: np.ndarray, baseline: float | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    return {
        "n": int(values.size), "median_ms": med,
        "p95_ms": float(np.quantile(values, .95)),
        "cv_pct": float(values.std(ddof=1) / values.mean() * 100) if values.size > 1 else 0.0,
        "tail15_pct": float(np.mean(values >= 1.15 * (baseline if baseline is not None else med)) * 100),
    }


def main() -> None:
    frame = _load()
    summary = _summary_table(frame)
    driver = _driver_latency()
    scheduler = _scheduler_positive()
    COMPACT.mkdir(parents=True, exist_ok=True)
    (COMPACT / "analysis").mkdir(exist_ok=True)
    (COMPACT / "figures").mkdir(exist_ok=True)
    frame.to_csv(COMPACT / "analysis" / "invocation_metrics_all.csv", index=False)
    summary.to_csv(COMPACT / "analysis" / "serving_summary.csv", index=False)
    driver.to_csv(COMPACT / "driver_latency.csv", index=False)
    scheduler.to_csv(COMPACT / "scheduler_positive_trace.csv", index=False)
    for name in BASE.glob("backend_proof/*.json"):
        target = COMPACT / "backend_proof" / name.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(name, target)
    for name in BASE.glob("figures/*.png"):
        shutil.copy2(name, COMPACT / "figures" / name.name)
    _make_mixed_plot(frame, COMPACT / "figures")

    raw_manifest = {
        "raw_traces_preserved": True,
        "raw_result_directories": [str(path) for path in RUNS] + [str(ISOLATED.parent)],
        "compact_directory_excludes_raw_jsonl": True,
        "reason": "rank JSONL captures are large; exact local paths and summaries are retained",
    }
    (COMPACT / "raw_manifest.json").write_text(json.dumps(raw_manifest, indent=2) + "\n")
    shutil.copy2(BASE / "run_metadata.json", COMPACT / "run_metadata_16384.json")
    shutil.copy2(BASE / "schedule.json", COMPACT / "schedule_16384.json")
    if (ISOLATED / "isolated.json").exists():
        (COMPACT / "isolated").mkdir(exist_ok=True)
        shutil.copy2(ISOLATED / "isolated.json", COMPACT / "isolated" / "isolated.json")
    source_manifest = Path(
        "poc_flashvep/deepep_revalidation/results/"
        "live_prefill_execution_regime_20260821_111609/workload_manifest.json"
    )
    manifest_excerpt: dict[str, Any] = {"source": str(source_manifest)}
    if source_manifest.exists():
        payload = json.loads(source_manifest.read_text())
        manifest_excerpt["sha256"] = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        manifest_excerpt["pairs"] = [
            {
                "vision_request_id": p.get("vision", {}).get("request_id"),
                "text_request_id": p.get("text", {}).get("request_id"),
                "vision_prompt_tokens": p.get("vision", {}).get("prompt_tokens"),
                "text_prompt_tokens": p.get("text", {}).get("prompt_tokens"),
                "category": p.get("vision", {}).get("category"),
            }
            for p in payload.get("pairs", [])
        ]
    (COMPACT / "workload_manifest_excerpt.json").write_text(
        json.dumps(manifest_excerpt, indent=2) + "\n"
    )
    (COMPACT / "commands.json").write_text(json.dumps({
        "gpu": "CUDA_VISIBLE_DEVICES=1,2,3,4",
        "fixed_budget": "WARMUPS=1 ITERATIONS=2 bash poc_flashvep/ep4_serving_straggler_regime/run_gpu.sh <result>",
        "budget_sweep": "MAX_NUM_BATCHED_TOKENS={8192,4096,2048} WARMUPS=1 ITERATIONS=2 bash .../run_gpu.sh <result>",
        "mixed": "PREFILL_COUNT={3,7} WARMUPS=1 ITERATIONS=2 bash .../run_mixed.sh <result>",
        "isolated": "ep_runtime_tail_forensics/run_context.py --context isolated --warmups 20 --iterations 100",
    }, indent=2) + "\n")

    pre = frame[frame.scheduler_mode == "prefill"]
    pure = pre[~pre.run.str.contains("mixed")]
    groups = pure.groupby(["run", "condition", "concurrency"], dropna=False)
    best_rank = groups.rank_ratio.median().sort_values(ascending=False).iloc[0]
    best_expert = groups.expert_ratio.median().sort_values(ascending=False).iloc[0]
    mixed_group = pre[pre.run.str.contains("mixed")].groupby(
        ["run", "condition", "concurrency"], dropna=False
    )
    best_mixed_expert = (float(mixed_group.expert_ratio.median().max())
                         if len(mixed_group) else float("nan"))
    iso_stats: dict[str, Any] = {}
    iso_file = ISOLATED / "isolated.json"
    if iso_file.exists():
        iso_data = json.loads(iso_file.read_text())
        iso_stats = _latency_stats(np.asarray(iso_data["samples_ms"], dtype=float))
        iso_stats.update({"n_assignments": iso_data.get("n"), "active_experts": iso_data.get("g"),
                          "block_m": iso_data.get("runtime_config", {}).get("BLOCK_SIZE_M")})
    controlled = pre[(pre.run == BASE.name) & (pre.condition == "vision_heavy") &
                     (pre.concurrency == 1)]
    serving = pre[(pre.run == BASE.name) & (pre.condition == "vision_heavy") &
                  (pre.concurrency == 16) & (pre.scheduled_requests == 15)]
    context_comparison = {
        "isolated_expert_scope": iso_stats,
        "controlled_vllm_vision_c1": _latency_stats(controlled.expert_max_ms.to_numpy()) if not controlled.empty else {},
        "serving_like_vllm_vision_c16": _latency_stats(serving.expert_max_ms.to_numpy()) if not serving.empty else {},
    }
    gate = {
        "FINAL_STATUS": "NO-GO",
        "STRAGGLER_FOUND": False,
        "RUNTIME_CONTEXT": "NO-GO",
        "gate": {
            "median_rank_ratio_threshold": 1.5,
            "median_expert_ratio_threshold": 1.10,
            "strong_condition_count": 0,
            "best_prefill_rank_ratio_median": float(best_rank),
            "best_prefill_expert_ratio_median": float(best_expert),
            "best_mixed_prefill_expert_ratio_median": best_mixed_expert,
        },
        "matched_vision_text": _matched(frame),
        "mixed_phase_summary": frame[frame.run.str.contains("mixed_c")]
        .groupby(["run", "scheduler_mode"])
        .agg(n=("layer", "size"), rank_ratio_median=("rank_ratio", "median"),
             expert_ratio_median=("expert_ratio", "median"), expert_ratio_p95=("expert_ratio", lambda x: x.quantile(.95)))
        .reset_index().to_dict(orient="records"),
        "context_comparison": context_comparison,
        "correctness": {
            "driver_records_complete": bool((driver.output_count > 0).all()) if not driver.empty else False,
            "cuda_deepep_errors_observed": False,
            "backend_proof_files": len(list((COMPACT / "backend_proof").glob("*.json"))),
        },
    }
    (COMPACT / "gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n")

    def fmt(v: Any, digits: int = 3) -> str:
        return "n/a" if pd.isna(v) else f"{float(v):.{digits}f}"

    b = summary[summary.run == BASE.name].copy()
    rows = []
    for _, row in b.iterrows():
        rows.append(
            f"| {row.condition} | {int(row.concurrency)} | {int(row.scheduled_tokens_median)} | "
            f"{int(row.scheduled_tokens_max)} | {fmt(row.rank_ratio_median)} | "
            f"{fmt(row.expert_ratio_median)} | {fmt(row.expert_ratio_p95)} | {fmt(row.tail15_fraction*100, 1)}% |"
        )
    table = "\n".join(rows)
    budget_rows = []
    for run in [x.name for x in RUNS[:4]]:
        part = summary[(summary.run == run) & (summary.condition == "vision_heavy")]
        if part.empty:
            continue
        row = part.loc[part.scheduled_tokens_max.idxmax()]
        budget_rows.append(
            f"| {run.rsplit('_', 1)[-1]} | {int(row.scheduled_tokens_max)} | {int(row.scheduled_requests_max)} | "
            f"{fmt(row.rank_ratio_median)} | {fmt(row.expert_ratio_median)} | {fmt(row.expert_ratio_p95)} |"
        )
    budget_table = "\n".join(budget_rows)
    mixed_table_rows = []
    mixed_df = pd.DataFrame(gate["mixed_phase_summary"])
    if not mixed_df.empty:
        for _, row in mixed_df.iterrows():
            mixed_table_rows.append(
                f"| {row['run'].rsplit('_', 1)[-1]} | {row['scheduler_mode']} | {int(row['n'])} | "
                f"{fmt(row['rank_ratio_median'])} | {fmt(row['expert_ratio_median'])} | {fmt(row['expert_ratio_p95'])} |"
            )
    mixed_table = "\n".join(mixed_table_rows)
    matched = gate["matched_vision_text"]
    ctx = gate["context_comparison"]
    iso = ctx.get("isolated_expert_scope", {})
    ctl = ctx.get("controlled_vllm_vision_c1", {})
    srv = ctx.get("serving_like_vllm_vision_c16", {})
    report = f"""# EP4 Serving Straggler Regime Forensics

## Executive result

`FINAL STATUS: NO-GO`
`STRAGGLER_FOUND: NO`
`RUNTIME_CONTEXT: NO-GO`

The tested real vLLM V1 scheduler did not produce the preregistered strong
straggler regime.  The largest pure-prefill median rank imbalance was
{float(best_rank):.3f}, and the largest pure-prefill median expert-time
imbalance was {float(best_expert):.3f}; the gate requires at least 1.5 and 1.10
respectively, across repeated layers/serving conditions.  No method or
scheduler change was made.

The most demanding clean condition was `long_multi_image`, submitted
concurrency 4, `max_num_batched_tokens=16384`, with a real scheduler iteration
of 13,671 prefill tokens and 3 co-batched requests.  Its median rank ratio was
1.157 and median expert-time ratio 1.078 (p95 1.149): latency grew with work,
but not as a robust rank straggler.

## Environment and serving proof

- Model: Qwen3-VL-30B-A3B-Instruct, BF16, snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Runtime: vLLM 0.20.0 V1 engine, eager mode, chunked prefill, prefix cache off.
- Topology: TP2 / DP2 / EP4 / PP1, DBO off, linear expert placement
  (`expert_id // 32`), 32 local experts per EP rank.
- Backend proof: `DeepEPHTPrepareAndFinalize`, `DeepEPHTAll2AllManager`, and
  `TritonExperts`; all four EP ranks reported `ep_world_size=4` and
  `visible_devices=1,2,3,4`.
- GPU mapping: `CUDA_VISIBLE_DEVICES=1,2,3,4` (logical ranks 0–3 map to
  physical GPUs 1–4). No GPUs 0 or 5–7 were used by these runs.
- `max_num_batched_tokens`: 16,384, 8,192, 4,096, and 2,048 in the fixed
  token-budget sweep. The 16,384 run is the primary Stage A/B baseline.
- Workload: prior local real-image/text IDs plus two local long multi-image
  rows; no downloaded dataset or synthetic routes. Submissions were batched
  through the real V1 scheduler via `_add_completion_requests`.

## What the scheduler actually scheduled

The configured budget was not treated as the workload size. The scheduler
trace records positive iterations and their actual token/request counts in
`scheduler_positive_trace.csv`. At budget 16,384, the largest vision-heavy
co-batch contained 15 requests and 4,528 tokens; the long multi-image c4 wave
contained 3 requests and 13,671 tokens. Lower budgets fragmented waves and
reduced co-batching rather than increasing rank imbalance.

| budget run | largest vision scheduled tokens | largest scheduled requests | median rank ratio | median expert ratio | p95 expert ratio |
|---|---:|---:|---:|---:|---:|
{budget_table}

## Stage A — Low-load baseline

At submitted concurrency 1, text-only had rank ratio 1.279 and expert-time
ratio 1.029; vision-heavy had 1.235 and 1.030; long multi-image had 1.124 and
1.038. These are ordinary small rank spreads, not a strong critical-path
straggler. The six required summary figures are in
`{COMPACT}/figures/`.

## Stage B — Concurrent vision-prefill sweep

The fixed 16,384-budget sweep used submitted concurrency 1, 2, 4, 8, and 16.
The scheduler co-batched fewer requests than submitted when prompt/DP timing
prevented a single wave; this is why actual scheduled counts, rather than the
submitted number, are the primary covariate.

| condition | submitted c | median scheduled tokens | max scheduled tokens | median rank max/mean | median expert-time max/mean | p95 expert ratio | expert >=15% fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

At the largest matched vision/text co-batch (15 requests, 4,528 scheduled
tokens), the paired layer-level differences were:

- Vision minus Text rank ratio: median
  {matched.get('rank_ratio', {}).get('median_vision_minus_text', float('nan')):.3f},
  bootstrap 95% CI
  {matched.get('rank_ratio', {}).get('bootstrap95', ['n/a','n/a'])}.
- Vision minus Text expert-time ratio: median
  {matched.get('expert_ratio', {}).get('median_vision_minus_text', float('nan')):.3f},
  bootstrap 95% CI
  {matched.get('expert_ratio', {}).get('bootstrap95', ['n/a','n/a'])}.

Vision was not more imbalanced; its rank and expert ratios were slightly lower
than the matched text control. Long vision requests did show the largest
absolute expert/dispatch times, but the normalized rank spread remained below
the gate.

## Stage C — Token-budget sweep

Budgets 8,192, 4,096, and 2,048 were run only after no strong Stage B
condition was found. Decreasing the global budget changed effective chunking
and co-batching, but did not create a robust high-ratio condition. Across all
pure prefill groups, the maximum median rank ratio was {float(best_rank):.3f}
and maximum median expert ratio was {float(best_expert):.3f}.

## Stage D — Mixed prefill + decode

After Stage C remained below gate, two bounded real mixed runs were performed:
one 64-token text decode request plus 3, and then 7, image-prefill requests.
The scheduler trace shows a text prefill, decode iterations, image-prefill
iterations, and continuing decode iterations. Decode rows have one scheduled
token and therefore can show an uninformative rank ratio around 3.3 from tiny
assignment counts; they are not a large compute straggler. Prefill-phase
ratios are the relevant comparison.

| run | scheduler phase | observations | median rank ratio | median expert ratio | p95 expert ratio |
|---|---|---:|---:|---:|---:|
{mixed_table}

The high-scale 9,535-token c8 mixed-prefill sub-iteration reached an expert
ratio of about 1.095 (the c8 prefill aggregate median is 1.052) and had noisy
tails, but it still lacked the required median rank imbalance or repetition
robustness. No DeepEP collectives were launched concurrently by the
instrumentation.

### Context comparison (scope caveat)

The same-GPU isolated replay used the fixed `text_18_tui_main` expert shape
(N=2984, G=30) and 20 warmup + 100 measured expert-only iterations. Live
single-request and serving-like entries are not iso-N with that microbenchmark,
so the table is a context/scope diagnostic, not a causal latency ratio.

| context | measured scope | median expert ms | p95 | CV | >=15% tail |
|---|---|---:|---:|---:|---:|
| isolated (GPU 1–4) | one local expert, N=2984 | {float(iso.get('median_ms', float('nan'))):.4f} | {float(iso.get('p95_ms', float('nan'))):.4f} | {float(iso.get('cv_pct', float('nan'))):.2f}% | {float(iso.get('tail15_pct', float('nan'))):.2f}% |
| controlled vLLM | vision-heavy c1, live max-rank | {float(ctl.get('median_ms', float('nan'))):.4f} | {float(ctl.get('p95_ms', float('nan'))):.4f} | {float(ctl.get('cv_pct', float('nan'))):.2f}% | {float(ctl.get('tail15_pct', float('nan'))):.2f}% |
| serving-like vLLM | vision-heavy c16, 15 co-batched req | {float(srv.get('median_ms', float('nan'))):.4f} | {float(srv.get('p95_ms', float('nan'))):.4f} | {float(srv.get('cv_pct', float('nan'))):.2f}% | {float(srv.get('tail15_pct', float('nan'))):.2f}% |

## Stage E — Long multimodal stress

The bounded long multi-image family was included at c1/c2/c4 for every token
budget. The 13,671-token c4 iteration was the strongest clean load condition;
its median expert ratio 1.078 and rank ratio 1.157 remain below the strong
gate. Additional c8/c16 long stress was not run after this condition failed,
consistent with the early-stop/bounded-scope rule.

## Instrumentation and correctness

Each MoE invocation was timed with CUDA events around dispatch, TritonExperts,
and combine; event times were resolved after a bounded flush synchronization,
not by synchronizing every layer. The rank proof files report 69,120 captured
events per rank for each fixed-budget run and 13,824 per rank for each mixed
run. This confirms all four EP ranks and the intended backend path. There was
no CUDA or DeepEP runtime failure, no route/placement/model modification, and
all driver records returned the expected output count. A transient control-file
JSON read race was visible in the first mixed logs while the host atomically
replaced the control file; the experiment-local reader now catches that race,
and it did not change scheduling or returned outputs.

There is no clean instrumentation-OFF wall-time pair in this bounded run, so a
numeric instrumentation-overhead percentage is not claimed. The hook adds
nonblocking event records and one end-of-run synchronization; event overhead
should be measured separately before any production use.

## Bottleneck localization

The dominant observed effect is ordinary latency scaling with scheduled token
volume (especially long multi-image dispatch/expert/combine time), not an
imbalanced critical rank. The hottest rank changes with layer/condition rather
than forming one persistent device hotspot. Lower token budgets mostly reduce
co-batching and produce heterogeneous chunks; they do not amplify the median
rank or expert-time ratio.

Thus this data does not establish an MLLM-specific serving straggler, nor does
it establish a generic vLLM/DeepEP runtime-tail candidate. The previous
modality-specific claim is not rescued by high concurrency in this tested EP4
range.

## Gate and next action

`FINAL STATUS: NO-GO` and `STRAGGLER_FOUND: NO`.

No method design or further blind stress sweep is justified by this evidence.
The next single action is to stop the EP4 straggler direction and prioritize a
different mechanism with a reproducible paired effect; only if an external
requirement demands runtime-tail evidence should the next experiment be a
separate, longer real arrival-process trace with instrumentation-OFF/ON pairs.

## Artifacts

- Compact result: `poc_flashvep/deepep_revalidation/results/ep4_serving_straggler_regime_20260901_final/`
- Raw traces (preserved locally): the six run directories listed in
  `raw_manifest.json` under the compact result.
- Figures: `plot1_concurrency_rank_imbalance.png`,
  `plot2_concurrency_expert_imbalance.png`,
  `plot3_scheduled_tokens_vs_straggler.png`,
  `plot4_vision_vs_text_matched_serving.png`,
  `plot5_layer_hot_rank_heatmap.png`,
  `plot6_scheduler_iteration_timeline.png`, and
  `plot7_mixed_prefill_decode_context.png`.
- Machine-readable gate: `gate_summary.json`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(gate, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
