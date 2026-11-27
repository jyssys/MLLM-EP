"""Render compact per-arm IDEFICS peer-copy and numerical evidence."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from statistics import median


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def lookup(value, *parts):
    for part in parts:
        if value is None:
            return 0
        value = value[part]
    return value


def interval(values):
    if len(values) < 2:
        return None
    rng = random.Random(20261127)
    samples = sorted(median(rng.choices(values, k=len(values))) for _ in range(2000))
    return [samples[49], samples[1949]]


def summary(runs, output, receipt_path):
    output.mkdir(parents=True, exist_ok=True)
    raw = []
    records = []
    compact = []
    counts = {}
    comparison = defaultdict(dict)
    for run in runs:
        complete = json.loads((run / "complete.json").read_text())
        config = json.loads((run / "config.json").read_text())
        arch = json.loads((run / "architecture.json").read_text())
        warmup = json.loads((run / "warmup.json").read_text())
        lease = run.parent / f"{run.name}_gpu_lease"
        command_log = lease / "logs/command.log"
        log_marker = next((line for line in command_log.read_text().splitlines()
                           if line.startswith("IDEFICS_DETERMINISTIC")), None) if command_log.exists() else None
        raw.append({
            "run": run.name, "config": config, "architecture": arch, "complete": complete,
            "warmup_pixel_preflight": warmup["stock_full_pixel_preflight"], "requests": {},
            "determinism_log_record": log_marker,
            "command_log_sha256": digest(command_log) if command_log.exists() else None,
        })
        for path in sorted(run.glob("request_*.json")):
            row = json.loads(path.read_text())
            key = row["request_id"]
            state_witness = row.get("full_projected_state_witness", {})
            native_trials = [arm for arm in row["experiments"] if arm["mode"] == "native_recompute"]
            a_a_stable = bool(native_trials) and all(
                arm["first_token_p_logits_exact"] and arm["correctness"]["logits_bitwise"]
                and arm["correctness"]["self_cache_exact_layers"] == arm["correctness"]["self_cache_layers"]
                for arm in native_trials
            )
            raw[-1]["requests"][path.name] = {"id": key, "sha256": digest(path)}
            compact.append({
                "run": run.name, "request_id": key, "source": row["study_source"],
                "text_tokens": row["text_prompt_tokens"], "image_count": row["vision_latent_shape"][1],
                "visual_rows_per_image": row["vision_latent_shape"][2],
                "prefill_mask_rows_per_image": json.dumps(row["per_image_visible_prompt_rows"]),
                "decode_last_mask": json.dumps(row["last_image_mask"]),
                "decode_visible_images": sum(row["last_image_mask"]), "first_eos_or_none": row["first_eos_or_none"],
                "vision_gpu_ms": row["vision_perceiver"]["cuda_ms"],
                "independent_d_exact_post_norm_k_layers": state_witness.get("exact_post_norm_k_layers"),
                "independent_d_exact_v_layers": state_witness.get("exact_v_layers"),
                "cross_layers": state_witness.get("cross_layers"),
                "native_a_a_stable": a_a_stable,
            })
            arms = defaultdict(list)
            comparison[(run.name, key)]["image_count"] = row["vision_latent_shape"][1]
            for arm in row["experiments"]:
                check = arm["correctness"]
                stats = arm["d_cross_projection"]
                visual_raw = arm["visual_raw_copy"]
                projected = arm["visual_projected_copy"]
                record = {
                    "run": run.name, "request_id": key, "mode": arm["mode"], "repeat": arm["repeat"],
                    "images": row["vision_latent_shape"][1], "transferred_slots": len(arm["decode_source_slots"]),
                    "native_a_a_stable": a_a_stable,
                    "prompt_tokens": row["text_prompt_tokens"], "teacher_steps": check["step_reference_count"],
                    "prefill_cuda_ms": arm["text_prefill"]["cuda_ms"], "decode_cuda_ms": arm["decode"]["cuda_ms"],
                    "prefill_wall_ms": arm["text_prefill"]["wall_ms"],
                    "prefill_plus_d_wall_ms": arm["text_prefill"]["wall_ms"] + arm["measured_post_prefill_wall_ms"],
                    "decode_wall_ms": arm["decode"]["wall_ms"], "post_prefill_wall_ms": arm["measured_post_prefill_wall_ms"],
                    "text_kv_copy_ms": arm["text_kv_copy"]["wall_ms"], "text_kv_logical_bytes": arm["text_kv_copy"]["logical_bytes"],
                    "visual_raw_copy_ms": visual_raw["wall_ms"], "visual_raw_logical_bytes": visual_raw["logical_bytes"],
                    "visual_projected_copy_ms": lookup(projected, "wall_ms"),
                    "visual_projected_logical_bytes": lookup(projected, "logical_bytes"),
                    "pack_on_p_cuda_ms": lookup(arm["pack_on_P"], "cuda_ms"),
                    "project_on_d_cuda_ms": lookup(arm["packed_project_on_D"], "cuda_ms"),
                    "restore_mask_geometry_cuda_ms": lookup(arm.get("restore_masked_geometry_on_D"), "cuda_ms"),
                    "cross_cache_logical_bytes": arm["d_cross_cache_resident_bytes"],
                    "cross_cache_storage_bytes": arm.get("d_cross_cache_storage_bytes"),
                    "d_projection_calls": stats["projection_calls"], "d_projection_hits": stats["cache_hits"],
                    "d_kv_gemms": stats["individual_gemms"], "d_fused_gemms": stats["fused_gemms"],
                    "d_norm_calls": stats["key_norm_calls"], "d_k_norm_evaluations": stats["actual_key_norm_evaluations"],
                    "prefill_logits_bitwise": arm["first_token_p_logits_exact"], **check,
                }
                records.append(record)
                arms[arm["mode"]].append(record)
                group = counts.setdefault((run.name, arm["mode"]), dict(trials=0, bitwise_trials=0,
                    greedy_matches=0, teacher_steps=0, exact_kv_layers=0, kv_layers=0, max_logit_error=0.0,
                    stable_native_trials=0, stable_native_bitwise_trials=0, stable_native_exact_kv_layers=0))
                group["trials"] += 1
                group["bitwise_trials"] += int(check["logits_bitwise"] and arm["first_token_p_logits_exact"])
                group["greedy_matches"] += check["greedy_matches"]
                group["teacher_steps"] += check["step_reference_count"]
                group["exact_kv_layers"] += check["self_cache_exact_layers"]
                group["kv_layers"] += check["self_cache_layers"]
                group["max_logit_error"] = max(group["max_logit_error"], check["logit_max_abs"])
                if a_a_stable:
                    group["stable_native_trials"] += 1
                    group["stable_native_bitwise_trials"] += int(check["logits_bitwise"] and arm["first_token_p_logits_exact"])
                    group["stable_native_exact_kv_layers"] += check["self_cache_exact_layers"]
            for name, trials in arms.items():
                comparison[(run.name, key)][name] = {
                    metric: median(item[metric] for item in trials)
                    for metric in ("post_prefill_wall_ms", "prefill_plus_d_wall_ms")
                }
        if len(raw[-1]["requests"]) != complete["requests"]:
            raise ValueError("Raw ledger and completed requests disagree")
    pairs = (
        ("generic_memoized_d", "native_recompute"), ("generic_packed_d", "native_recompute"),
        ("generic_memoized_d", "generic_packed_d"), ("projected_handoff_p", "generic_memoized_d"),
        ("projected_handoff_p", "generic_packed_d"), ("generic_visibility_cached_d", "generic_packed_d"),
        ("projected_visibility_handoff_p", "generic_visibility_cached_d"),
        ("generic_visibility_padded_d", "generic_packed_d"),
        ("generic_visibility_padded_d", "generic_memoized_d"),
        ("projected_visibility_padded_p", "generic_visibility_padded_d"),
        ("projected_visibility_padded_p", "generic_packed_d"),
        ("projected_visibility_padded_p", "generic_memoized_d"),
    )
    ratios = []
    for run in runs:
        for candidate, baseline in pairs:
            for stratum, minimum_images in (("all", 1), ("multiple_images", 2)):
                for metric in ("post_prefill_wall_ms", "prefill_plus_d_wall_ms"):
                    values = [arms[candidate][metric] / arms[baseline][metric]
                              for (name, _), arms in comparison.items()
                              if (name == run.name and {candidate, baseline} <= arms.keys()
                                  and arms["image_count"] >= minimum_images)]
                    if values:
                        ratios.append(dict(run=run.name, candidate=candidate, baseline=baseline,
                                           n_requests=len(values), stratum=stratum, metric=metric,
                                           wall_ratio_median=median(values), request_bootstrap_95_ci=interval(values),
                                           ratio_min=min(values), ratio_max=max(values)))
    for filename, rows in (("trials.csv", records), ("mask_geometry.csv", compact)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    output_summary = {
        "evidence": "ISOLATED_GPU_MEASUREMENT", "runs": raw,
        "control": "Matched two-GPU text self-KV and fixed native token prefix; CUDA event stream plus synchronized host timings.",
        "numerical": [{"run": run, "mode": mode, **fields} for (run, mode), fields in counts.items()],
        "paired_request_wall_ratios": ratios,
        "model_receipt_SHA256": digest(receipt_path),
    }
    (output / "summary.json").write_text(json.dumps(output_summary, ensure_ascii=False, indent=2) + "\n")
    return records


def figure(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = defaultdict(set)
    for row in rows:
        cases[row["run"]].add(row["request_id"])
    selected_run = max(cases, key=lambda name: len(cases[name]))
    arms = {
        "native_recompute": ("Stock recompute", "#53585d"),
        "generic_packed_d": ("Generic packed on D", "#347394"),
        "projected_handoff_p": ("Whole projected P→D", "#b75d31"),
        "generic_visibility_padded_d": ("Live raw, same shape", "#338640"),
        "projected_visibility_padded_p": ("Live projected, same shape", "#7845a6"),
    }
    grouped = defaultdict(list)
    for row in rows:
        if row["run"] == selected_run and row["mode"] in arms:
            grouped[(row["mode"], row["images"])].append(row)
    fig, (traffic, service) = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    for method, (label, color) in arms.items():
        images = sorted(n for mode, n in grouped if mode == method)
        if not images:
            continue
        traffic.plot(images, [median(r["visual_raw_logical_bytes"] + r["visual_projected_logical_bytes"]
                                     for r in grouped[method, n])/1048576 for n in images], "o-", label=label, color=color)
        service.plot(images, [median(r["prefill_plus_d_wall_ms"] for r in grouped[method, n])
                              for n in images], "o-", label=label, color=color)
    for axis in (traffic, service):
        axis.set_xlabel("Images per original prompt")
        axis.grid(alpha=.25)
    traffic.set_ylabel("P→D visual payload (MiB)")
    traffic.set_yscale("log")
    service.set_ylabel("Synchronized prefill + D wall time (ms)")
    fig.legend(*service.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"visual_cross_state_handoff.{ext}", dpi=185)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = summary(args.run, args.out, args.model_receipt)
    figure(records, args.out)
    print("IDEFICS_SUMMARY", len(records))


if __name__ == "__main__":
    main()
