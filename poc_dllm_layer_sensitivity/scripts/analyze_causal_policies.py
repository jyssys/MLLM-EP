#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import torch


def number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{\s*([-+]?[$]?[\d,]+(?:\.\d+)?)\s*\}", text)
    values = boxed or re.findall(r"[-+]?[$]?[\d,]+(?:\.\d+)?", text)
    return values[-1].replace("$", "").replace(",", "") if values else None


def humaneval_pass(prompt: str, answer: str, test: str, entry_point: str) -> bool:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", answer, flags=re.S | re.I)
    continuation = fenced[0] if fenced else answer
    source = (
        continuation
        if fenced and f"def {entry_point}" in continuation
        else prompt + continuation
    )
    source += f"\n\n{test}\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py") as handle:
        handle.write(source)
        handle.flush()
        try:
            result = subprocess.run(
                ["python", "-I", handle.name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def score(
    task: str,
    rows: list[dict],
    truth: list[dict],
    cache: dict[tuple[int, str], bool] | None = None,
) -> tuple[int, list[bool]]:
    by_id = {int(row["original_index"]): row for row in rows}
    passed = []
    for index, item in enumerate(truth):
        answer = by_id[index]["answer"]
        if task == "gsm8k":
            gold = item["answer"].split("####")[-1].strip().replace(",", "")
            passed.append(number(answer) == gold)
        else:
            cache_key = (index, answer)
            if cache is not None and cache_key in cache:
                passed.append(cache[cache_key])
            else:
                result = humaneval_pass(
                    item["prompt"], answer, item["test"], item["entry_point"]
                )
                if cache is not None:
                    cache[cache_key] = result
                passed.append(result)
    return sum(passed), passed


def row_key_map(capture: dict):
    keys = []
    starts = capture["block_starts"].tolist()
    seqs = capture["sequence_ids"].tolist()
    q_len = int(capture["mask_before"].shape[1])
    for sequence_id, block_start in zip(seqs, starts):
        keys.extend((int(sequence_id), int(block_start), position) for position in range(q_len))
    return keys


def capture_metrics(baseline_path: Path, policy_path: Path, compute_logits: bool = True) -> dict:
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    baseline_keys = row_key_map(baseline)
    policy_keys = row_key_map(policy)
    b_index = {key: index for index, key in enumerate(baseline_keys)}
    p_index = {key: index for index, key in enumerate(policy_keys)}
    common = sorted(set(b_index) & set(p_index))
    if not common:
        return {"aligned_rows": 0}

    b_top1 = baseline["top1_tokens"].flatten()
    p_top1 = policy["top1_tokens"].flatten()
    b_live = baseline["mask_before"].flatten()
    b_accept = baseline["accepted_mask"].flatten()
    p_accept = policy["accepted_mask"].flatten()
    b_margin = (baseline["top1_logits"] - baseline["top2_logits"]).flatten().float()
    p_margin = (policy["top1_logits"] - policy["top2_logits"]).flatten().float()
    flips = []
    live_flips = []
    accept_diffs = []
    margin_changes = []
    for key in common:
        bi, pi = b_index[key], p_index[key]
        flips.append(bool(b_top1[bi] != p_top1[pi]))
        if bool(b_live[bi]):
            live_flips.append(bool(b_top1[bi] != p_top1[pi]))
            accept_diffs.append(bool(b_accept[bi] != p_accept[pi]))
            margin_changes.append(abs(float(b_margin[bi] - p_margin[pi])))

    baseline_sample = {
        baseline_keys[int(flat_index)]: row
        for flat_index, row in zip(
            baseline["sample_flat_indices"].tolist(), baseline["sampled_logits"].float()
        )
    }
    policy_sample = {
        policy_keys[int(flat_index)]: row
        for flat_index, row in zip(
            policy["sample_flat_indices"].tolist(), policy["sampled_logits"].float()
        )
    }
    sample_common = sorted(set(baseline_sample) & set(policy_sample))
    kls, jss = [], []
    if sample_common and compute_logits:
        baseline_logits = torch.stack([baseline_sample[key] for key in sample_common])
        policy_logits = torch.stack([policy_sample[key] for key in sample_common])
        b_logp = torch.log_softmax(baseline_logits, dim=-1)
        p_logp = torch.log_softmax(policy_logits, dim=-1)
        b_prob, p_prob = b_logp.exp(), p_logp.exp()
        mixture = 0.5 * (b_prob + p_prob)
        log_mixture = mixture.clamp_min(1e-30).log()
        kls = (b_prob * (b_logp - p_logp)).sum(dim=-1).tolist()
        jss = (
            0.5 * (b_prob * (b_logp - log_mixture)).sum(dim=-1)
            + 0.5 * (p_prob * (p_logp - log_mixture)).sum(dim=-1)
        ).tolist()
    return {
        "aligned_rows": len(common),
        "top1_flip_fraction": sum(flips) / len(flips),
        "live_aligned_rows": len(live_flips),
        "live_top1_flip_fraction": sum(live_flips) / len(live_flips)
        if live_flips
        else math.nan,
        "accepted_set_change_fraction": sum(accept_diffs) / len(accept_diffs)
        if accept_diffs
        else math.nan,
        "confidence_logit_margin_abs_change_mean": sum(margin_changes) / len(margin_changes)
        if margin_changes
        else math.nan,
        "sampled_logits_rows": len(sample_common),
        "sampled_kl_mean": sum(kls) / len(kls) if kls else math.nan,
        "sampled_js_mean": sum(jss) / len(jss) if jss else math.nan,
        "baseline_wave": int(baseline["wave"]),
        "policy_wave": int(policy["wave"]),
        "baseline_live_ratio": float(baseline["live_ratio"]),
        "policy_live_ratio": float(policy["live_ratio"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gsm8k", "humaneval"], required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-logit-divergence", action="store_true")
    parser.add_argument(
        "--policy-regex",
        default="",
        help="Analyze baseline plus only policy names matching this regex.",
    )
    args = parser.parse_args()

    truth = json.loads(args.truth.read_text())
    documents = {}
    for path in sorted(args.policy_dir.glob("policy_*.json")):
        document = json.loads(path.read_text())
        policy_name = document["policy"]["name"]
        if (
            policy_name != "baseline"
            and args.policy_regex
            and re.search(args.policy_regex, policy_name) is None
        ):
            continue
        documents[policy_name] = document
    baseline = documents["baseline"]
    baseline_tokens = {
        int(row["original_index"]): row["token_ids"] for row in baseline["rows"]
    }
    quality_cache: dict[tuple[int, str], bool] = {}
    baseline_correct, baseline_pass = score(
        args.task, baseline["rows"], truth, quality_cache
    )

    rows = []
    for policy_name, document in sorted(documents.items()):
        correct, passed = score(args.task, document["rows"], truth, quality_cache)
        exact_sequences = sum(
            row["token_ids"] == baseline_tokens[int(row["original_index"])]
            for row in document["rows"]
        )
        phase = document["policy"].get("phases", "early,middle,late")
        capture = {}
        policy_capture = args.capture_dir / f"{policy_name}_{phase}.pt"
        baseline_capture = args.capture_dir / f"baseline_{phase}.pt"
        if policy_name != "baseline" and policy_capture.exists() and baseline_capture.exists():
            capture = capture_metrics(
                baseline_capture,
                policy_capture,
                compute_logits=not args.skip_logit_divergence,
            )
        rows.append(
            {
                "task": args.task,
                "policy": policy_name,
                "mode": document["policy"].get("mode", "none"),
                "layers": document["policy"].get("layers", ""),
                "phase": phase,
                "nfe": int(document["nfe"]),
                "elapsed_seconds_observer": float(document["elapsed_seconds"]),
                "final_sequence_exact": exact_sequences,
                "final_sequence_total": len(document["rows"]),
                "benchmark_correct": correct,
                "benchmark_total": len(truth),
                "benchmark_delta_vs_baseline": correct - baseline_correct,
                "per_sample_score_flips": sum(a != b for a, b in zip(passed, baseline_pass)),
                **capture,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"policies": len(rows), "baseline_correct": baseline_correct}))


if __name__ == "__main__":
    main()
