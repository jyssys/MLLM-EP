"""Resume TRUE-EP4 generation one external-DP wave per engine lifetime.

The installed vLLM 0.20 external-DP runtime can stall when the same frontend
starts another wave.  Per-DP JSONL checkpoints are atomic, so a bounded
supervisor can restart the engine after every successful global wave without
changing request semantics or losing completed samples.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from poc_mllm_ep_foundation.io import append_jsonl_atomic


def _ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["request_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _done_ids(output: Path) -> set[str]:
    done = _ids(output / "results.jsonl") | _ids(output / "failures.jsonl")
    done |= _ids(output / "results.dp0.jsonl") | _ids(output / "results.dp1.jsonl")
    return done


def _templates(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        (
            str(row.get("base_request_id") or str(row["request_id"]).split("__", 1)[0]),
            str(row["resolution_label"]),
        ): row
        for row in rows
    }


def _timeout_record(row: dict, template: dict, args: argparse.Namespace) -> dict:
    record = dict(template)
    record.update({key: value for key, value in row.items() if key != "answer"})
    record.update({
        "request_id": str(row["request_id"]),
        "base_request_id": str(row.get("base_request_id") or str(row["request_id"]).split("__", 1)[0]),
        "answer": row.get("answer"),
        "k_visual": args.k_visual,
        "k_text": 8,
        "k_decode": 8,
        "output_text": "",
        "output_token_ids": [],
        "output_token_count": None,
        "finish_reason": "generation_timeout",
        "stop_reason": f"owned_launch_wall_timeout_{args.launch_timeout_seconds:g}s",
        "ttft_ms": None,
        "request_engine_ms": 1000.0 * args.launch_timeout_seconds,
        "request_engine_ms_is_lower_bound": True,
        "queue_ms": None,
        "batch_makespan_ms": 1000.0 * args.launch_timeout_seconds,
        "backend": "physical_ep4",
        "lane": "official",
        "actual_executed_k_contract": "VISUAL_RAGGED_INVALID_ID_MINUS_ONE_TEXT_AND_DECODE_K8",
        "evidence": "MEASURED_TRUE_EP4_GENERATION_TIMEOUT",
        "failure": True,
        "timeout_seconds": args.launch_timeout_seconds,
    })
    return record


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k-visual", type=int, choices=(8, 6, 4), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--waves-per-launch", type=int, choices=(1, 2, 4, 8), default=1,
        help=(
            "bounded engine lifetime; values above two require a successful "
            "same-batch-size smoke and remain protected by the launch timeout"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=40000)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--kv-cache-gib", type=float, default=4.0)
    parser.add_argument("--lane", choices=("official", "deterministic"), default="official")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--launch-timeout-seconds", type=float, default=5400.0)
    parser.add_argument("--wave-offset", type=int, default=10_000)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument(
        "--record-timeout-failures", action="store_true",
        help="record the actively stalled DP wave as abnormal termination after a bounded timeout",
    )
    parser.add_argument(
        "--failure-template-results", type=Path,
        help="same-input K8 JSONL used only to copy processor/grid metadata for timeout records",
    )
    args = parser.parse_args()

    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))["requests"]
    expected = {str(row["request_id"]) for row in manifest_rows}
    templates = _templates(args.failure_template_results)
    if args.record_timeout_failures and not templates:
        raise ValueError("--record-timeout-failures requires --failure-template-results")
    args.output.mkdir(parents=True, exist_ok=True)
    attempt = 0
    failures = 0
    while True:
        done = _done_ids(args.output)
        missing = expected - done
        print(
            f"[single-wave supervisor] completed={len(done)}/{len(expected)} "
            f"missing={len(missing)} attempt={attempt}",
            flush=True,
        )
        if not missing:
            break
        command = [
            sys.executable,
            "-m", "poc_mllm_resolution_expert_tradeoff.scripts.run_factorial_ep4",
            "--physical-gpus", "4,5,6,7",
            "--manifest", str(args.manifest),
            "--output", str(args.output),
            "--k-visual", str(args.k_visual),
            "--batch-size", str(args.batch_size),
            "--max-tokens", str(args.max_tokens),
            "--max-model-len", str(args.max_model_len),
            "--max-num-seqs", str(args.max_num_seqs),
            "--kv-cache-gib", str(args.kv_cache_gib),
            "--lane", args.lane,
            "--limit", str(2 * args.batch_size * args.waves_per_launch),
            "--resume",
            "--wave-offset", str(args.wave_offset + attempt),
        ]
        if args.ignore_eos:
            command.append("--ignore-eos")
        process = subprocess.Popen(command, start_new_session=True)
        try:
            return_code = process.wait(timeout=args.launch_timeout_seconds)
        except subprocess.TimeoutExpired:
            print("[single-wave supervisor] timed out; terminating owned process group", flush=True)
            _terminate(process)
            return_code = 124
        after = _done_ids(args.output)
        if return_code == 124 and args.record_timeout_failures:
            launch_limit = 2 * args.batch_size * args.waves_per_launch
            attempted = [
                row for row in manifest_rows if str(row["request_id"]) not in done
            ][:launch_limit]
            completed_in_launch = len(after - done)
            stalled = attempted[
                completed_in_launch: completed_in_launch + 2 * args.batch_size
            ]
            failures_to_write = []
            for row in stalled:
                key = (str(row["base_request_id"]), str(row["resolution_label"]))
                if key not in templates:
                    raise RuntimeError(f"missing timeout metadata template for {key}")
                failures_to_write.append(_timeout_record(row, templates[key], args))
            if failures_to_write:
                append_jsonl_atomic(args.output / "failures.jsonl", failures_to_write)
                after = _done_ids(args.output)
                print(
                    "[single-wave supervisor] recorded bounded abnormal termination: "
                    + ",".join(str(row["request_id"]) for row in stalled),
                    flush=True,
                )
        progressed = len(after) > len(done)
        if return_code == 0 and progressed:
            failures = 0
        else:
            failures += 1
            print(
                f"[single-wave supervisor] rc={return_code} progressed={progressed} "
                f"consecutive_failures={failures}",
                flush=True,
            )
        if failures >= args.max_consecutive_failures:
            raise RuntimeError("single-wave engine failed repeatedly without progress")
        attempt += 1
        time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
