"""Randomized official-mechanism screening; no runtime policy changes."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--p2p-disable", choices=["0", "1"], default="1")
    p.add_argument("--model")
    p.add_argument("--correctness-trace", type=Path)
    p.add_argument("--workload-warmup-repeats", type=int, default=0)
    p.add_argument("--max-total-tokens", type=int)
    p.add_argument("--workloads", nargs="+", choices=["heterogeneous_steady", "heterogeneous_bursty",
                                                     "short_high_concurrency", "fastpp_official_azure_conv_192"])
    p.add_argument("--variants", nargs="+", choices=["pp_only", "pp_chunk128", "pp_chunk512", "greedy", "alp", "alp_rebalance"],
                   default=["pp_only", "greedy", "alp", "alp_rebalance"])
    args = p.parse_args()
    destination = args.results / "fastpp_runs" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    traces = [args.results / "request_traces/text_v1" / f"{x}.jsonl"
              for x in ("heterogeneous_steady", "heterogeneous_bursty", "short_high_concurrency")]
    traces.append(args.results / "request_traces/fastpp_official_azure_conv_192.jsonl")
    if args.workloads:
        traces = [trace for trace in traces if trace.stem in args.workloads]
    plan = []
    for block in range(args.restarts):
        variants = list(args.variants)
        rng.shuffle(variants)
        # Within a restart block each configuration sees identical trace order.
        ordered_traces = list(traces)
        rng.shuffle(ordered_traces)
        for variant in variants:
            plan.append({"block": block, "variant": variant,
                         "traces": [str(x) for x in ordered_traces]})
    manifest = {"seed": args.seed, "plan": plan, "status": "RUNNING",
                "p2p_disable": args.p2p_disable, "model": args.model, "completed": [],
                "workload_warmup_repeats": args.workload_warmup_repeats,
                "fixed_max_total_tokens": args.max_total_tokens}
    manifest_path = destination / "plan.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        for item in plan:
            run_id = f"b{item['block']}_{item['variant']}"
            out = destination / run_id
            command = [sys.executable, str(ROOT / "FASTPP/run_configuration.py"),
                       "--variant", item["variant"], "--out", str(out),
                       "--warmup", str(args.results / "request_traces/text_v1/warmup.jsonl"),
                       "--p2p-disable", args.p2p_disable,
                       "--workload-warmup-repeats", str(args.workload_warmup_repeats)]
            if args.model:
                command += ["--model", args.model]
            if args.max_total_tokens:
                command += ["--max-total-tokens", str(args.max_total_tokens)]
            if args.correctness_trace:
                command += ["--correctness-trace", str(args.correctness_trace)]
            for trace in item["traces"]:
                command += ["--trace", trace]
            print("START", run_id, flush=True)
            with (destination / (run_id + ".log")).open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=2400)
            meta = json.loads((out / "run.json").read_text())
            for measured in meta["request_runs"]:
                if not measured["label"].startswith("measure"):
                    continue
                subprocess.run([
                    sys.executable, str(ROOT / "COMMON/record_gpu_interval.py"),
                    "--log", str(ROOT / "GPU_TIME_LOG.csv"),
                    "--run-id", args.name + ":" + measured["run_id"],
                    "--candidate", "FASTPP", "--question",
                    "Official PP/DC/ALP/BR independent-restart request-level comparison",
                    "--start", str(measured["start_unix"]),
                    "--end", str(measured["start_unix"] + measured["elapsed_s"]),
                    "--kind", "measurement", "--valid", "CROSS_CONFIG_CORRECTNESS_PENDING",
                    "--artifact", str(out / (measured["label"] + ".jsonl"))], check=True)
            manifest["completed"].append(run_id)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print("DONE", run_id, flush=True)
        manifest["status"] = "COLLECTED_CROSS_CONFIG_CORRECTNESS_PENDING"
    except BaseException as exc:
        manifest.update(status="FAILED", error=repr(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
