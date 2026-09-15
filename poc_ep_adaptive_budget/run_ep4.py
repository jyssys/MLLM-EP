"""Launch a clean, matched LLaDA2 EP4 dataset trajectory.

The physical CUDA mask is fixed. The benchmark's config=42 rewrites --threshold,
so this uses config=0 with the exact equivalent decoding/cache flags.
"""

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME = Path("/home/esjung/external/dinfer-llada2-flash-poc")
MODEL = Path("/home/esjung/models/LLaDA2.0-flash-744c3f8")
PYTHON = Path("/home/esjung/.venvs/llada2-flash-sglang-053/bin/python")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--submitted", type=int, required=True)
    parser.add_argument("--mini", type=int, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--policy", default="none")
    parser.add_argument("--trace-decisions", action="store_true")
    parser.add_argument("--ep-trace-layers", default="")
    parser.add_argument("--shape-trace-layers", default="")
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve(strict=True)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("Only physical GPU mask 4,5,6,7 is permitted")
    if args.submitted < 1 or args.mini < 1 or args.mini > args.submitted:
        raise ValueError("invalid submitted/mini")
    result = ROOT / "results" / args.label
    result.mkdir(parents=True, exist_ok=os.environ.get("EP_BUDGET_PRECREATED_RESULT") == "1")
    env = os.environ.copy()
    env.update(
        SGL_ENABLE_JIT_DEEPGEMM="false",
        SGLANG_ENABLE_JIT_DEEPGEMM="false",
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="1024",
        EP_BUDGET_POLICY=args.policy,
    )
    for name in ("EP_UNMASK_ATLAS_DIR", "LLADA_DENOISE_TRACE", "LLADA_EP_TRACE_DIR", "LLADA_EP_SHAPE_TRACE_DIR"):
        env.pop(name, None)
    if args.ep_trace_layers:
        env["LLADA_EP_TRACE_DIR"] = str(result / "ep_trace")
        env["LLADA_EP_TRACE_LAYERS"] = args.ep_trace_layers
        env["LLADA_DENOISE_TRACE"] = "1"
    if args.shape_trace_layers:
        env["LLADA_EP_SHAPE_TRACE_DIR"] = str(result / "shape_trace")
        env["LLADA_EP_SHAPE_TRACE_LAYERS"] = args.shape_trace_layers
        env["LLADA_DENOISE_TRACE"] = "1"
    if args.trace_decisions or args.policy != "none":
        env["PYTHONPATH"] = str(ROOT / "instrumentation") + os.pathsep + str(RUNTIME / "python") + os.pathsep + env.get("PYTHONPATH", "")
        env["EP_BUDGET_DECISIONS"] = str(result / "decisions.jsonl") if args.trace_decisions else ""
    command = [
        str(PYTHON), str(RUNTIME / "benchmarks/benchmark_dataset_sglang.py"),
        "--model_name", str(MODEL), "--dataset", str(dataset),
        "--gpu", "0,1,2,3", "--batch_size", str(args.submitted),
        "--gen_len", str(args.generation), "--block_length", "32",
        "--threshold", str(args.threshold), "--config", "0",
        "--parallel_decoding", "threshold", "--cache", "prefix", "--use_bd",
        "--model_type", "flash", "--mini_batch_size", str(args.mini),
        "--ep_size", "4", "--moe_a2a_backend", "deepep",
        "--deepep_mode", "normal", "--output_dir", str(result / "benchmark"),
        "--exp_name", args.label,
    ]
    (result / "command.txt").write_text(" ".join(command) + "\n")
    with (result / "benchmark.log").open("w") as output:
        process = subprocess.Popen(command, cwd=RUNTIME, env=env, stdout=output, stderr=subprocess.STDOUT)
        code = process.wait()
    print(f"{args.label}: exit={code} result={result}")
    if code:
        raise SystemExit(code)
    answers = list((result / "benchmark").glob("*.jsonl"))
    if len(answers) != 1 or not answers[0].is_file():
        raise RuntimeError("benchmark workers failed without propagating exit; no answer JSONL")


if __name__ == "__main__":
    main()
