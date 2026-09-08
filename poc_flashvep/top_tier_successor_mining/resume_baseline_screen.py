"""One fixed research handoff: completed calibration -> bridge captures -> sanity.

This is experiment orchestration, not a serving scheduler. Every GPU child has
an explicit baseline question; no burn, synthetic utilization, or hidden retry.
The active agent must inspect parity before adding performance follow-ups.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from gpu_scope import allowed_devices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--frontier-pid", type=int, required=True)
    args = ap.parse_args()
    allowed_devices()
    root = args.results.resolve()
    task = Path(__file__).resolve().parent
    policy_path = task / "GPU_EXECUTION_POLICY.json"
    if policy_path.exists():
        assert json.loads(policy_path.read_text())["gpu_runs_allowed"], "GPU use paused by user; do not auto-resume"
    quality_python = "/home/esjung/.venvs/top-tier-successor-quality/bin/python"
    libra_python = "/home/esjung/.venvs/libra-supplement-py310/bin/python"
    vl = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
    llm = "/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "OMP_NUM_THREADS": "4"}
    log = root / "raw/resume_baseline_screen.jsonl"

    def record(**values):
        row = {"utc": datetime.now(timezone.utc).isoformat(), **values}
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    # Check only the known, own calibration launcher; never discover/kill peers.
    pid_path = Path(f"/proc/{args.frontier_pid}/cmdline")
    if pid_path.exists():
        assert b"modes_frontier.py" in pid_path.read_bytes()
    record(stage="WAIT_FOR_OWN_CALIBRATION", pid=args.frontier_pid,
           gpu_experiment_time=False)
    deadline = time.monotonic() + 7200
    while pid_path.exists():
        if policy_path.exists():
            assert json.loads(policy_path.read_text())["gpu_runs_allowed"], "GPU use paused by user"
        if time.monotonic() >= deadline:
            raise TimeoutError("Own calibration still running; no child launched")
        time.sleep(5)
    completion = root / "quality/modes_frontier1024_grid100/completed.json"
    assert completion.exists(), "Calibration exited without completion; inspect its log"
    frontier = json.loads((completion.parent / "frontier.json").read_text())
    assert len(frontier) == 2 and all(r["tau_text"] is not None for r in frontier)
    policies = [{"name": "vanilla", "method": "vanilla"}]
    for row in frontier:
        policies.append({"name": f"modes_official_target{round(row['target_skip']*100)}",
                         "method": "modes", "alpha": str(root / "analysis/modes_alpha1024_official.json"),
                         "tau_text": row["tau_text"], "tau_vision": row["tau_vision"],
                         "calibration_actual_skip": row["actual_skip"],
                         "calibration_kl": row["kl"], "fast_skip": True})
    (root / "analysis/modes_confirmatory_policies.json").write_text(json.dumps(policies, indent=2))
    record(stage="CALIBRATION_COMPLETE", frontier=frontier)

    def group(label, commands, run_env, timeout):
        if policy_path.exists():
            assert json.loads(policy_path.read_text())["gpu_runs_allowed"], "GPU use paused by user"
        children, handles = [], []
        record(stage="START", experiment=label, commands=commands)
        start = time.monotonic()
        try:
            for index, command in enumerate(commands):
                handle = (root / f"raw/{label}_{index}.log").open("w")
                handles.append(handle)
                children.append(subprocess.Popen(command, env=run_env, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True))
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None, 0) for p in children):
                    raise RuntimeError(f"{label}: child failed; see raw log")
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"{label}: bounded diagnostic timeout")
                time.sleep(2)
            assert all(p.returncode == 0 for p in children)
            record(stage="COMPLETE", experiment=label, elapsed_seconds=time.monotonic()-start)
        finally:
            # Only process groups explicitly launched in this function are ours.
            for p in children:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for p in children:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            for handle in handles:
                handle.close()

    captures = root / "quality/libra_vl_bridge_inputs"
    assert not list(captures.glob("*.pt")), "Capture already exists; do not overwrite"
    commands = [[quality_python, str(task / "capture_libra_vl_inputs.py"),
                 "--model", vl, "--data", str(root / "data/requests.jsonl"),
                 "--out", str(captures), "--gpu", str(rank), "--shard", str(rank),
                 "--shards", "4"] for rank in range(4)]
    group("libra_vl_exact_capture", commands, env, 1800)
    assert len(list(captures.glob("*_source*.pt"))) == 32

    libra_env = {**env, "PATH": str(Path(libra_python).parent)+os.pathsep+env["PATH"],
                 "CUDA_HOME": "/usr/local/cuda-12.8"}
    common = [libra_python, str(task / "libra_native_sanity.py"),
              "--data", str(root / "data/fineweb.jsonl"), "--warmup", "3", "--reps", "5"]
    text_out = root / "online/libra_native_text48_m128"
    assert not text_out.exists(), "Native run already exists; do not overwrite"
    group("libra_native_text48_m128", [common + ["--model", llm,
          "--out", str(text_out), "--tokens", "128", "--cases", "3"]], libra_env, 1800)
    vl_out = root / "online/libra_native_vl48_gqa0"
    assert not vl_out.exists(), "Native VL run already exists; do not overwrite"
    group("libra_native_vl48_gqa0", [common + ["--model", str(root / "references/qwen_vl_text_descriptor"),
          "--out", str(vl_out), "--cases", "1", "--vl-inputs", str(captures),
          "--vl-group", "gqa_0"]], libra_env, 1800)
    record(stage="AWAIT_AGENT_PARITY_REVIEW", note="No successor or winner selected")


if __name__ == "__main__":
    main()
