"""Run an isolated GPU command with the verified local burner lifecycle."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from ..mllm_epd_h200_prefill_heavy_batch_matched import runtime as rt
from ..mllm_epd_h200_prefill_heavy_batch_matched.resources import burner_status, experiment_lease
from ..mllm_epd_h200_wide_decode_batch_token_split.host import snapshot


def wait_owned(process, wall_seconds):
    try:
        return process.wait(timeout=wall_seconds)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Comma-separated physical NVIDIA GPU indices")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    devices = list(map(int, args.devices.split(",")))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or len(set(devices)) != len(devices) or args.wall_seconds <= 0:
        parser.error("Provide distinct devices, a positive timeout and one owned GPU command")
    available = {int(gpu): (uuid, model) for gpu, uuid, model in rt.nvidia("--query-gpu=index,uuid,name")}
    rt.require(all(device in available and available[device][1] == "NVIDIA H200" for device in devices),
               "Unexpected physical GPU")
    selected = {device: available[device][0] for device in devices}
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(selected.values()),
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    run = args.run_dir.resolve()
    run.mkdir(mode=0o700)
    for directory in ("state", "logs"):
        (run / directory).mkdir()
    before = burner_status()
    rt.write(run / "state/environment.json", {"selected_uuids": selected, "burner_before": before,
                                              "command": command, "host_before": snapshot(topology=True),
                                              "start_unix": time.time()})

    def on_cancel(*_):
        raise InterruptedError("GPU epoch interrupted")

    signal.signal(signal.SIGTERM, on_cancel)
    signal.signal(signal.SIGINT, on_cancel)
    try:
        with experiment_lease(selected, run / "state") as stopped:
            rt.write(run / "state/stopped_burners.json", stopped)
            rt.write(run / "state/host_ready.json", snapshot())
            with (run / "logs/command.log").open("x") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                exit_code = wait_owned(process, args.wall_seconds)
                rt.require(exit_code == 0, f"Owned GPU command exited with status {exit_code}")
    except BaseException as error:
        rt.write(run / "state/failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    rt.write(run / "state/complete.json", {"status": "COMPLETE", "end_unix": time.time()})


if __name__ == "__main__":
    main()
