"""Record the local experiment environment and GPU lease outcomes without process payloads."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    leases = []
    environment = None
    for path in sorted(args.runs.glob("*gpu_lease")):
        state = path / "state"
        receipt = state / "environment.json"
        if not receipt.is_file():
            continue
        data = read(receipt)
        host = data["host_before"]
        if environment is None:
            environment = {
                "gpu_inventory": host["gpu"]["stdout"], "topology": host["topology"]["stdout"],
                "cpu_numa": host["cpu"]["stdout"], "memory": host["memory"].splitlines()[:6],
                "driver_version_line": next((line for line in host["driver"]["stdout"].splitlines()
                                             if "Driver Version:" in line), None),
            }
        status = (read(state / "failure.json") if (state / "failure.json").is_file() else
                  read(state / "complete.json") if (state / "complete.json").is_file() else
                  {"status": "NO_TERMINAL_RECEIPT"})
        leases.append({"run": path.name, "result": status, "devices": data["selected_uuids"],
                       "command": data["command"], "start_unix": data["start_unix"],
                       "outside_burners": len(data["burner_before"]) - len(data["selected_uuids"]),
                       "ready_gpu_snapshot": read(state / "host_ready.json")["gpu"]["stdout"]
                                             if (state / "host_ready.json").is_file() else None})
    raw = {
        "base_sha": args.base, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "vllm", "safetensors", "numpy", "triton")},
        "environment": environment, "leases": leases,
        "scope": "GPU UUID, driver, topology and per-run readiness. GPU process command payloads omitted.",
    }
    with args.out.open("x") as stream:
        json.dump(raw, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
