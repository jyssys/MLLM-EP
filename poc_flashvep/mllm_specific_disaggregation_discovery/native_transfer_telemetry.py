"""Read NIXL and KV observations without reinterpreting descriptor bytes as counters."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo


TIMESTAMP = re.compile(r"\b(\d\d-\d\d \d\d:\d\d:\d\d) \[")
KV_USAGE = re.compile(r"GPU KV cache usage: (\d+\.?\d*)%")
METRICS = ("bytes_transferred", "xfer_time_seconds", "post_time_seconds", "num_descriptors")


def read(path):
    return json.loads(path.read_text())


def logs(path, *, year, zone, cohorts):
    recent = []
    with path.open() as stream:
        for line in stream:
            nch = "KV Transfer metrics:" in line
            kv = KV_USAGE.search(line)
            if not nch and kv is None:
                continue
            match = TIMESTAMP.search(line)
            if not match:
                continue
            time = datetime.strptime(f"{year}-{match[1]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).timestamp()
            scope = [label for label, start, end in cohorts if time >= start and time - 10.0 <= end]
            row = {"wall_epoch": time, "possibly_overlapping_cohorts": scope}
            if nch:
                row.update({key: float(value) for key, value in
                            (entry.strip().rsplit("=", 1) for entry in
                             line.split("KV Transfer metrics:", 1)[1].strip().split(", "))})
            if kv is not None:
                row["KV_cache_utilization_percent_reported"] = float(kv[1])
            recent.append(row)
    return recent


def prom_values(lines):
    result = {}
    if isinstance(lines, dict):
        return result
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name = line.split(maxsplit=1)[0].split("{", 1)[0]
        value = line.rsplit(maxsplit=1)[-1]
        if any(name == f"vllm:nixl_{field}_{unit}" for field in METRICS for unit in ("count", "sum")):
            result[name] = result.get(name, 0.0) + float(value)
    return result


def prom_delta(receipt):
    result = {}
    for role, after in receipt["after"].items():
        previous = prom_values(receipt["before"].get(role, {}))
        subsequent = prom_values(after)
        result[role] = {key: value - previous.get(key, 0.0) for key, value in subsequent.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zone", default="Asia/Seoul")
    args = parser.parse_args()
    cohorts = {}
    for path in sorted((args.run / "state").glob("*_complete.json")):
        state = read(path)
        if "wall_start" in state:
            cohorts[state["run"]] = state
    zone = ZoneInfo(args.zone)
    year = datetime.fromtimestamp(min(item["wall_start"] for item in cohorts.values()), zone).year
    boundaries = [(label, item["wall_start"], item["wall_end"]) for label, item in cohorts.items()]
    observed = logs(args.run / "logs/discovery_native_decode.log", year=year, zone=zone, cohorts=boundaries)
    telemetry = {}
    for path in sorted((args.run / "raw").glob("*_nixl_prometheus.json")):
        label = path.name.removesuffix("_nixl_prometheus.json")
        delta = prom_delta(read(path))
        d = delta.get("decode", {})
        n = d.get("vllm:nixl_bytes_transferred_count")
        expected = cohorts.get(label, {}).get("completed")
        telemetry[label] = {"raw_deltas": delta, "decode_transfers": n, "client_completed": expected,
                           "one_transfer_per_completed_request_count_matches": n is not None and n == expected}
    report = {"prometheus": telemetry, "decode_log": observed,
              "scope": "NIXL telemetry counts connector transport bytes and duration. Ten-second approximate log "
                       "windows may cross cohort boundaries. KV utilization uses the engine's sampled cache pool."}
    with args.out.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
