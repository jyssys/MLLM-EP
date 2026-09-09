"""CPU-only consistency checks for the bounded study handoff."""
import csv
import hashlib
import json
import os
from pathlib import Path


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    study = Path(__file__).resolve().parents[1]
    repo = study.parents[1]
    with (study / "SUCCESSOR_SCOREBOARD.csv").open() as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames[2:12]
        scores = list(reader)
    assert len(scores) == 3 and len(columns) == 10
    assert sum(r["best_successor_candidate"] == "YES" for r in scores) == 1
    for row in scores:
        assert sum(int(row[c]) for c in columns) == int(row["total_of_50"])
        assert all(0 <= int(row[c]) <= 5 for c in columns)
    bundle = json.loads((study / "COMMON/EVIDENCE_BUNDLE.json").read_text())
    for row in bundle["files"]:
        path = repo / row["path"]
        assert path.stat().st_size == row["bytes"], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    for path in study.glob("*.csv"):
        with path.open() as stream:
            reader = csv.reader(stream)
            size = len(next(reader))
            assert all(len(row) == size for row in reader), path
    with (study / "GPU_TIME_LOG.csv").open() as stream:
        timing = list(csv.DictReader(stream))
    assert len({r["run_id"] for r in timing}) == len(timing)
    assert all(set(r["physical_gpus"].split(",")) <= {"4", "5", "6", "7"} for r in timing)
    assert not any("burn" in r["kind"].lower() for r in timing)
    with (study / "REQUEST_E2E_RESULTS.csv").open() as stream:
        requests = sum(1 for _ in csv.DictReader(stream))
    accounting = json.loads((Path(bundle["results_root"]) / "cpu_analysis/interim_accounting.json").read_text())
    assert requests == accounting["request_rows"]
    print(json.dumps({"status": "PASS", "score_rows": len(scores), "evidence_files": len(bundle["files"]),
                      "request_rows": requests, "unique_time_intervals": len(timing)}))


if __name__ == "__main__":
    main()
