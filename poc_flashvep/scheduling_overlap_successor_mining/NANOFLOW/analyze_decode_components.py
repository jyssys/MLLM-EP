"""Expose first-decode cost instead of calling capture an online regression."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.screen / "manifest.json").read_text())
    assert manifest["status"] == "SCREEN_COMPLETE_CORRECTNESS_PENDING"
    assert manifest["phase"] == "decode" and not manifest["nsight_profiled"]
    with (args.screen / "paired_results.csv").open() as stream:
        labels = {(int(r["block"]), r["workload"], r["variant"]): r["status"]
                  for r in csv.DictReader(stream)}
    rows = []
    for record in manifest["records"]:
        assert record["returncode"] == 0
        data = json.loads((args.screen / (record["name"] + ".json")).read_text())
        requests = data["requests"]
        for request in requests:
            assert math.isclose(request["ttft_s"] + sum(request["itl_s"]),
                                request["e2e_s"], rel_tol=1e-7, abs_tol=1e-6)
        mean = lambda values: statistics.mean(values)
        row = {"block": record["block"], "workload": record["workload"], "variant": record["variant"],
               "requests": len(requests), "output_tokens": len(requests[0]["output_ids"]),
               "mean_e2e_s": mean(r["e2e_s"] for r in requests),
               "mean_ttft_s": mean(r["ttft_s"] for r in requests),
               "first_decode_itl_s": mean(r["itl_s"][0] for r in requests),
               "steady_itl_mean_s": mean(x for r in requests for x in r["itl_s"][1:]),
               "correctness": labels.get((record["block"], record["workload"], record["variant"]),
                                         "BASELINE_SANITY_ONLY")}
        row["first_decode_fraction_of_e2e_pct"] = 100 * row["first_decode_itl_s"] / row["mean_e2e_s"]
        rows.append(row)
    aggregate = []
    for workload, variant in sorted({(r["workload"], r["variant"]) for r in rows}):
        selected = [r for r in rows if (r["workload"], r["variant"]) == (workload, variant)]
        aggregate.append({"workload": workload, "variant": variant, "restarts": len(selected),
                          "token_exact_pairs": sum(r["correctness"] == "TOKEN_EXACT_DESCRIPTIVE_ONLY" for r in selected),
                          **{name: statistics.median(r[name] for r in selected) for name in (
                              "mean_e2e_s", "first_decode_itl_s", "steady_itl_mean_s",
                              "first_decode_fraction_of_e2e_pct")}})
    for name, values in (("decode_components.csv", rows), ("decode_component_aggregate.csv", aggregate)):
        with (args.screen / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    result = {"scope": "raw descriptive decomposition including correctness-excluded pairs",
              "no_latency_removed": True,
              "warning": "first decode includes setup/capture and required work; not all removable; steady ITL is not E2E",
              "aggregate": aggregate}
    (args.screen / "decode_component_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
