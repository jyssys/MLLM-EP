"""Read native scheduler prints; no hook, no inferred request association."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("screen", type=Path)
    args = parser.parse_args()
    pattern = re.compile(r"num_seqs: (\d+), len\(prefill_scheduled_seqs\): (\d+), "
                         r"num_batched_tokens: (\d+), num_attn_tokens: (\d+), num_stages: (\d+)")
    runs = []
    for path in sorted(args.screen.glob("b*/server.log")):
        meta = json.loads((path.parent/"run.json").read_text())
        matches = [tuple(map(int, x)) for x in pattern.findall(path.read_text())]
        runs.append({"run": path.parent.name, "configured_stage_cap": meta["stages"],
                     "mode": meta["mode"], "native_stage_choices": dict(Counter(x[4] for x in matches)),
                     "admissions": [{"active_seqs": x[0], "prefill_seqs": x[1],
                                     "batched_tokens": x[2], "attention_work_proxy": x[3],
                                     "actual_stages": x[4]} for x in matches]})
    result = {"scope": "native stdout includes warmup; no exact request/time join claimed",
              "source_semantics": "configured cap selects a native token-dependent group-count table",
              "runs": runs}
    (args.screen/"native_scheduler_choices.json").write_text(json.dumps(result, indent=2))
    print(json.dumps([{k: v for k, v in run.items() if k != "admissions"} for run in runs], indent=2))


if __name__ == "__main__":
    main()
