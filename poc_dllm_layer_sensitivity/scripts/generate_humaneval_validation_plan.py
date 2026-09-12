#!/usr/bin/env python3
"""Build the bounded cross-task validation plan after GSM8K screening."""

import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
baseline = json.loads((root / "plans/baseline.json").read_text())[0]
single = json.loads((root / "plans/layer_bypass_all.json").read_text())
groups = json.loads((root / "plans/group_quarters.json").read_text())
representative_layers = {0, 1, 4, 8, 12, 16, 20, 24, 28, 31}

policies = [baseline]
policies.extend(
    policy
    for policy in single
    if policy["name"] != "baseline"
    and int(policy["layers"]) in representative_layers
)
policies.extend(policy for policy in groups if policy["name"] != "baseline")

destination = root / "plans/humaneval_validation.json"
destination.write_text(json.dumps(policies, indent=2) + "\n")
print(json.dumps({"output": str(destination), "policies": len(policies)}))
