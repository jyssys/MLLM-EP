#!/usr/bin/env python3
import json
from pathlib import Path


def entry(name, mode="none", layers="", phase="early,middle,late"):
    return {
        "name": name,
        "mode": mode,
        "layers": layers,
        "phases": phase,
        "waves": "",
        "capture_phases": phase,
    }


groups = {
    "shallow": "0,1,2,3,4,5,6,7",
    "middle_low": "8,9,10,11,12,13,14,15",
    "middle_high": "16,17,18,19,20,21,22,23",
    "deep": "24,25,26,27,28,29,30,31",
}
policies = [entry("baseline")]
for mode in ("layer_bypass", "routed_bypass", "stale_routed", "routed_only"):
    for phase in ("early", "middle", "late"):
        for group_name, layers in groups.items():
            policies.append(
                entry(
                    f"{mode}_{group_name}_{phase}",
                    mode,
                    layers,
                    phase,
                )
            )

output = Path(__file__).resolve().parents[1] / "plans/group_quarters.json"
output.write_text(json.dumps(policies, indent=2) + "\n")
print(output, len(policies))
