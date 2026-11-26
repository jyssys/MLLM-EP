"""Audit online cohorts without pooling numerical regimes, transport, or shadow."""

import argparse
from functools import cached_property
import json
from pathlib import Path

from ..mllm_epd_h200_modality_aware_dynamic_split.online_evidence import DynamicCohort, aggregate


class RecoveryCohort(DynamicCohort):
    @cached_property
    def variant(self):
        flags = self.environment.get("argv", [])
        cap = flags[flags.index("--idle-budget-ms") + 1] if "--idle-budget-ms" in flags else "160"
        gain = flags[flags.index("--gain-estimate") + 1] if "--gain-estimate" in flags else "lower"
        return {
            **super().variant,
            "numeric_regime": self.environment.get("numeric_regime", "bi0"),
            "embedding_mode": "slice" if "--slice-aware-transfer" in flags else "whole",
            "stock_cut_observer": "--stock-cut-trace" in flags,
            "budget_policy": "adaptive" if "--adaptive-budget" in flags else f"{float(cap):g}ms",
            "gain_estimate": gain,
            "preissue_crossing": "--preissue-crossing-ec" in flags,
            "inline_transport_required": "--require-ready-transport" in flags,
        }

    @property
    def family(self):
        base = super().family
        if self.variant["embedding_mode"] == "slice":
            base += "-slice"
        if self.variant["numeric_regime"] != "bi0":
            base += "-" + self.variant["numeric_regime"]
        if base.startswith(("b-new", "c-new")):
            base += f"-{self.variant['budget_policy']}-{self.variant['gain_estimate']}"
        if self.variant["preissue_crossing"]:
            base += "-preissued"
        if self.variant["inline_transport_required"]:
            base += "-ready-only"
        if self.variant["stock_cut_observer"]:
            base += "-stock-cut-observer"
        return base

    @cached_property
    def match(self):
        return (*super().match, self.variant["stock_cut_observer"])


def active_status(row):
    if row["stock_cut_observer"]:
        return "DIAGNOSTIC_OBSERVER"
    if row["execution_mode"] == "shadow":
        return "DIAGNOSTIC_SHADOW"
    if row["execution_mode"] != "active":
        return "LEGACY_IDLE10_BASELINE" if row["method"].startswith("idle10") else "NATIVE_BASELINE"
    if row.get("suffix_executions") is None:
        return "ACTIVE_POLICY_PROXY_COVERAGE_MISSING"
    if row["suffix_executions"] <= 0:
        return "ACTIVE_POLICY_NO_OFFLOAD"
    return "ACTIVE_C3_OFFLOAD_WITNESSED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = aggregate(args.runs, args.output, cohort_factory=RecoveryCohort)
    for row in bundle["clean_cohorts"]:
        row["active_status"] = active_status(row)
    output = args.output / "online_evidence.json"
    output.write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"records": bundle["counts"]["cohorts"], "clean_cohorts": len(bundle["clean_cohorts"])}))


if __name__ == "__main__":
    main()
