from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc_mllm_ep_foundation.io import atomic_json
from poc_mllm_resolution_expert_tradeoff.scripts.summarize_physical_costs import _wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--custom-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native = _wall(args.native_root, "R0_K8")
    custom = _wall(args.custom_root, "R0_K8")
    if native is None or custom is None:
        raise RuntimeError("native/custom measured timing missing")
    result = {
        "schema_version": "mllm_resolution_expert_tradeoff.noop_runtime.v1",
        "scope": "TRUE_EP4_DETERMINISTIC_ONE_TOKEN_PAIRED_REPRESENTATIVE_COHORT",
        "native_fused_p0_ms": native,
        "custom_mixed_k_adapter_p0_ms": custom,
        "custom_overhead_fraction": custom / native - 1,
        "custom_overhead_pct": 100 * (custom / native - 1),
        "evidence": "MEASURED_TRUE_EP4",
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
