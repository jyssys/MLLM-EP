from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ..contract import MODEL_REVISION, atomic_json, read_json


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root, artifacts = args.root, args.root / "artifacts"
    compact = [
        artifacts / "baseline_audit.json",
        artifacts / "manifests/frozen_visual_grounding_cohort.json",
        artifacts / "manifests/frozen_structured_probe_plan.json",
        artifacts / "visual_grounding_signals.jsonl.gz",
        artifacts / "structured_damage_labels.jsonl.gz",
        artifacts / "phase1_gate.json",
        artifacts / "exploratory_distributional_diagnostic.json",
        artifacts / "final_verdict.json",
        artifacts / "gpu_restore_final.json",
    ]
    missing = [str(path) for path in compact if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    capture = read_json(artifacts / "raw_runs/subspace_ep4_v3/run.json")
    probes = read_json(artifacts / "raw_runs/structured_ep4_v2/run.json")
    gate, verdict = read_json(artifacts / "phase1_gate.json"), read_json(artifacts / "final_verdict.json")
    atomic_json(artifacts / "research_contract.json", {
        "question": "Does current-site visual grounding predict whether native K8-to-K6/K4 reduction is safe?",
        "model_revision": MODEL_REVISION, "dtype": "BF16", "topology": "TRUE_TP2_DP2_EP4_PP1",
        "physical_gpus": [2,3,6,7], "expert_policy": "NATIVE_ROUTER_TAIL_DIAGNOSTIC_ONLY",
        "phase1_only": True, "phase2_expert_selection": "NOT_IMPLEMENTED",
        "modes_comparison": "NOT_RUN_BY_USER_SCOPE", "latency": "NOT_RUN",
        "heldout_gate_frozen_before_damage": True,
    })
    commands = [
        "PYTHONPATH=. python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.audit_baseline ...",
        "PYTHONPATH=. python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.build_cohort ...",
        "CUDA_VISIBLE_DEVICES=2,3,6,7 PYTHONPATH=. /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.run_subspace_capture_ep4 ...",
        "PYTHONPATH=. python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.build_structured_probe_plan ...",
        "CUDA_VISIBLE_DEVICES=2,3,6,7 PYTHONPATH=. /home/esjung/.venvs/flashvep-deepep-v020/bin/python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.run_structured_probe_ep4 ...",
        "PYTHONPATH=. python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.analyze_visual_grounding ...",
        "PYTHONPATH=. python -m poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.scripts.analyze_distributional_diagnostics ...",
        "PYTHONPATH=. pytest -q poc_mllm_rank_aware_active_expert_reduction/q2_3_visual_grounding_budget/tests",
    ]
    atomic_json(artifacts / "reproducible_commands.json", {"canonical_wrapper": str(root / "reproduce.sh"), "commands": commands})
    compact.extend((artifacts / "research_contract.json", artifacts / "reproducible_commands.json"))
    atomic_json(artifacts / "execution_manifest.json", {
        "schema_version": 1, "status": "COMPLETE", "model_revision": MODEL_REVISION,
        "physical_gpus": [2,3,6,7], "topology": "TRUE_TP2_DP2_EP4_PP1",
        "subspace_condition_executions": capture["condition_executions"],
        "structured_probe_executions": probes["probe_count"],
        "heldout_requests": gate["heldout_requests"], "exact_marker_audit": gate["marker_audit"],
        "phase1_gate": "PASS" if gate["gate_pass"] else "FAIL", "verdict": verdict["verdict"],
        "phase2": "NOT_RUN_BY_GATE", "latency": "NOT_RUN", "modes_comparison": "NOT_RUN_BY_USER_SCOPE",
        "failed_attempts_preserved_locally": [
            "raw_runs/subspace_ep4: prefill SP-mask mismatch fixed before canonical rerun",
            "raw_runs/subspace_smoke_v2 and structured_smoke_v1: bounded successful smoke runs",
        ],
        "compact_artifacts": [{"path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size} for path in compact],
    })
    print(read_json(artifacts / "execution_manifest.json"))


if __name__ == "__main__": main()
