"""Executable interface for the Part-1 completion bundle.

The commands in this module are deliberately thin wrappers around the GPU
programs that produced the validation artifacts.  They preserve the exact
command line in a dry-run manifest, propagate the child exit status, and do
not turn a missing measurement into a zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .contract import audit_environment, parse_physical_gpus
from .io import atomic_json, completed_ids


def _command(module: str, arguments: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def _execute(command: list[str], *, dry_run: bool, manifest: Path | None,
             raw_log: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "mllm_ep_completion_command.v1",
        "command": command,
        "status": "DRY_RUN" if dry_run else "RUNNING",
    }
    if manifest is not None:
        atomic_json(manifest, payload)
    if dry_run:
        return payload
    if raw_log is None:
        completed = subprocess.run(command, check=False)
    else:
        raw_log.parent.mkdir(parents=True, exist_ok=True)
        with raw_log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                       check=False)
    payload["exit_code"] = completed.returncode
    payload["status"] = "COMPLETE" if completed.returncode == 0 else "FAILED"
    if manifest is not None:
        atomic_json(manifest, payload)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    return payload


def _gpu_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--physical-gpus", default="4,5,6,7")
    parser.add_argument("--lane", choices=("official", "deterministic"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-cache-gib", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-manifest", type=Path)
    parser.add_argument("--raw-log", type=Path)


def _gpu_command(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    parse_physical_gpus(args.physical_gpus)
    values = [
        "--physical-gpus", args.physical_gpus, "--backend", backend,
        "--lane", args.lane, "--manifest", str(args.manifest),
        "--output", str(args.output), "--batch-size", str(args.batch_size),
        "--max-num-seqs", str(args.max_num_seqs),
        "--kv-cache-gib", str(args.kv_cache_gib),
    ]
    for flag, value in (("--max-tokens", args.max_tokens),
                        ("--max-model-len", args.max_model_len)):
        if value is not None:
            values += [flag, str(value)]
    if args.resume:
        values.append("--resume")
    if getattr(args, "ignore_eos", False):
        values.append("--ignore-eos")
    profile_dir = getattr(args, "profile_dir", None)
    if profile_dir is not None:
        values += ["--profile-dir", str(profile_dir),
                   "--warmup-repeats", str(args.warmup_repeats),
                   "--measured-repeats", str(args.measured_repeats)]
    return _execute(
        _command("poc_mllm_ep_foundation.scripts.run_full_model", values),
        dry_run=args.dry_run, manifest=args.dry_run_manifest, raw_log=args.raw_log,
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(prog="mllm-ep-foundation-completion")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("audit", "diagnose_baseline"):
        item = sub.add_parser(name)
        item.add_argument("--physical-gpus", default="4,5,6,7")
        item.add_argument("--model", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--dry-run", action="store_true")

    reference = sub.add_parser("evaluate_reference")
    _gpu_options(reference)

    physical = sub.add_parser("evaluate_physical_ep")
    _gpu_options(physical)
    physical.add_argument("--ep", type=int, choices=(2, 4), required=True)

    anchor = sub.add_parser("collect_physical_anchor")
    _gpu_options(anchor)
    anchor.add_argument("--ep", type=int, choices=(2, 4), required=True)
    anchor.add_argument("--profile-dir", type=Path, required=True)
    anchor.add_argument("--warmup-repeats", type=int, default=1)
    anchor.add_argument("--measured-repeats", type=int, default=3)

    fit = sub.add_parser("fit_cost_model")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--ep2-root", type=Path, required=True)
    fit.add_argument("--ep4-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--final-test-manifest", type=Path)
    fit.add_argument("--ep2-final-test-root", type=Path)
    fit.add_argument("--ep4-final-test-root", type=Path)
    fit.add_argument("--supplemental-manifest", type=Path)
    fit.add_argument("--ep2-supplemental-root", type=Path)
    fit.add_argument("--ep4-supplemental-root", type=Path)
    fit.add_argument("--coherent-request-target", action="store_true")
    fit.add_argument("--dry-run", action="store_true")
    fit.add_argument("--dry-run-manifest", type=Path)
    fit.add_argument("--raw-log", type=Path)

    validate = sub.add_parser("validate_cost_model")
    validate.add_argument("--cost-model", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)

    virtual = sub.add_parser("run_virtual")
    virtual.add_argument("--physical-gpus", default="4")
    virtual.add_argument("--trace-root", type=Path, required=True)
    virtual.add_argument("--target-ep", type=int, choices=(2, 4, 8), required=True)
    virtual.add_argument("--output", type=Path, required=True)
    virtual.add_argument("--cost-model", type=Path)
    virtual.add_argument("--prediction-output", type=Path)
    virtual.add_argument("--dry-run", action="store_true")
    virtual.add_argument("--dry-run-manifest", type=Path)
    virtual.add_argument("--raw-log", type=Path)

    predict = sub.add_parser("predict_from_trace")
    predict.add_argument("--physical-gpus", default="4")
    predict.add_argument("--replay", type=Path, required=True)
    predict.add_argument("--cost-model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--dry-run", action="store_true")
    predict.add_argument("--dry-run-manifest", type=Path)
    predict.add_argument("--raw-log", type=Path)

    ep8 = sub.add_parser("project_ep8_sensitivity")
    ep8.add_argument("--replay", type=Path, required=True)
    ep8.add_argument("--cost-model", type=Path, required=True)
    ep8.add_argument("--local-kernel", type=Path, required=True)
    ep8.add_argument("--ep2-to-ep4-transfer", type=Path, required=True)
    ep8.add_argument("--output", type=Path, required=True)
    ep8.add_argument("--dry-run", action="store_true")
    ep8.add_argument("--dry-run-manifest", type=Path)
    ep8.add_argument("--raw-log", type=Path)

    resume = sub.add_parser("resume")
    resume.add_argument("--request-manifest", type=Path, required=True)
    resume.add_argument("--result-index", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)

    export = sub.add_parser("export_bundle")
    export.add_argument("--asset-manifest", type=Path, required=True,
                        help="JSON list of paths, or an object with an assets list")
    export.add_argument("--output", type=Path, required=True)

    verify = sub.add_parser("verify_bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command in ("audit", "diagnose_baseline"):
        result = audit_environment(args.output, args.physical_gpus, args.model, args.dry_run)
    elif args.command == "evaluate_reference":
        result = _gpu_command(args, "reference")
    elif args.command == "evaluate_physical_ep":
        result = _gpu_command(args, f"physical_ep{args.ep}")
    elif args.command == "collect_physical_anchor":
        result = _gpu_command(args, f"physical_ep{args.ep}")
    elif args.command == "fit_cost_model":
        values = ["--manifest", str(args.manifest), "--ep2-root", str(args.ep2_root),
                  "--ep4-root", str(args.ep4_root), "--output", str(args.output)]
        for flag, value in (("--final-test-manifest", args.final_test_manifest),
                            ("--ep2-final-test-root", args.ep2_final_test_root),
                            ("--ep4-final-test-root", args.ep4_final_test_root),
                            ("--supplemental-manifest", args.supplemental_manifest),
                            ("--ep2-supplemental-root", args.ep2_supplemental_root),
                            ("--ep4-supplemental-root", args.ep4_supplemental_root)):
            if value is not None:
                values += [flag, str(value)]
        if args.coherent_request_target:
            values.append("--coherent-request-target")
        result = _execute(_command(
            "poc_mllm_ep_foundation.scripts.analyze_physical_anchors_v4", values),
            dry_run=args.dry_run, manifest=args.dry_run_manifest, raw_log=args.raw_log)
    elif args.command == "validate_cost_model":
        source = json.loads(args.cost_model.read_text())
        validations = source.get("validation", source.get("validations", {}))
        result = {"schema_version": "mllm_ep_cost_validation_summary.v1",
                  "source": str(args.cost_model.resolve()), "validation": validations,
                  "status": "PRESENT" if validations else "MISSING_VALIDATION"}
        atomic_json(args.output, result)
    elif args.command == "run_virtual":
        parse_physical_gpus(args.physical_gpus)
        result = _execute(_command(
            "poc_mllm_ep_foundation.scripts.build_virtual_replay_v2",
            ["--trace-root", str(args.trace_root), "--target-ep", str(args.target_ep),
             "--output", str(args.output)]), dry_run=args.dry_run,
            manifest=args.dry_run_manifest, raw_log=args.raw_log)
        if not args.dry_run and args.cost_model is not None:
            if args.prediction_output is None:
                raise ValueError("--prediction-output is required with --cost-model")
            result["prediction"] = _execute(_command(
                "poc_mllm_ep_foundation.scripts.predict_replay_v4",
                ["--physical-gpus", args.physical_gpus, "--replay", str(args.output),
                 "--cost-model", str(args.cost_model), "--output",
                 str(args.prediction_output)]), dry_run=False, manifest=None,
                raw_log=args.raw_log)
    elif args.command == "predict_from_trace":
        parse_physical_gpus(args.physical_gpus)
        result = _execute(_command(
            "poc_mllm_ep_foundation.scripts.predict_replay_v4",
            ["--physical-gpus", args.physical_gpus, "--replay", str(args.replay),
             "--cost-model", str(args.cost_model), "--output", str(args.output)]),
            dry_run=args.dry_run, manifest=args.dry_run_manifest, raw_log=args.raw_log)
    elif args.command == "project_ep8_sensitivity":
        result = _execute(_command(
            "poc_mllm_ep_foundation.scripts.project_ep8_sensitivity",
            ["--replay", str(args.replay), "--cost-model", str(args.cost_model),
             "--local-kernel", str(args.local_kernel),
             "--ep2-to-ep4-transfer", str(args.ep2_to_ep4_transfer),
             "--output", str(args.output)]),
            dry_run=args.dry_run, manifest=args.dry_run_manifest,
            raw_log=args.raw_log)
    elif args.command == "resume":
        manifest = json.loads(args.request_manifest.read_text())
        expected = [str(row["request_id"]) for row in manifest["requests"]]
        done = completed_ids(args.result_index)
        result = {"schema_version": "mllm_ep_resume.v1", "completed": sorted(done),
                  "pending": [value for value in expected if value not in done],
                  "unexpected": sorted(done - set(expected)),
                  "status": "COMPLETE" if set(expected) <= done else "READY_TO_RESUME"}
        atomic_json(args.output, result)
    elif args.command == "export_bundle":
        source = json.loads(args.asset_manifest.read_text())
        assets = source if isinstance(source, list) else source["assets"]
        records = []
        for value in assets:
            path = Path(value).resolve()
            records.append({"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size})
        result = {"schema_version": "mllm_ep_frozen_bundle.v2", "assets": records,
                  "weights_embedded": False, "credentials_embedded": False}
        atomic_json(args.output, result)
    elif args.command == "verify_bundle":
        bundle = json.loads(args.bundle.read_text())
        checks = []
        for item in bundle["assets"]:
            path = Path(item["path"])
            observed = _sha(path) if path.is_file() else None
            checks.append({"path": str(path), "expected_sha256": item["sha256"],
                           "observed_sha256": observed, "ok": observed == item["sha256"]})
        result = {"schema_version": "mllm_ep_bundle_verification.v1", "checks": checks,
                  "status": "PASS" if all(item["ok"] for item in checks) else "FAIL"}
        atomic_json(args.output, result)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
