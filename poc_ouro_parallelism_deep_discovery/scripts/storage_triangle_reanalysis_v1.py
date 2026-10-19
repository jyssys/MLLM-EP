"""Read six frozen NPZ files and audit the three-arm numerical triangle. No Torch import."""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = {"v4alias": "storage_alias_base14_P2_fully_isolated_v4",
        "v4copies": "storage_copies_base14_P2_fully_isolated_v4",
        "v5alias": "storage_alias_base14_P2_fully_isolated_repeat_v5"}
POSITIONS = [0, 1, 1023, 1024, 1026]
FEATURES = [0, 1, 2, 3, 124, 125, 126, 127, 128, 129, 130, 131, 252, 253, 254, 255]
SOURCES = ["scripts/compare_same_storage_repeat.py", "scripts/compare_storage_controls.py",
           "scripts/native_parity_analysis.py", "scripts/native_baseline.py", "worker_probe.py", "alias_control.py"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            result.update(block)
    return result.hexdigest()


def array_stats(left, right):
    if left.shape != right.shape or left.dtype != right.dtype or left.dtype != np.float32:
        raise ValueError("The frozen capture shape or float32 transport dtype differs.")
    mask = left.view(np.uint32) != right.view(np.uint32)
    indices = np.argwhere(mask)
    delta = right.astype(np.float64) - left.astype(np.float64)
    first = None
    if len(indices):
        index = tuple(int(i) for i in indices[0])
        first = dict(index=list(index), left=float(left[index]), right=float(right[index]),
                     left_word=hex(int(left.view(np.uint32)[index])),
                     right_word=hex(int(right.view(np.uint32)[index])))
    return dict(shape=list(left.shape), dtype=str(left.dtype), elements=int(left.size),
        bitwise_mismatches=int(mask.sum()), numeric_mismatches=int(np.count_nonzero(left != right)),
        bitwise_equal=not bool(mask.any()), first_lexicographic_difference=first,
        max_abs=float(np.max(np.abs(delta))),
        relative_L2_legacy_reference_float32_norm=float(np.linalg.norm(delta) / max(float(np.linalg.norm(left)), 1e-12)),
        relative_L2_reference_float64_norm=float(np.linalg.norm(delta) / max(float(np.linalg.norm(left.astype(np.float64))), 1e-12)))


def analyze():
    inputs, loaded, records, metadata, names = {}, {}, {}, {}, None
    normalized_configs, workload_hashes, cache_roots = {}, {}, {}
    for arm, name in RUNS.items():
        run, child = ROOT / "outputs" / name, ROOT / "outputs" / name / "1.4B_R4_TP2"
        capture = read(child / "parity_capture.json")
        config = read(child / "engine_config.json")
        normalized_configs[arm] = json.loads(json.dumps(config))
        normalized_configs[arm]["compilation_config"].pop("cache_dir")
        workload_hashes[arm] = sha(child / "workload.json")
        cache_roots[arm] = read(child / "cache_environment.json")["variables"]
        contract, storage = read(run / "run_contract.json"), read(child / "storage_control.json")
        metadata[arm] = dict(run=str(run), command=contract["command"], mask=contract["mask"],
            requested_modes=[v["mode"] for v in storage],
            generated_ids=capture["generated"]["output_ids"], capture_wall_ms=capture["generated"]["wall_ms"],
            parity_capture_json=str(child / "parity_capture.json"),
            capture_record_count=len(capture["evidence"]),
            numerical_call_count_source="Capture record logits_calls. No independent per-call trace was saved.",
            completed_timing_trials=len(list(child.glob("P*_request*_repeat*.json"))))
        for path in [child / "parity_capture.json", child / "engine_config.json", child / "workload.json",
                     child / "cache_environment.json", child / "storage_control.json", run / "run_contract.json"]:
            inputs[str(path)] = sha(path)
        rows = []
        for rank in range(2):
            evidence = next(v for v in capture["evidence"] if v["rank"] == rank)
            if evidence["positions"] != POSITIONS or evidence["logits_calls"] != 4 or evidence["teacher_steps"] != 4:
                raise ValueError("The frozen capture call count or sampled positions differ.")
            if names is None:
                names = evidence["KV_names"]
            if evidence["KV_names"] != names:
                raise ValueError("The frozen KV-name orders differ.")
            # Open the independently constructed expected path, not the metadata path.
            path = run / f"base_1.4B_R4_TP2.rank{rank}.npz"
            digest, info = sha(path), path.stat()
            inputs[str(path)] = digest
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in ("logits", "hidden", "KV")}
            loaded[arm, rank] = arrays
            arrays_meta = {key: dict(shape=list(v.shape), dtype=str(v.dtype), bytes=int(v.nbytes),
                sha256=hashlib.sha256(v.tobytes(order="C")).hexdigest(), finite_values=int(np.isfinite(v).sum()),
                elements=int(v.size), all_float32_low16_bits_zero=bool(np.all((v.view(np.uint32) & 65535) == 0)))
                for key, v in arrays.items()}
            with zipfile.ZipFile(path) as container:
                zip_members = [dict(name=v.filename, date_time=list(v.date_time), CRC=v.CRC,
                                    bytes=v.file_size, compressed_bytes=v.compress_size) for v in container.infolist()]
            rows.append(dict(rank=rank, expected_path=str(path), recorded_path=evidence["path"],
                recorded_path_matches_expected=Path(evidence["path"]).resolve() == path.resolve(),
                sha256=digest, recorded_sha256=evidence["sha256"], sha256_matches_record=evidence["sha256"] == digest,
                regular_file=path.is_file(), is_symlink=path.is_symlink(), device=info.st_dev, inode=info.st_ino,
                file_bytes=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                logits_calls=evidence["logits_calls"], teacher_steps=evidence["teacher_steps"],
                cache_slots=evidence["cache_slots"], block_ids=evidence["block_ids"], arrays=arrays_meta,
                zip_members=zip_members))
        records[arm] = rows
        relevant = ["alias_control.py", "worker_probe.py", "scripts/native_baseline.py", "scripts/compare_storage_controls.py"]
        with zipfile.ZipFile(run / "source_snapshot.zip") as snapshot:
            metadata[arm]["snapshot_members"] = len(snapshot.namelist())
            metadata[arm]["snapshot_sources"] = [{"path": key,
                "sha256": hashlib.sha256(snapshot.read(key)).hexdigest(), "matches_current":
                hashlib.sha256(snapshot.read(key)).hexdigest() == sha(ROOT / key)} for key in relevant]
            metadata[arm]["external_capture_helper_in_snapshot"] = any(
                "poc_looped_recurrence_tp_4models/worker_probe.py" in key for key in snapshot.namelist())
    pairs = []
    for left, right in combinations(RUNS, 2):
        for rank in range(2):
            a, b = loaded[left, rank], loaded[right, rank]
            fields = {}
            for key in ("logits", "hidden", "KV"):
                result = array_stats(a[key], b[key])
                if key != "KV":
                    result["calls"] = [dict(call=i, expected_token_position=1023 + i,
                        phase_inferred_from_runner="PREFILL_LAST_PROMPT_TOKEN" if i == 0 else "CACHED_DECODE",
                        **array_stats(a[key][i], b[key][i])) for i in range(4)]
                else:
                    result["positions"] = [dict(position=p, phase="PREFILL_HISTORY" if p < 1024 else "CACHED_DECODE_WRITE",
                        **array_stats(a[key][:, i], b[key][:, i])) for i, p in enumerate(POSITIONS)]
                    mask = a[key].view(np.uint32) != b[key].view(np.uint32)
                    slots = []
                    for i, name in enumerate(names):
                        logical = int(re.search(r"layers\.(\d+)", name).group(1))
                        slots.append(dict(array_cache_index=i, name=name, logical_layer=logical,
                            depth=logical // 24, physical_layer=logical % 24,
                            mismatches_by_sampled_position=[int(mask[i, j].sum()) for j in range(5)],
                            prefill_mismatches=int(mask[i, :3].sum()), prefill_elements=96,
                            cached_mismatches=int(mask[i, 3:].sum()), cached_elements=64))
                    slots.sort(key=lambda row: (row["depth"], row["physical_layer"]))
                    result["slot_counts_in_execution_order"] = slots
                    for phase, field in [("prefill", "prefill_mismatches"), ("cached", "cached_mismatches")]:
                        first = next((row for row in slots if row[field]), None)
                        result[f"first_sampled_{phase}_slot_in_execution_order"] = first
                        if first:
                            i = first["array_cache_index"]
                            phase_indices = range(3) if phase == "prefill" else range(3, 5)
                            differences = [(j, h, f) for j in phase_indices for h in range(2)
                                           for f in range(16) if mask[i, j, h, f]]
                            j, h, f = differences[0]
                            result[f"first_sampled_{phase}_coordinate_in_execution_order"] = dict(
                                slot=first, token_position=POSITIONS[j], local_head=h, global_head=rank * 8 + h,
                                selected_feature_index=f, head_feature=FEATURES[f] % 128,
                                kind="K" if FEATURES[f] < 128 else "V", left=float(a[key][i, j, h, f]),
                                right=float(b[key][i, j, h, f]))
                fields[key] = result
            pairs.append(dict(left=left, right=right, rank=rank,
                raw_npz_sha256_equal=records[left][rank]["sha256"] == records[right][rank]["sha256"],
                fields=fields, top1_match=(a["logits"].argmax(-1) == b["logits"].argmax(-1)).tolist()))
    old_repeat = read(ROOT / "artifacts/storage_same_alias_repeat_v1.json")
    old_controls = read(ROOT / "artifacts/storage_fully_isolated_checkpoint3.json")
    legacy = next(row for row in old_controls["records"] if row["model"] == "1.4B" and row["P"] == 2)
    historical_metrics = []
    for rank in range(2):
        for right, old in [("v4copies", legacy["capture_comparisons"][rank]),
                           ("v5alias", old_repeat["numerical"][rank])]:
            fresh = next(row for row in pairs if row["left"] == "v4alias" and row["right"] == right and row["rank"] == rank)
            old_fields = old.get("comparisons", old)
            historical_metrics.append(dict(right=right, rank=rank, fields={key: dict(
                recorded_relative_L2=old_fields[key]["relative_L2"],
                recomputed_relative_L2=fresh["fields"][key]["relative_L2_legacy_reference_float32_norm"],
                exactly_matches=old_fields[key]["relative_L2"] == fresh["fields"][key]["relative_L2_legacy_reference_float32_norm"])
                for key in ("logits", "hidden", "KV")}))
    sources = {name: dict(path=str(ROOT / name), sha256=sha(ROOT / name)) for name in SOURCES}
    external = ROOT.parent / "poc_looped_recurrence_tp_4models/worker_probe.py"
    sources["external_capture_helper"] = dict(path=str(external), sha256=sha(external),
        historical_snapshot_status="NOT_INCLUDED in these three campaign source_snapshot.zip files.")
    return dict(runs=metadata, raw_captures=records, KV_name_order=names, sampled_positions=POSITIONS,
        selected_packed_KV_features=FEATURES, comparisons=pairs, historical_metric_reproduction=historical_metrics,
        normalized_engine_configs_equal=all(v == normalized_configs["v4alias"] for v in normalized_configs.values()),
        workload_sha256=workload_hashes, all_workloads_identical=len(set(workload_hashes.values())) == 1,
        cache_environment_roots=cache_roots,
        all_nine_cache_environment_roots_distinct=len({v for roots in cache_roots.values() for v in roots.values()}) == 9,
        sources=sources, input_hashes_before=inputs,
        input_hashes_after={path: sha(path) for path in inputs},
        all_inputs_unchanged=all(sha(path) == digest for path, digest in inputs.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    target = args.output.resolve()
    if target.exists() or not target.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Use a new artifact path within this campaign.")
    started = datetime.now(timezone.utc)
    result = analyze()
    result.update(schema_version=1, analysis_start_UTC=started.isoformat(),
        analysis_end_UTC=datetime.now(timezone.utc).isoformat(), worker="independent_axes",
        script_sha256=sha(Path(__file__)), numpy_version=np.__version__, python=sys.version,
        GPU_execution="NOT_RUN", Torch_import="NOT_RUN", raw_file_changes="NONE",
        evidence_class="READ_ONLY_REANALYSIS_OF_FROZEN_CAPTURES",
        phase_mapping_limit="Call order follows the unchunked, prefix-cache-disabled four-token runner protocol. No per-call position trace exists in NPZ.",
        first_difference_limit="KV entries sample two local heads and16 packed features. First sampled difference is not the first complete-tensor or kernel difference.",
        causal_claim="NONE. No kernel, HBM, quality, or new-architecture conclusion.")
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(output=str(target), pair_rank_comparisons=len(result["comparisons"]),
        raw_files=sum(len(v) for v in result["raw_captures"].values()),
        all_inputs_unchanged=result["all_inputs_unchanged"],
        historical_metrics_reproduced=all(v["exactly_matches"] for r in result["historical_metric_reproduction"] for v in r["fields"].values())), sort_keys=True))


if __name__ == "__main__":
    main()
