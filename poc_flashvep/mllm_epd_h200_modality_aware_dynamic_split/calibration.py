"""Measure H200 exact imported-KV continuations on isolated owned E/P/D workers."""
import asyncio
from collections import defaultdict
from contextlib import contextmanager
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import time

from .modality import inventory, read, require, write


NATIVE_KV_BYTES_PER_TOKEN = 28 * 2 * 4 * 128 * 2
SEED = 2026112607
ON_DEMAND = "on_demand"
INLINE_REDIRECT = "inline_redirect"
METRICS = {
    "bytes": "vllm:nixl_bytes_transferred_sum",
    "transfers": "vllm:nixl_bytes_transferred_count",
    "descriptors": "vllm:nixl_num_descriptors_sum",
    "transfer_ms": "vllm:nixl_xfer_time_seconds_sum",
    "post_ms": "vllm:nixl_post_time_seconds_sum",
}


async def get(session, port, path, *, as_text=False):
    async with session.get(f"http://127.0.0.1:{port}{path}") as response:
        response.raise_for_status()
        return await (response.text() if as_text else response.json())


def parse_nixl_metrics(payload):
    metric_names = {name: key for key, name in METRICS.items()}
    counts = {}
    for line in payload.splitlines():
        metric, sep, value = line.partition(" ")
        key = metric_names.get(metric.split("{", 1)[0])
        if sep and key is not None:
            counts[key] = counts.get(key, 0.) + float(value.split()[0])
    return counts


def counter_delta(before, after):
    keys = before.keys() & after.keys()
    differences = {key: after[key] - before[key] for key in keys}
    if not differences or any(value < 0 for value in differences.values()):
        return None
    return differences


@contextmanager
def nvlink(indices):
    """Read device-level data counters, with no reset or polling during GPU work."""
    try:
        import pynvml as nvml
    except ImportError:
        yield lambda: None
        return
    try:
        nvml.nvmlInit()
    except (OSError, nvml.NVMLError):
        yield lambda: None
        return

    def sample():
        request = (nvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
                   nvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX)
        data = {}
        for role, index in zip(("E", "P", "D"), indices, strict=True):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            fields = [(kind, link) for kind in request for link in range(18)]
            values = nvml.nvmlDeviceGetFieldValues(handle, fields)
            if not all(item.nvmlReturn == nvml.NVML_SUCCESS for item in values):
                return None
            for label, kind in zip(("TX", "RX"), request, strict=True):
                data[f"{role}_{label}_bytes"] = 1024 * sum(int(item.value.ullVal) for item in values
                                                          if item.fieldId == kind)
        return data
    def safe_sample():
        try:
            return sample()
        except (OSError, nvml.NVMLError):
            return None

    try:
        yield safe_sample
    finally:
        nvml.nvmlShutdown()


def observe_reader(run):
    """Read existing whole-tensor clone observations once, after all timings."""
    clones = defaultdict(list)
    for role in ("prefill", "decode"):
        for path in (run / "raw").glob(f"ec_cuda_ipc_{role}_gpu*.jsonl"):
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    for key in record["keys"]:
                        clones[key].append((role, record))
    return clones


def buffer_observation(ec_path, image_keys):
    lifetimes = []
    payload = {}
    for key in image_keys:
        folder = ec_path / key
        ready_path = folder / "cuda_ipc_ready.json"
        ready = read(ready_path)
        released = read(folder / "cuda_ipc_released.json")
        require(ready["source_original_E"] is True and ready["producer_pid"] == released["producer_pid"],
                "Source/release identity mismatch")
        length, width = ready["shape"]
        require(ready["dtype"] == "torch.bfloat16" and width == 3584, "Unexpected original E tensor geometry")
        payload[key] = length * width * 2
        elapsed_ms = (released["time"] - ready_path.stat().st_mtime) * 1000
        require(elapsed_ms >= 0, "Invalid E buffer marker clocks")
        lifetimes.append(elapsed_ms)
    return {"producer_host_marker_lifetime_max_ms": max(lifetimes),
            "producer_host_marker_lifetime_mean_ms": sum(lifetimes) / len(lifetimes),
            "source_tensor_payload_by_key": payload}


async def trial(session, row, cell, name, ec_path, sample_nvlink=None):
    from ..mllm_epd_prefill_heavy_token_split.batch_token_correctness_control import headers, post
    from ..mllm_epd_prefill_heavy_token_split.client import payload
    from ..mllm_epd_prefill_heavy_token_split.token_split import PLAN_KEY

    ep, pp, dp = 18840, 18841, 18842
    length = row["total_input_tokens"]
    prefix, suffix = cell["prefix_tokens"], cell["suffix_tokens"]
    body = payload(row, name, 1, False)
    # Encode the original input without prescribing future P/D image ownership.
    # The epoch's transport determines whether E issues P descriptors in advance.
    mark = time.monotonic()
    encoder_output = await post(session, ep, "/v1/chat/completions", body, headers(name, "E:full"))
    e_ms = 1000 * (time.monotonic() - mark)
    require(encoder_output["usage"]["prompt_tokens"] == length, "E rendered a different prompt")
    key = sha256(name.encode()).hexdigest()
    render_path = ec_path / "processed_prompts" / key / "manifest.json"
    render = read(render_path)
    require(render["images"] == row["image_count"] and render["prompt_tokens"] == length, "E metadata mismatch")
    image_keys = render["image_hashes"]

    source = dict(do_remote_decode=True, do_remote_prefill=False, remote_engine_id=None,
                  remote_block_ids=None, remote_host=None, remote_port=None)
    transfer_plan = {PLAN_KEY: {"prefix_tokens": prefix, "full_prompt_tokens": length}} if suffix else {}
    marker = sample_nvlink() if sample_nvlink is not None else None
    mark = time.monotonic()
    prefill_output = await post(session, pp, "/v1/chat/completions",
                                dict(body, kv_transfer_params=source | transfer_plan), headers(name, "P"))
    p_ms = 1000 * (time.monotonic() - mark)
    params = prefill_output["kv_transfer_params"]
    require(params["do_remote_prefill"] and params["remote_block_ids"], "No real remote KV export")
    if suffix:
        require(params["remote_num_tokens"] == prefix, "P did not export the requested prefix")
    p_snapshot = sample_nvlink() if marker is not None else None
    nixl_before = parse_nixl_metrics(await get(session, dp, "/metrics", as_text=True)) if sample_nvlink else None
    if sample_nvlink:
        producer_state, = await get(session, ep, "/campaign/ec_ipc/state")
    else:
        producer_state = None
    mark = time.monotonic()
    output = await post(session, dp, "/v1/chat/completions",
                        dict(body, kv_transfer_params=params | transfer_plan), headers(name, "D"))
    d_ms = 1000 * (time.monotonic() - mark)
    ids = output["choices"][0]["token_ids"]
    require(output["usage"]["prompt_tokens"] == length and output["usage"]["completion_tokens"] == len(ids) == 1,
            "D did not complete the same original input with one greedy token")
    d_snapshot = sample_nvlink() if p_snapshot is not None else None
    if nixl_before is not None:
        nixl_after = parse_nixl_metrics(await get(session, dp, "/metrics", as_text=True))
        # API statistics can appear one output-loop tick after the completion.
        for _ in range(4):
            if nixl_after.get("transfers", 0) > nixl_before.get("transfers", 0):
                break
            await asyncio.sleep(.05)
            nixl_after = parse_nixl_metrics(await get(session, dp, "/metrics", as_text=True))
        nixl_delta = counter_delta(nixl_before, nixl_after)
        if nixl_delta:
            for metric in ("transfer_ms", "post_ms"):
                if metric in nixl_delta:
                    nixl_delta[metric] *= 1000
            if not nixl_delta.get("transfers"):
                nixl_delta = None
    else:
        nixl_delta = None

    mark = time.monotonic()
    state, = await post(session, ep, "/campaign/ec_ipc/release", {"key": key})
    release_ms = 1000 * (time.monotonic() - mark)
    require(state["live_images"] == state["outstanding_exports"] == 0,
            "Serial diagnostic did not release EC buffers and import handles")
    observations = buffer_observation(ec_path, image_keys)
    return cell | {
        "trial": name, "input_tokens": length, "image_count": row["image_count"],
        "E_request_http_ms": e_ms, "P_source_http_ms": p_ms, "D_first_token_http_ms": d_ms,
        "E_release_rpc_ms": release_ms, "first_token": ids[0], "image_keys": image_keys,
        "logical_prefix_kv_bytes": prefix * NATIVE_KV_BYTES_PER_TOKEN,
        "nixl_D_counters": nixl_delta,
        "NVLink_P_stage_device_delta": counter_delta(marker, p_snapshot) if p_snapshot is not None else None,
        "NVLink_D_stage_device_delta": counter_delta(p_snapshot, d_snapshot) if d_snapshot is not None else None,
        "producer_buffer_storage_snapshot_bytes": producer_state["live_storage_bytes"] if producer_state else None,
        "E_registry_released": True,
    } | observations


def add_observed_clones(run, records):
    clones = observe_reader(run)
    for record in records:
        events = {"prefill": {}, "decode": {}}
        payload = record["source_tensor_payload_by_key"]
        for key in record["image_keys"]:
            for role, observed in clones[key]:
                events[role][id(observed)] = observed
        for role, values in events.items():
            assigned = list(values.values())
            prefix = f"EC_{role}_"
            for op in assigned:
                require(set(op["keys"]) <= payload.keys(), "An EC operation crossed isolated request ownership")
                require(op["logical_bytes"] == sum(payload[key] for key in op["keys"]),
                        "Observed cloned tensor payload does not match original BF16 shape")
            record.update({
                prefix + "clone_operations": sum(len(op["keys"]) for op in assigned),
                prefix + "clone_payload_bytes": sum(op["logical_bytes"] for op in assigned),
                prefix + "redirected_inline_images": sum(op.get("redirected_inline_images", 0) for op in assigned),
                prefix + "descriptor_groups": sum(op["inline_images"] + bool(op["images"] - op["inline_images"])
                                                  for op in assigned),
                prefix + "clone_sync_host_ms": sum(op["import_clone_sync_host_ms"] for op in assigned),
                prefix + "prepare_host_ms": sum(op["descriptor_prepare_host_ms"] for op in assigned),
            })
        pkeys = {key for op in events["prefill"].values() for key in op["keys"]}
        dkeys = {key for op in events["decode"].values() for key in op["keys"]}
        record["EC_duplicate_image_clones"] = len(pkeys & dkeys)
        record["EC_duplicate_cloned_tensor_payload_bytes"] = sum(payload[key] for key in pkeys & dkeys)
        record["observed_EC_all_images_consumed"] = pkeys | dkeys == payload.keys()


def model_features(row, variant):
    values = [row["input_tokens"] / 8192, row["prefix_tokens"] / 8192, row["suffix_tokens"] / 1024,
              row["causal_attention_pairs_per_layer"] / (8192 * 1024)]
    if variant in ("encoding_only", "modality"):
        values += [row["image_count"] / 10, row["E_request_http_ms"] / 1000]
    if variant in ("boundary_only", "modality"):
        values += [row["cross_boundary_image_count"], row["cross_boundary_tensor_payload_bytes"] / 2**20]
    return values


def suffix_upper_ms(generic_model, prefix, full):
    """Return isolate-based upper estimate for a supported B1 D continuation geometry."""
    require(0 < prefix < full, "Nonempty prefix and suffix required")
    suffix = full - prefix
    values = [full / 8192, prefix / 8192, suffix / 1024,
              (prefix * suffix + suffix * (suffix + 1) // 2) / (8192 * 1024)]
    require(len(generic_model["weights"]) == len(values), "Expected Generic cost model")
    center = generic_model["intercept"] + sum(weight * value for weight, value in zip(generic_model["weights"], values, strict=True))
    return max(0., center) + generic_model["training_p95_underprediction_ms"]


def fit_predictors(records, *, transport=ON_DEMAND):
    import numpy as np

    results = {}
    budgets = (16, 32, 64, 100, 160)
    training = [row for row in records if row["series"] == "cost" and row["fit_group"] == "fit"
                and row["suffix_tokens"] > 0]
    validation = [row for row in records if row["series"] == "cost" and row["fit_group"] == "validation"
                  and row["suffix_tokens"] > 0]
    require({r["request_id"] for r in training}.isdisjoint(r["request_id"] for r in validation), "Calibration leakage")
    for name in ("generic", "encoding_only", "boundary_only", "modality"):
        train_x = np.array([model_features(row, name) for row in training])
        fitted = np.column_stack((np.ones(len(training)), train_x))
        actual = np.array([row["D_first_token_http_ms"] for row in training])
        penalty = np.eye(fitted.shape[1]) * .1
        penalty[0, 0] = 0
        weights = np.linalg.solve(fitted.T @ fitted + penalty, fitted.T @ actual)
        margin = max(0., float(np.percentile(actual - fitted @ weights, 95)))
        evaluation = []
        for row in validation:
            begin = time.perf_counter_ns()
            prediction = max(0., float(weights[0] + np.dot(model_features(row, name), weights[1:])))
            overhead = (time.perf_counter_ns() - begin) / 1000
            evaluation.append({"trial": row["trial"], "request_id": row["request_id"],
                               "prediction_ms": prediction, "upper_ms": prediction + margin,
                               "actual_ms": row["D_first_token_http_ms"], "prediction_cpu_us": overhead})
        errors = [row["actual_ms"] - row["prediction_ms"] for row in evaluation]
        safety = {}
        for budget in budgets:
            predict_safe = [row for row in evaluation if row["upper_ms"] <= budget]
            actual_safe = [row for row in evaluation if row["actual_ms"] <= budget]
            correct = sum(row["actual_ms"] <= budget for row in predict_safe)
            safety[str(budget)] = {
                "predicted_safe": len(predict_safe), "actual_safe": len(actual_safe), "both_safe": correct,
                "false_safe_rate_among_predictions": 1 - correct / len(predict_safe) if predict_safe else None,
                "false_reject_rate_among_actual_safe": 1 - correct / len(actual_safe) if actual_safe else None,
            }
        results[name] = {
            "intercept": float(weights[0]), "weights": weights[1:].tolist(),
            "training_p95_underprediction_ms": margin,
            "validation_MAE_ms": float(np.mean(np.abs(errors))),
            "validation_relative_MAE": float(np.mean([abs(error) / row["actual_ms"]
                                                     for row, error in zip(evaluation, errors, strict=True)])),
            "validation_p95_underprediction_ms": max(0., float(np.percentile(errors, 95))),
            "validation_model_cpu_median_us": float(np.median([row["prediction_cpu_us"] for row in evaluation])),
            "validation_safety_at_fixed_budgets": safety, "validation_points": evaluation,
        }
    return {"evidence": "ISOLATED_GPU_MEASUREMENT", "transport": transport,
            "cost_scope": "D HTTP from imported prefix submit to one greedy token completion. Includes KV import, EC, suffix scheduling and output delivery.",
            "predictor_scope": "Same Ridge family with fixed lambda 0.1 and disjoint request IDs. No C3 decision accuracy or future idle truth measured. Upper estimate is not an SLA guarantee.",
            "base_features": ["input_tokens/8192", "prefix_tokens/8192", "suffix_tokens/1024",
                              "causal_attention_pairs_per_layer/(8192*1024)"],
            "additional_features": ["image_count/10", "observed_E_request_http_ms/1000",
                                    "cross_boundary_image_count", "cross_boundary_tensor_payload_bytes/MiB"],
            "training_observed_bounds": {key: [min(r[key] for r in training), max(r[key] for r in training)]
                                         for key in ("input_tokens", "prefix_tokens", "suffix_tokens")},
            "train_requests": sorted({row["request_id"] for row in training}),
            "heldout_calibration_requests": sorted({row["request_id"] for row in validation}),
            **results}


def transfer_pairs(records):
    grouped = defaultdict(dict)
    for row in records:
        if row["series"] == "transfer":
            grouped[(row["request_id"], row["repeat"])][row["arm"]] = row
    pairs = []
    for (request_id, repeat), arms in grouped.items():
        edge = arms["image_boundary"]
        inside = arms["intra_image"]
        require(edge["prefix_blocks_128"] == inside["prefix_blocks_128"]
                and abs(edge["suffix_tokens"] - inside["suffix_tokens"]) == 1, "Not a matched-work pair")
        fields = ("suffix_tokens", "causal_attention_pairs_per_layer", "prefix_blocks_128",
                  "EC_duplicate_image_clones", "EC_duplicate_cloned_tensor_payload_bytes",
                  "EC_prefill_clone_operations", "EC_decode_clone_operations", "EC_prefill_clone_payload_bytes",
                  "EC_decode_clone_payload_bytes", "EC_decode_redirected_inline_images",
                  "EC_prefill_clone_sync_host_ms", "EC_decode_clone_sync_host_ms",
                  "nixl_D_counters", "NVLink_D_stage_device_delta", "NVLink_P_stage_device_delta",
                  "D_first_token_http_ms", "producer_buffer_storage_snapshot_bytes",
                  "producer_host_marker_lifetime_max_ms", "first_token_exact_to_unsplit")
        pairs.append({"request_id": request_id, "repeat": repeat,
                      "arms": {arm: {field: row[field] for field in fields} for arm, row in arms.items()}})
    return {"evidence": "ISOLATED_GPU_MEASUREMENT",
            "match": "One token difference, same original prompt and prefix KV block count, same engine and EC transport within each pair. Generic and block-aligned controls are independent of image geometry.",
            "payload_scope": "Observed whole-E tensor clone operations and payload, plus native NIXL telemetry. NVLink data deltas are device-wide isolated-stage observations, not attributed physical per-tensor bytes.",
            "timing_scope": "Transfer arms include extra device/counter snapshots outside each HTTP interval. Producer lifetime uses host ready/release markers, not exact image GPU-ready time. No per-image encoder time.",
            "pairs": pairs}


def write_records(path, records):
    fields = sorted({k for row in records for k, v in row.items()
                     if not isinstance(v, (list, dict)) and k not in ("first_token",)})
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


async def forward_diagnostic(session, run, source, cost_cells, ec_path):
    """Sample actual D forward geometry apart from uninstrumented cost trials."""
    from ..mllm_epd_prefill_heavy_token_split.batch_token_correctness_control import post

    name = "dynamic_suffix_forward_diagnostic"
    geometries = sorted((row for row in cost_cells if row["arm"] == "ratio_0.3"),
                        key=lambda row: row["suffix_tokens"])
    selected = [geometries[0], geometries[len(geometries) // 2], *geometries[-2:]]
    start = await post(session, 18842, "/campaign/gpu_enclosures/begin", {"name": name})
    observed = []
    try:
        for index, case in enumerate(selected):
            record = await trial(session, source[case["index"]], case, f"{name}_{index:02d}", ec_path)
            observed.append(record)
    finally:
        end = await post(session, 18842, "/campaign/gpu_enclosures/end", {})
        write(run / "state/suffix_forward_observer.json", {"begin": start, "end": end})
    path = run / "raw/gpu_enclosures" / name / "decode"
    with (path / "forward_enclosures.csv").open(newline="") as stream:
        timings = list(csv.DictReader(stream))
    geometry = read(path / "geometry.json")["rows"]
    result = []
    for item in observed:
        entries = []
        for shape, timing in zip(geometry, timings, strict=True):
            count = sum(tokens for request, tokens in shape["requests"].items() if item["trial"] in request)
            if count:
                entries.append({"suffix_query_tokens": count, "CUDA_stream_enclosure_ms": float(timing["GPU_enclosure_ms"])})
        require(sum(row["suffix_query_tokens"] for row in entries) == item["suffix_tokens"],
                f"Observed D scheduled query tokens differ from isolated suffix for {item['trial']}")
        result.append({"request_id": item["request_id"], "input_tokens": item["input_tokens"],
                       "suffix_tokens": item["suffix_tokens"], "actual_D_forward_count": len(entries),
                       "forwards": entries})
    write(run / "analysis/suffix_forward_diagnostic.json", {
        "evidence": "DIAGNOSTIC", "cases": result,
        "scope": "Separate D CUDA stream enclosures and scheduled query tokens. No kernel-exclusive timing or observer qualification.",
    })


async def measure(run, calibration_manifest, main_manifest, ec_path, device_indices=(4, 5, 6), repeats=2,
                  *, inline_redirect=False):
    """Call within an isolated owned Common token=True worker epoch on fixed ports."""
    import aiohttp
    from ..mllm_epd_prefill_heavy_token_split.batch_token_correctness_control import post

    run, ec_path = Path(run), Path(ec_path)
    seed_inventory = inventory(main_manifest, calibration_manifest)
    write(run / "analysis/modality_inventory.json", seed_inventory)
    cells = seed_inventory["plan"]["cells"]
    require(repeats > 0, "Expected at least one repeat per fixed cell")
    require(len({cell["request_id"] for cell in cells if cell["series"] == "cost"}) >= 10, "Incomplete cost plan")
    split = {series: [cell for cell in cells if cell["series"] == series] for series in ("cost", "transfer")}
    source = read(calibration_manifest)["requests"]
    records = []
    baseline_ids = defaultdict(list)
    transport = INLINE_REDIRECT if inline_redirect else ON_DEMAND
    timeout = aiohttp.ClientTimeout(total=1800)
    journal_path = run / "raw/suffix_calibration_trials.jsonl"
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        await post(session, 18840, "/campaign/ec_ipc/inline_policy", {"enabled": inline_redirect})
        large = max(split["cost"], key=lambda cell: cell["suffix_tokens"])
        full = next(cell for cell in split["cost"] if cell["index"] == large["index"] and not cell["suffix_tokens"])
        short = min((cell for cell in split["cost"] if cell["arm"] == "ratio_0.1"), key=lambda cell: cell["suffix_tokens"])
        warmup = []
        for index, cell in enumerate((short, large, full)):
            item = await trial(session, source[cell["index"]], cell, f"dynamic_calibration_warmup_{index}", ec_path)
            warmup.append({key: item[key] for key in ("trial", "request_id", "suffix_tokens", "D_first_token_http_ms")})
        write(run / "state/suffix_warmup.json", {"warmup": warmup, "included_in_cost_fit": False})
        with journal_path.open("x", buffering=1) as journal, nvlink(device_indices) as sample:
            for series, group in split.items():
                for repeat in range(repeats):
                    schedule = random.Random(SEED + repeat).sample(group, len(group))
                    for ordinal, cell in enumerate(schedule):
                        name = f"dynamic_{series}_{repeat}_{ordinal:03d}"
                        value = await trial(session, source[cell["index"]], cell, name, ec_path,
                                            sample if series == "transfer" else None)
                        value["repeat"] = repeat
                        value["transport"] = transport
                        records.append(value)
                        journal.write(json.dumps(value, ensure_ascii=False) + "\n")
                        if cell["suffix_tokens"] == 0:
                            baseline_ids[cell["request_id"]].append(value["first_token"])
                    print(f"{series}: completed repeat {repeat + 1}/{repeats} with {len(group)} cells", flush=True)

    add_observed_clones(run, records)
    redirected_images = sum(row["EC_decode_redirected_inline_images"] for row in records)
    require(bool(redirected_images) == inline_redirect, "Observed D descriptor transport differs from selected mode")
    for row in records:
        ids = baseline_ids[row["request_id"]]
        stable = len(set(ids)) == 1
        row["first_token_exact_to_unsplit"] = row["first_token"] == ids[0] if stable else None
        row["first_token_reference_stable"] = stable
    write_records(run / "analysis/suffix_calibration.csv", records)
    write(run / "analysis/suffix_cost_model.json", fit_predictors(records, transport=transport))
    write(run / "analysis/modality_matched_transfer.json", transfer_pairs(records) | {"transport": transport})
    write(run / "analysis/suffix_calibration_receipt.json", {
        "evidence": "ISOLATED_GPU_MEASUREMENT", "samples": len(records), "repeats": repeats, "seed": SEED,
        "transport": transport, "decode_redirected_inline_images": redirected_images,
        "first_token_only": True, "full28_KV_and_natural_output_correctness": "NOT_MEASURED by this timing arm",
        "per_image_encoder_execution_and_gpu_ready": "NOT_AVAILABLE", "exclusive_suffix_kernel_time": "NOT_MEASURED",
        "transfer_metric_scope": "Observed EC clone operation counts and BF16 payload. Native NIXL transfer counters when exported. Device-wide NVLink deltas in isolation.",
        "raw_trials_sha256": sha256(journal_path.read_bytes()).hexdigest(),
    })
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        try:
            await forward_diagnostic(session, run, source, split["cost"], ec_path)
        except (ValueError, FileNotFoundError, RuntimeError, AssertionError, aiohttp.ClientError) as error:
            write(run / "analysis/suffix_forward_diagnostic_failure.json", {
                "evidence": "FAILED_EXECUTION", "error": str(error),
                "scope": "Optional geometry observer only. Saved uninstrumented cost trials remain independent.",
            })
