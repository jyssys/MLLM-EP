"""Measure paired inline Native and same-cut whole/slice costs in one H200 epoch."""
import argparse
from collections import defaultdict
from functools import partial
import json
from pathlib import Path
import random

import aiohttp

from ..mllm_epd_h200_modality_aware_dynamic_split.calibration import INLINE_REDIRECT, trial, write_records
from ..mllm_epd_h200_modality_aware_dynamic_split.calibration_main import INVARIANCE_ROLES, run_calibration
from ..mllm_epd_h200_modality_aware_dynamic_split.modality import cut_features, matched_cuts, request_geometry
from .cut_slack import RATIOS, TOKEN_TARGETS, read_jsonl
from .embedding_rows import SLICE_KEY


FIT = (2, 7, 13, 14, 19, 21)
VALIDATION = (1, 6, 20, 29, 31)
SMALL_TOKENS = (19, 64, 128, 256, 512)


def geometry(rows, compact=False):
    cells = []
    for cohort, indices in (("fit", FIT), ("validation", VALIDATION)):
        for index in indices:
            row = rows[index]
            length = row["total_input_tokens"]
            shape = request_geometry(row)
            # Keep the campaign ratios, and bracket the small whole/slice costs
            # within both short-image and longer blended-context strata.
            ratios = (0, .05, .10, .20) if compact else RATIOS
            suffixes = {round(length * ratio): f"ratio_{ratio:g}" for ratio in ratios}
            for suffix in SMALL_TOKENS:
                if 0 < suffix <= .35 * length:
                    suffixes.setdefault(suffix, f"tokens_{suffix}")
            if not compact:
                for suffix in TOKEN_TARGETS[-3:]:
                    if suffix <= .35 * length:
                        suffixes.setdefault(suffix, f"tokens_{suffix}")
            matched = matched_cuts(shape) if not compact else None
            if matched:
                for control in ("image_boundary", "intra_image"):
                    suffixes.setdefault(length - matched[control], control)
            for suffix, arm in sorted(suffixes.items()):
                expected_images = sum(image["span"][1] > length - suffix for image in shape["images"])
                cells.append({"index": index, "request_id": row["request_id"], "series": "cost",
                              "fit_group": cohort, "arm": arm, "D_expected_suffix_images": expected_images,
                              **cut_features(shape, length - suffix)})
    return cells


def annotate_transport(run, records):
    owner = {}
    for record in records:
        for key in record["image_keys"]:
            if key in owner and owner[key] != record["trial"]:
                raise ValueError("Cost calibration reused an image key across independent requests")
            owner[key] = record["trial"]
    observations = defaultdict(list)
    for path in (run / "raw").glob("ec_cuda_ipc_decode*.jsonl"):
        for event in read_jsonl(path):
            if not event["keys"] or event["keys"][0] not in owner:
                continue
            trial_id = owner[event["keys"][0]]
            if any(owner.get(key) != trial_id for key in event["keys"]):
                raise ValueError("Decoder transport did not preserve isolated trial ownership")
            observations[trial_id].append(event)
    for row in records:
        events = observations[row["trial"]]
        count = sum(event["images"] for event in events)
        actual_rpc = sum(event["export_RPC_ms"] for event in events)
        actual_spares = sum(event.get("preissued_crossing_images", 0) for event in events)
        if not row["suffix_tokens"]:
            route = "NATIVE_NO_SUFFIX"
        elif count != row["D_expected_suffix_images"]:
            route = "UNCONFIRMED_D_TRANSFER"
        elif actual_rpc > 0.:
            route = "RPC_FALLBACK"
        elif actual_spares < row["cross_boundary_image_count"]:
            route = "UNCONFIRMED_SPARE_CLAIM"
        else:
            route = "NO_RPC_VERIFIED"
        row.update(D_transport_route=route, D_EC_export_RPC_ms=actual_rpc,
                   D_EC_spares_claimed=actual_spares, D_EC_images_observed=count)


async def measure(run, calibration, main_manifest, ec, *, device_indices, repeats, inline_redirect, regime,
                  compact, preissue):
    from ..mllm_epd_prefill_heavy_token_split.batch_token_correctness_control import post

    if not inline_redirect:
        raise ValueError("A shared inline path is required for suffix cost calibration")
    rows = json.loads(calibration.read_text())["requests"]
    cells = geometry(rows, compact)
    out = {"whole": [], "slice": []}
    timeout = aiohttp.ClientTimeout(total=240.)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        await post(session, 18840, "/campaign/ec_ipc/inline_policy", {"enabled": True})
        # Warm a long prefix and short suffix plus the Native first-token path.
        for step, cell in enumerate((cells[-1], cells[0], cells[1])):
            await trial(session, rows[cell["index"]], cell, f"recovery_cost_warm_{step}", ec,
                        decode_params={SLICE_KEY: bool(step % 2)})
        with (run / "raw" / "split_cost_trials.jsonl").open("x", buffering=1) as journal:
            for repeat in range(repeats):
                rng = random.Random(2026112607 + repeat)
                for ordinal, cell in enumerate(rng.sample(cells, len(cells))):
                    modes = ["whole", "slice"] if cell["suffix_tokens"] else ["whole"]
                    rng.shuffle(modes)
                    for mode in modes:
                        name = f"recovery_cost_{repeat}_{ordinal}_{mode}"
                        row = await trial(session, rows[cell["index"]], cell, name, ec,
                                          decode_params={SLICE_KEY: mode == "slice"})
                        row.update(transport=INLINE_REDIRECT, transport_variant=mode, repeat=repeat,
                                   numeric_regime=regime, preissue_crossing=preissue)
                        journal.write(json.dumps(row) + "\n")
                        out[mode].append(row)
                        if not cell["suffix_tokens"]:
                            # Both policies share the actual Native baseline path.
                            out["slice"].append(dict(row, transport_variant="slice"))
                print(f"Completed isolated cost repeat {repeat + 1}/{repeats} for {len(cells)} geometries", flush=True)
    for mode, records in out.items():
        if preissue:
            annotate_transport(run, records)
        write_records(run / "analysis" / f"split_cost_{mode}.csv", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--numeric-regime", choices=tuple(INVARIANCE_ROLES), required=True)
    parser.add_argument("--compact", action="store_true", help="Measure 0/5/10/20% and n19/64/128/256/512")
    parser.add_argument("--preissue-crossing-ec", action="store_true",
                        help="Calibrate the verified spare-descriptor Inline path")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--wall-seconds", type=int, default=1600)
    parser.add_argument("--all-burners-off", action="store_true")
    args = parser.parse_args()
    args.inline_redirect = True
    run_calibration(args, measure_epoch=partial(measure, regime=args.numeric_regime, compact=args.compact,
                                               preissue=args.preissue_crossing_ec),
                    decode_environment={"MLLM_EPD_SLICE_AWARE_EC": "1"},
                    tag="split-cost", freeze_auxiliary=True, numeric_regime=args.numeric_regime,
                    preissue_crossing=args.preissue_crossing_ec)


if __name__ == "__main__":
    main()
